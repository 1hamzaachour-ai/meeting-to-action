"""Unit tests for the pieces that do not need HTTP."""

import asyncio
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx2
import pytest

from app.schemas.extraction import MeetingContext
from app.schemas.jobs import Job, JobStatus, TranscriptSource
from app.services.job_store import InMemoryJobStore
from app.services.pipeline import TranscriptPipeline
from app.services.transcripts import TranscriptError, decode_transcript, load_transcript, safe_filename, vtt_to_text
from app.services.zoom import ZoomClient

# --- Startup cost ---


def test_app_starts_without_loading_the_anthropic_sdk():
    """The SDK takes seconds to import, so it must only load when a key is configured."""
    check = "import sys, app.main; sys.exit(1 if 'anthropic' in sys.modules else 0)"
    project_root = Path(__file__).resolve().parent.parent
    assert subprocess.run([sys.executable, "-c", check], cwd=project_root).returncode == 0


# --- Decoding and file names ---


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Réunion d'équipe".encode("utf-8"), "Réunion d'équipe"),
        ("Réunion d'équipe".encode("utf-8-sig"), "Réunion d'équipe"),  # UTF-8 with BOM
        ("Réunion d'équipe".encode("utf-16"), "Réunion d'équipe"),  # UTF-16 with BOM
        (b"\r\nline one\r\nline two\r\n", "line one\nline two"),  # Windows newlines, padding
    ],
)
def test_decode_transcript_returns_normalized_text(raw, expected):
    assert decode_transcript(raw) == expected


@pytest.mark.parametrize("raw", [b"", b" \n\t ", b"caf\xe9", b"h\x00i\x00"])
def test_decode_transcript_rejects_unusable_content(raw):
    with pytest.raises(TranscriptError):
        decode_transcript(raw)


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("notes.txt", "notes.txt"),
        ("../../etc/notes.txt", "notes.txt"),
        ("C:\\Users\\someone\\notes.txt", "notes.txt"),
        (None, ""),
        ("a" * 300 + ".txt", "a" * 251 + ".txt"),  # capped at 255, extension kept
    ],
)
def test_safe_filename(given, expected):
    assert safe_filename(given) == expected


# --- Zoom's caption format ---

ZOOM_VTT = """WEBVTT

1
00:00:00.000 --> 00:00:04.200
Sarah Chen: Morning everyone.
Three topics today.

NOTE this block is a comment

2
00:00:04.500 --> 00:00:09.000 align:start
<v Omar Haddad>I&#39;ll fix the <i>billing</i> bug.</v>
"""


def test_vtt_keeps_only_what_was_said():
    assert vtt_to_text(ZOOM_VTT) == (
        "Sarah Chen: Morning everyone. Three topics today.\nOmar Haddad: I'll fix the billing bug."
    )


def test_load_transcript_detects_vtt_by_its_header_not_its_name():
    assert load_transcript(ZOOM_VTT.encode()).startswith("Sarah Chen: Morning everyone.")
    assert load_transcript(b"Sarah: plain text stays as it is") == "Sarah: plain text stays as it is"


def test_vtt_without_captions_is_rejected():
    with pytest.raises(TranscriptError):
        vtt_to_text("WEBVTT\n\nNOTE nothing here\n")


# --- Job store ---


def make_job(job_id: str) -> Job:
    now = datetime.now(UTC)
    return Job(id=job_id, source=TranscriptSource.UPLOAD, created_at=now, updated_at=now)


def test_store_drops_the_oldest_finished_job_but_never_an_active_one():
    async def scenario() -> list[Job | None]:
        store = InMemoryJobStore(max_jobs=2)
        await store.add(make_job("still-queued"))
        await store.add(make_job("finished"), dedupe_key="file-a")
        await store.update("finished", status=JobStatus.COMPLETED)
        await store.add(make_job("newest"))  # one over capacity
        found = [await store.get(job_id) for job_id in ("still-queued", "finished", "newest")]
        return [*found, await store.find_duplicate("file-a")]

    still_queued, finished, newest, duplicate_of_dropped_job = asyncio.run(scenario())
    assert still_queued is not None
    assert finished is None
    assert newest is not None
    assert duplicate_of_dropped_job is None


def test_store_update_refreshes_the_timestamp():
    async def scenario() -> tuple[Job, Job]:
        store = InMemoryJobStore()
        job = make_job("job-1")
        await store.add(job)
        return job, await store.update("job-1", status=JobStatus.PROCESSING)

    before, after = asyncio.run(scenario())
    assert after.status is JobStatus.PROCESSING
    assert after.updated_at >= before.updated_at
    assert after.created_at == before.created_at


def test_store_finds_a_duplicate_unless_that_job_failed():
    async def scenario() -> tuple[Job | None, Job | None, Job | None]:
        store = InMemoryJobStore()
        await store.add(make_job("job-1"), dedupe_key="file-a")
        while_active = await store.find_duplicate("file-a")
        await store.update("job-1", status=JobStatus.FAILED)
        after_failure = await store.find_duplicate("file-a")
        return while_active, after_failure, await store.find_duplicate("file-b")

    while_active, after_failure, unknown = asyncio.run(scenario())
    assert while_active is not None and while_active.id == "job-1"
    assert after_failure is None
    assert unknown is None


# --- Pipeline concurrency ---


def test_pipeline_processes_a_limited_number_of_jobs_at_once(monkeypatch):
    async def scenario() -> tuple[list[JobStatus], list[JobStatus]]:
        store = InMemoryJobStore()
        async with httpx2.AsyncClient() as http:
            pipeline = TranscriptPipeline(
                store,
                extractor=None,
                notion=None,
                zoom=ZoomClient(http, allowed_hosts=[], max_bytes=1),
                max_concurrent_jobs=2,
            )
            release = asyncio.Event()

            async def held_process(*args):
                await release.wait()  # keeps the job "processing" until the test lets go
                raise RuntimeError("the outcome does not matter here")

            monkeypatch.setattr(pipeline, "_process", held_process)

            job_ids = [f"job-{number}" for number in range(5)]
            for job_id in job_ids:
                await store.add(make_job(job_id))
            tasks = [
                asyncio.create_task(pipeline.run(job_id, "hello world", MeetingContext())) for job_id in job_ids
            ]

            for _ in range(5):  # give every task the chance to run up to its waiting point
                await asyncio.sleep(0)
            while_held = [(await store.get(job_id)).status for job_id in job_ids]

            release.set()
            await asyncio.gather(*tasks)
            when_done = [(await store.get(job_id)).status for job_id in job_ids]
            return while_held, when_done

    while_held, when_done = asyncio.run(scenario())
    assert while_held.count(JobStatus.PROCESSING) == 2
    assert while_held.count(JobStatus.QUEUED) == 3
    assert when_done == [JobStatus.FAILED] * 5  # every job reached a final status
