"""Ingestion by manual upload: a transcript (.txt, .vtt) or a recording (.mp3, .m4a)."""

import hashlib
import logging
import re
from datetime import UTC, date, datetime
from pathlib import PurePosixPath
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, Response, UploadFile

from app.api.deps import JobStoreDep, PipelineDep, SettingsDep
from app.schemas.extraction import MeetingContext
from app.schemas.jobs import Job, JobAccepted, TranscriptSource
from app.schemas.transcription import SpeechHints, SpokenLanguage
from app.services.audio import AUDIO_EXTENSIONS, AudioTooLarge, looks_like_audio, save_audio_upload
from app.services.transcripts import ALLOWED_EXTENSIONS, TranscriptError, load_transcript, safe_filename

logger = logging.getLogger(__name__)
router = APIRouter(tags=["transcripts"])

MAX_PARTICIPANTS = 30


@router.post(
    "/transcripts",
    response_model=JobAccepted,
    status_code=202,
    summary="Upload a meeting transcript or recording",
    responses={
        200: {"description": "This file was already processed; the earlier job is returned"},
        401: {"description": "Missing or invalid API key"},
        413: {"description": "File larger than the limit for its kind"},
        415: {"description": "Not a .txt, .vtt, .mp3 or .m4a file"},
        422: {"description": "Empty or unreadable content, or a file that is not what its name says"},
        503: {"description": "The LLM, or speech-to-text for a recording, is not configured"},
    },
)
async def upload_transcript(
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    settings: SettingsDep,
    store: JobStoreDep,
    pipeline: PipelineDep,
    file: Annotated[
        UploadFile, File(description="A transcript (.txt, or .vtt from Zoom) or a recording (.mp3, .m4a)")
    ],
    meeting_title: Annotated[str | None, Form(max_length=200)] = None,
    meeting_date: Annotated[
        date | None, Form(description="Day of the meeting (YYYY-MM-DD), used to resolve 'by Friday'")
    ] = None,
    participants: Annotated[
        str | None,
        Form(max_length=1000, description="Names of the people in the meeting, separated by commas"),
    ] = None,
    spoken_language: Annotated[
        SpokenLanguage | None, Form(description="Recordings only: what is spoken. Default: the server's setting")
    ] = None,
    force: Annotated[bool, Form(description="Process again even if this file was already processed")] = False,
) -> JobAccepted:
    """Validate the file, register a job and answer straight away with HTTP 202.

    The slow work (speech-to-text, LLM extraction, Notion writes) runs in a
    background task, so the caller never waits for it. Poll `status_url` to follow the job.
    """
    if not pipeline.can_extract:
        raise HTTPException(
            status_code=503, detail="The LLM is not configured on this server: set ANTHROPIC_API_KEY."
        )

    filename = safe_filename(file.filename)
    extension = PurePosixPath(filename).suffix.lower()
    is_audio = extension in AUDIO_EXTENSIONS
    if not is_audio and extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Supported files: .txt and .vtt transcripts, .mp3 and .m4a recordings.",
        )
    if is_audio and not pipeline.can_transcribe:
        raise HTTPException(
            status_code=503,
            detail="Speech-to-text is not configured on this server: set OPENAI_API_KEY, or STT_PROVIDER=local.",
        )

    names = _split_names(participants)
    title = (meeting_title or "").strip() or None
    now = datetime.now(UTC)

    if is_audio:
        try:
            audio = await save_audio_upload(file, extension, settings.max_audio_upload_bytes)
        except AudioTooLarge:
            megabytes = settings.max_audio_upload_bytes / 1_000_000
            raise HTTPException(
                status_code=413,
                detail=f"The recording is larger than the {megabytes:.0f} MB limit. Compress it "
                "(mono at 32 kbps holds about 100 minutes in 25 MB) or split it.",
            ) from None
        if not looks_like_audio(extension, audio.head):
            audio.path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=422, detail=f"The file is not a valid {extension} recording."
            )

        dedupe_key = f"upload:{audio.sha256}"
        earlier = None if force else await store.find_duplicate(dedupe_key)
        if earlier is not None:
            audio.path.unlink(missing_ok=True)
            response.status_code = 200
            return _accepted(request, response, earlier, duplicate=True)

        language = spoken_language or settings.stt_default_language
        job = Job(
            id=uuid4().hex,
            source=TranscriptSource.UPLOAD,
            media="audio",
            filename=filename,
            meeting_title=title,
            meeting_date=meeting_date,
            size_bytes=audio.size_bytes,
            sha256=audio.sha256,
            created_at=now,
            updated_at=now,
        )
        await store.add(job, dedupe_key=dedupe_key)
        # Participants' names help twice: the speech engine spells them right,
        # and the LLM can tell who a task belongs to.
        hints = SpeechHints(language=language, keywords=(*names, *settings.stt_keyword_list))
        context = MeetingContext(
            title=title, meeting_date=meeting_date, participants=names, spoken_language=language
        )
        background_tasks.add_task(pipeline.run_audio, job.id, audio.path, filename, hints, context)
        logger.info("Job %s queued: recording=%r size=%d bytes language=%s", job.id, filename, job.size_bytes, language)
        return _accepted(request, response, job)

    # Read at most one byte past the limit: enough to detect an oversized file
    # without loading it whole.
    raw = await file.read(settings.max_upload_bytes + 1)
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"The transcript is larger than the {settings.max_upload_bytes:,} byte limit.",
        )

    try:
        transcript = load_transcript(raw)
    except TranscriptError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Uploading the same file twice would create every Notion task twice, so the
    # earlier job is returned instead, unless the caller insists.
    digest = hashlib.sha256(raw).hexdigest()
    dedupe_key = f"upload:{digest}"
    if not force:
        earlier = await store.find_duplicate(dedupe_key)
        if earlier is not None:
            response.status_code = 200
            return _accepted(request, response, earlier, duplicate=True)

    job = Job(
        id=uuid4().hex,
        source=TranscriptSource.UPLOAD,
        filename=filename,
        meeting_title=title,
        meeting_date=meeting_date,
        size_bytes=len(raw),
        sha256=digest,
        created_at=now,
        updated_at=now,
    )
    await store.add(job, dedupe_key=dedupe_key)
    # The task gets the decoded text rather than the UploadFile, so it depends on
    # nothing that belongs to this request.
    context = MeetingContext(title=title, meeting_date=meeting_date, participants=names)
    background_tasks.add_task(pipeline.run, job.id, transcript, context)
    logger.info("Job %s queued: file=%r size=%d bytes", job.id, filename, job.size_bytes)
    return _accepted(request, response, job)


def _split_names(participants: str | None) -> tuple[str, ...]:
    """ "Amira, Youssef; Sarah" -> ("Amira", "Youssef", "Sarah"), without blanks or repeats."""
    names: list[str] = []
    for name in re.split(r"[,;\n]", participants or ""):
        name = " ".join(name.split())[:60]
        if name and name not in names:
            names.append(name)
    return tuple(names[:MAX_PARTICIPANTS])


def _accepted(request: Request, response: Response, job: Job, *, duplicate: bool = False) -> JobAccepted:
    status_url = str(request.app.url_path_for("get_job", job_id=job.id))
    response.headers["Location"] = status_url
    return JobAccepted(job_id=job.id, status=job.status, status_url=status_url, duplicate=duplicate)
