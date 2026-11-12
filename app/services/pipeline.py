"""The background pipeline: (audio -> text) -> action items (LLM) -> tasks (Notion).

One instance is shared by every request. Routes hand it a job through FastAPI's
BackgroundTasks, so the work starts after the HTTP response has been sent.
"""

import asyncio
import hashlib
import logging
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from app.errors import UserFacingError
from app.schemas.extraction import MeetingContext
from app.schemas.jobs import JobResult, JobStatus, LLMUsage, NotionSync
from app.schemas.transcription import SpeechHints, Transcription
from app.services.extraction import ActionItemExtractor, ExtractionError, verify_quotes
from app.services.job_store import JobStore
from app.services.notion import NotionTaskWriter
from app.services.transcription import Transcriber, TranscriptionError
from app.services.transcripts import load_transcript, transcript_stats
from app.services.zoom import ZoomClient

logger = logging.getLogger(__name__)

# A loader produces the transcript text, plus the transcription when it came from audio.
TranscriptLoader = Callable[[], Awaitable[tuple[str, Transcription | None]]]


class TranscriptPipeline:
    def __init__(
        self,
        store: JobStore,
        *,
        extractor: ActionItemExtractor | None,
        notion: NotionTaskWriter | None,
        zoom: ZoomClient,
        max_concurrent_jobs: int,
        transcriber: Transcriber | None = None,
    ) -> None:
        self._store = store
        self._extractor = extractor
        self._notion = notion
        self._zoom = zoom
        self._transcriber = transcriber
        # Caps how many transcripts are processed at once, so a burst of uploads
        # cannot become a burst of LLM and Notion calls. Extra jobs stay "queued".
        self._slots = asyncio.Semaphore(max_concurrent_jobs)

    @property
    def can_extract(self) -> bool:
        return self._extractor is not None

    @property
    def can_transcribe(self) -> bool:
        return self._transcriber is not None

    async def run(self, job_id: str, transcript: str, context: MeetingContext) -> None:
        """Process a transcript that is already in hand (the text upload path)."""

        async def load() -> tuple[str, Transcription | None]:
            return transcript, None

        await self._execute(job_id, context, load)

    async def run_audio(
        self, job_id: str, audio: Path, filename: str, hints: SpeechHints, context: MeetingContext
    ) -> None:
        """Transcribe a recording, then process the text like any other transcript."""

        async def load() -> tuple[str, Transcription | None]:
            try:
                if self._transcriber is None:
                    raise TranscriptionError("Speech-to-text is not configured on this server.")
                await self._store.update(job_id, stage="transcribing")
                transcription = await self._transcriber.transcribe(audio, filename, hints)
            finally:
                audio.unlink(missing_ok=True)  # the recording is not kept once it has been read
            return transcription.text, transcription

        await self._execute(job_id, context, load)

    async def run_zoom_recording(
        self, job_id: str, download_url: str, download_token: str | None, context: MeetingContext
    ) -> None:
        """Download a Zoom transcript, then process it (the webhook path)."""

        async def load() -> tuple[str, Transcription | None]:
            raw = await self._zoom.download_transcript(download_url, download_token)
            await self._store.update(job_id, size_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            return load_transcript(raw), None

        await self._execute(job_id, context, load)

    async def _execute(self, job_id: str, context: MeetingContext, load: TranscriptLoader) -> None:
        """Run one job to a final status.

        This runs after the HTTP response has been sent, so nobody is left to receive
        an exception: every failure is caught here and recorded on the job instead.
        """
        async with self._slots:
            started = time.perf_counter()
            await self._store.update(job_id, status=JobStatus.PROCESSING)
            try:
                transcript, transcription = await load()
                result = await self._process(job_id, transcript, context, transcription)
            except UserFacingError as exc:  # expected: its message is meant for the user
                logger.warning("Job %s failed: %s", job_id, exc)
                await self._store.update(job_id, status=JobStatus.FAILED, stage=None, error=str(exc))
                return
            except Exception as exc:  # a bug: keep the details in the logs
                logger.exception("Job %s failed", job_id)
                await self._store.update(
                    job_id,
                    status=JobStatus.FAILED,
                    stage=None,
                    error=f"Processing failed ({type(exc).__name__}). See the server logs for details.",
                )
                return

            elapsed = time.perf_counter() - started
            if result.notion.status in ("partial", "failed"):
                # The extraction is kept on the job so nothing has to be recomputed by hand.
                error = f"Notion: {result.notion.detail}"
                await self._store.update(job_id, status=JobStatus.FAILED, stage=None, result=result, error=error)
                logger.warning("Job %s failed after %.1fs: %s", job_id, elapsed, error)
            else:
                await self._store.update(job_id, status=JobStatus.COMPLETED, stage=None, result=result)
                logger.info(
                    "Job %s completed in %.1fs: %d action items, Notion %s",
                    job_id,
                    elapsed,
                    len(result.action_items),
                    result.notion.status,
                )

    async def _process(
        self, job_id: str, transcript: str, context: MeetingContext, transcription: Transcription | None
    ) -> JobResult:
        if self._extractor is None:
            raise ExtractionError("The LLM is not configured on this server: set ANTHROPIC_API_KEY.")

        await self._store.update(job_id, stage="extracting")
        outcome = await self._extractor.extract(transcript, context)
        action_items = verify_quotes(outcome.extraction.action_items, transcript)
        meeting_title = context.title or outcome.extraction.meeting_title

        if self._notion is None:
            notion = NotionSync(status="skipped", detail="Notion is not configured on this server.")
        else:
            await self._store.update(job_id, stage="writing to Notion")
            notion = await self._notion.create_tasks(
                action_items, meeting_title=meeting_title, meeting_date=context.meeting_date
            )

        return JobResult(
            transcript=transcript_stats(transcript),
            transcription=transcription,
            meeting_title=meeting_title,
            summary=outcome.extraction.summary,
            action_items=action_items,
            llm=LLMUsage(
                model=outcome.model,
                input_tokens=outcome.input_tokens,
                output_tokens=outcome.output_tokens,
            ),
            notion=notion,
        )
