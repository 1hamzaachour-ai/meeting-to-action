"""Job models: the processing record for one transcript, and its API shapes."""

from datetime import date, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel

from app.schemas.extraction import VerifiedActionItem
from app.schemas.transcription import Transcription


class JobStatus(StrEnum):
    QUEUED = "queued"  # accepted, waiting for a free processing slot
    PROCESSING = "processing"  # the pipeline is running
    COMPLETED = "completed"  # finished successfully, `result` is set
    FAILED = "failed"  # finished with an error, `error` is set (`result` may hold partial work)


FINISHED_STATUSES = frozenset({JobStatus.COMPLETED, JobStatus.FAILED})


class TranscriptSource(StrEnum):
    UPLOAD = "upload"
    ZOOM = "zoom"


class TranscriptStats(BaseModel):
    characters: int
    words: int
    lines: int


class LLMUsage(BaseModel):
    model: str
    input_tokens: int
    output_tokens: int


class NotionTask(BaseModel):
    """Outcome for one action item: either a created page or the reason it failed."""

    title: str
    page_id: str | None = None
    url: str | None = None
    error: str | None = None


class NotionSync(BaseModel):
    # created: every task was written; partial: some failed; failed: none written;
    # skipped: Notion is not configured or there was nothing to write.
    status: Literal["created", "partial", "failed", "skipped"]
    detail: str | None = None
    tasks: list[NotionTask] = []


class JobResult(BaseModel):
    """What the pipeline produced."""

    transcript: TranscriptStats
    # Only for recordings: the text the speech engine produced, so it can be checked.
    transcription: Transcription | None = None
    meeting_title: str
    summary: str
    action_items: list[VerifiedActionItem]
    llm: LLMUsage
    notion: NotionSync


class Job(BaseModel):
    """One transcript moving through the pipeline. Returned by GET /api/v1/jobs/{id}.

    The transcript text is deliberately not stored here: it goes straight to the
    background task and is dropped when processing ends.
    """

    id: str
    status: JobStatus = JobStatus.QUEUED
    # What the pipeline is doing right now, while the status is "processing".
    stage: Literal["transcribing", "extracting", "writing to Notion"] | None = None
    source: TranscriptSource
    media: Literal["text", "audio"] = "text"
    filename: str | None = None
    meeting_title: str | None = None
    meeting_date: date | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    created_at: datetime
    updated_at: datetime
    result: JobResult | None = None
    error: str | None = None


class JobAccepted(BaseModel):
    """Response to an upload: HTTP 202 for a new job, 200 for a duplicate."""

    job_id: str
    status: JobStatus
    status_url: str
    # True when the same transcript was already processed and that job is returned instead.
    duplicate: bool = False
