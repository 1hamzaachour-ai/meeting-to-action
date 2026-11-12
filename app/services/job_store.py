"""Where job records live.

`JobStore` is the interface the rest of the app depends on. This file ships an
in-memory implementation; a Redis or Postgres one can replace it later without
touching the routes or the pipeline.
"""

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any

from app.schemas.jobs import FINISHED_STATUSES, Job, JobStatus


class JobStore(ABC):
    @abstractmethod
    async def add(self, job: Job, dedupe_key: str | None = None) -> None:
        """Save a new job. `dedupe_key` identifies its input (file hash, Zoom meeting)."""

    @abstractmethod
    async def get(self, job_id: str) -> Job | None:
        """Return the job, or None when the id is unknown."""

    @abstractmethod
    async def update(self, job_id: str, **changes: Any) -> Job:
        """Apply field changes to an existing job and return the updated record."""

    @abstractmethod
    async def find_duplicate(self, dedupe_key: str) -> Job | None:
        """Return the job already handling this input, unless it failed (then a retry is fine)."""


class InMemoryJobStore(JobStore):
    """Jobs kept in a dict inside this process.

    Two limits follow from that:
    - jobs disappear when the server restarts;
    - they are not shared between processes, so run a single uvicorn worker.

    Every call happens on the event loop thread, so no lock is needed.
    """

    def __init__(self, max_jobs: int = 500) -> None:
        self._jobs: dict[str, Job] = {}
        self._job_id_by_key: dict[str, str] = {}
        self._max_jobs = max_jobs

    async def add(self, job: Job, dedupe_key: str | None = None) -> None:
        self._jobs[job.id] = job
        if dedupe_key is not None:
            self._job_id_by_key[dedupe_key] = job.id  # the newest job for an input wins
        self._drop_oldest_finished_jobs()

    async def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    async def update(self, job_id: str, **changes: Any) -> Job:
        job = self._jobs[job_id]
        updated = job.model_copy(update={**changes, "updated_at": datetime.now(UTC)})
        self._jobs[job_id] = updated
        return updated

    async def find_duplicate(self, dedupe_key: str) -> Job | None:
        job = self._jobs.get(self._job_id_by_key.get(dedupe_key, ""))
        if job is None or job.status is JobStatus.FAILED:
            return None
        return job

    def _drop_oldest_finished_jobs(self) -> None:
        """Keep memory bounded. Jobs still queued or processing are never dropped."""
        overflow = len(self._jobs) - self._max_jobs
        if overflow <= 0:
            return
        # A dict keeps insertion order, so the first matches are the oldest jobs.
        finished = [job.id for job in self._jobs.values() if job.status in FINISHED_STATUSES]
        for job_id in finished[:overflow]:
            del self._jobs[job_id]
        self._job_id_by_key = {
            key: job_id for key, job_id in self._job_id_by_key.items() if job_id in self._jobs
        }
