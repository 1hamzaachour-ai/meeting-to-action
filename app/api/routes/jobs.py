"""Job status polling."""

from fastapi import APIRouter, HTTPException

from app.api.deps import JobStoreDep
from app.schemas.jobs import Job

router = APIRouter(tags=["jobs"])


@router.get(
    "/jobs/{job_id}",
    response_model=Job,
    name="get_job",
    summary="Get the status and result of a job",
    responses={404: {"description": "Unknown job id"}},
)
async def get_job(job_id: str, store: JobStoreDep) -> Job:
    job = await store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job
