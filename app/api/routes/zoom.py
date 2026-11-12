"""Zoom webhook: starts a job when a meeting's transcript is ready.

Zoom requires an answer within 3 seconds, so this endpoint only verifies the
request and queues the work; downloading and processing happen in the background.
"""

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import ValidationError

from app.api.deps import JobStoreDep, PipelineDep, SettingsDep
from app.schemas.extraction import MeetingContext
from app.schemas.jobs import Job, TranscriptSource
from app.schemas.zoom import ZoomEvent
from app.services.zoom import find_transcript_file, is_from_zoom, meeting_date, url_validation_response

logger = logging.getLogger(__name__)
router = APIRouter(tags=["zoom"])

URL_VALIDATION = "endpoint.url_validation"
TRANSCRIPT_COMPLETED = "recording.transcript_completed"


@router.post(
    "/webhooks/zoom",
    summary="Receive Zoom webhook events",
    responses={
        400: {"description": "The body is not a Zoom event"},
        401: {"description": "Missing, invalid or expired signature"},
        503: {"description": "Zoom webhooks or the LLM are not configured"},
    },
)
async def zoom_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    settings: SettingsDep,
    store: JobStoreDep,
    pipeline: PipelineDep,
) -> dict[str, Any]:
    if settings.zoom_webhook_secret_token is None:
        raise HTTPException(
            status_code=503, detail="Zoom webhooks are not configured: set ZOOM_WEBHOOK_SECRET_TOKEN."
        )
    secret = settings.zoom_webhook_secret_token.get_secret_value()

    # The signature covers the exact bytes Zoom sent, so the raw body is read here
    # instead of letting FastAPI parse it.
    body = await request.body()
    if not is_from_zoom(
        secret=secret,
        timestamp=request.headers.get("x-zm-request-timestamp"),
        signature=request.headers.get("x-zm-signature"),
        body=body,
        tolerance_seconds=settings.zoom_timestamp_tolerance_seconds,
    ):
        raise HTTPException(status_code=401, detail="Invalid Zoom signature.")

    try:
        event = ZoomEvent.model_validate_json(body)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail="The body is not a Zoom event.") from exc

    # Zoom sends this when the endpoint is registered, then every 72 hours.
    if event.event == URL_VALIDATION:
        if not event.payload.plainToken:
            raise HTTPException(status_code=400, detail="The validation event has no plainToken.")
        return url_validation_response(secret, event.payload.plainToken)

    if event.event != TRANSCRIPT_COMPLETED or event.payload.object is None:
        return {"status": "ignored", "reason": f"unhandled event {event.event!r}"}

    recording = event.payload.object
    transcript_file = find_transcript_file(recording)
    if transcript_file is None or transcript_file.download_url is None:
        return {"status": "ignored", "reason": "the event lists no transcript file"}

    # Zoom delivers an event again when it gets no answer in time. The meeting's
    # uuid recognises a repeat, so the tasks are not created twice.
    meeting_ref = recording.uuid or recording.id
    dedupe_key = f"zoom:{meeting_ref}" if meeting_ref else None
    earlier = await store.find_duplicate(dedupe_key) if dedupe_key else None
    if earlier is not None:
        return {"status": "duplicate", "job_id": earlier.id}

    if not pipeline.can_extract:
        # A 5xx answer makes Zoom retry later, which leaves time to fix the setup.
        raise HTTPException(
            status_code=503, detail="The LLM is not configured on this server: set ANTHROPIC_API_KEY."
        )

    now = datetime.now(UTC)
    job = Job(
        id=uuid4().hex,
        source=TranscriptSource.ZOOM,
        meeting_title=(recording.topic or "").strip() or None,
        meeting_date=meeting_date(recording),
        created_at=now,
        updated_at=now,
    )
    await store.add(job, dedupe_key=dedupe_key)
    context = MeetingContext(title=job.meeting_title, meeting_date=job.meeting_date)
    background_tasks.add_task(
        pipeline.run_zoom_recording, job.id, transcript_file.download_url, event.download_token, context
    )
    logger.info("Job %s queued from Zoom meeting %r", job.id, recording.id)
    return {"status": "accepted", "job_id": job.id}
