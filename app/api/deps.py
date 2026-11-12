"""Request dependencies.

The shared objects are created once at startup (see `lifespan` in app/main.py) and
stored on `app.state`; these helpers hand them to the route functions.
"""

import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from app.config import Settings
from app.services.job_store import JobStore
from app.services.pipeline import TranscriptPipeline

# Declared this way (rather than as a plain header) so the docs page at /docs
# gets an "Authorize" button.
api_key_header = APIKeyHeader(
    name="X-API-Key", auto_error=False, description="Required when the server sets API_KEY."
)


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_job_store(request: Request) -> JobStore:
    return request.app.state.job_store


def get_pipeline(request: Request) -> TranscriptPipeline:
    return request.app.state.pipeline


SettingsDep = Annotated[Settings, Depends(get_settings)]
JobStoreDep = Annotated[JobStore, Depends(get_job_store)]
PipelineDep = Annotated[TranscriptPipeline, Depends(get_pipeline)]


def require_api_key(
    settings: SettingsDep, provided: Annotated[str | None, Security(api_key_header)] = None
) -> None:
    """Reject the request unless it carries the server's API key (when one is set)."""
    if settings.api_key is None:
        return
    expected = settings.api_key.get_secret_value()
    # compare_digest takes the same time whatever the input, so response timing
    # reveals nothing about the key.
    if provided is None or not secrets.compare_digest(provided.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")
