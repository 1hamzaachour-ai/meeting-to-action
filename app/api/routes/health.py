"""Liveness endpoint, used by the hosting platform's health check and by the upload page."""

from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel

from app import __version__
from app.api.deps import SettingsDep

router = APIRouter(tags=["health"])


class Integrations(BaseModel):
    """Which optional parts are configured. Says nothing about whether the keys are valid."""

    llm: bool
    speech_to_text: bool
    notion: bool
    zoom_webhook: bool


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    app: str
    version: str
    environment: str
    auth_required: bool
    integrations: Integrations


@router.get("/health", response_model=HealthResponse, summary="Check that the service is up")
async def health(request: Request, settings: SettingsDep) -> HealthResponse:
    return HealthResponse(
        app=settings.app_name,
        version=__version__,
        environment=settings.environment,
        auth_required=settings.api_key is not None,
        integrations=Integrations(
            llm=request.app.state.pipeline.can_extract,
            speech_to_text=request.app.state.pipeline.can_transcribe,
            notion=settings.notion_configured,
            zoom_webhook=settings.zoom_webhook_secret_token is not None,
        ),
    )
