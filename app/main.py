"""Application factory and ASGI entry point.

Run locally with:  uvicorn app.main:app --reload
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx2
from fastapi import Depends, FastAPI

from app import __version__
from app.api.deps import require_api_key
from app.api.routes import health, jobs, transcripts, ui, zoom
from app.config import Settings
from app.logging_config import configure_logging
from app.middleware import MaxBodySizeMiddleware
from app.services.extraction import ActionItemExtractor
from app.services.job_store import InMemoryJobStore
from app.services.local_whisper import LocalWhisperTranscriber, missing_packages
from app.services.notion import NotionError, NotionTaskWriter
from app.services.pipeline import TranscriptPipeline
from app.services.transcription import OpenAITranscriber, Transcriber
from app.services.zoom import ZoomClient

logger = logging.getLogger(__name__)

# Room for the multipart envelope (boundaries, part headers, the form fields) on top
# of the transcript itself.
MULTIPART_OVERHEAD_BYTES = 64 * 1024


def create_app(
    settings: Settings | None = None,
    *,
    extractor: ActionItemExtractor | None = None,
    transcriber: Transcriber | None = None,
    http_transport: httpx2.AsyncBaseTransport | None = None,
) -> FastAPI:
    """Build the app.

    The keyword arguments exist for tests: `extractor` replaces the LLM,
    `transcriber` the speech engine, and `http_transport` the network for the
    Notion, Zoom and speech-to-text calls.
    """
    settings = settings or Settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Create the long-lived objects once at startup, and close them at shutdown."""
        active_extractor = extractor
        if active_extractor is None and settings.anthropic_api_key is not None:
            # Imported here rather than at the top of the file: the Anthropic SDK
            # takes seconds to load, a cost worth paying only when it is used.
            from app.services.anthropic_extractor import AnthropicExtractor

            active_extractor = AnthropicExtractor.from_settings(settings)

        # One HTTP client, with its connection pool, shared by the Notion, Zoom and
        # speech-to-text calls.
        http = httpx2.AsyncClient(timeout=httpx2.Timeout(30.0, connect=10.0), transport=http_transport)

        active_transcriber = transcriber
        if active_transcriber is None and settings.stt_provider == "local":
            active_transcriber = LocalWhisperTranscriber(settings.local_stt_model)
            if missing := missing_packages():
                logger.warning(
                    "STT_PROVIDER=local, but these packages are not installed: %s. "
                    "Run: pip install -r requirements-local-stt.txt",
                    ", ".join(missing),
                )
        elif active_transcriber is None and settings.openai_api_key is not None:
            active_transcriber = OpenAITranscriber(
                http,
                api_key=settings.openai_api_key.get_secret_value(),
                model=settings.stt_model,
                base_url=settings.stt_base_url,
                timeout_seconds=settings.stt_timeout_seconds,
            )

        notion: NotionTaskWriter | None = None
        if settings.notion_token is not None and settings.notion_database_id is not None:
            notion = NotionTaskWriter(
                http,
                token=settings.notion_token.get_secret_value(),
                database_id=settings.notion_database_id,
                data_source_id=settings.notion_data_source_id,
                base_url=settings.notion_base_url,
                owner_property=settings.notion_owner_property,
                due_property=settings.notion_due_property,
                priority_property=settings.notion_priority_property,
                meeting_property=settings.notion_meeting_property,
            )
            await _report_notion_setup(notion)

        zoom_client = ZoomClient(
            http,
            allowed_hosts=settings.zoom_download_host_list,
            max_bytes=settings.max_upload_bytes,
            account_id=settings.zoom_account_id,
            client_id=settings.zoom_client_id,
            client_secret=(
                settings.zoom_client_secret.get_secret_value() if settings.zoom_client_secret else None
            ),
            oauth_url=settings.zoom_oauth_url,
        )

        store = InMemoryJobStore(max_jobs=settings.max_jobs_in_memory)
        app.state.job_store = store
        app.state.pipeline = TranscriptPipeline(
            store,
            extractor=active_extractor,
            notion=notion,
            zoom=zoom_client,
            max_concurrent_jobs=settings.max_concurrent_jobs,
            transcriber=active_transcriber,
        )
        logger.info(
            "%s %s started (environment=%s, llm=%s, speech-to-text=%s, notion=%s, zoom webhook=%s, api key=%s)",
            settings.app_name,
            __version__,
            settings.environment,
            "on" if active_extractor else "off",
            "on" if active_transcriber else "off",
            "on" if notion else "off",
            "on" if settings.zoom_webhook_secret_token else "off",
            "required" if settings.api_key else "not required",
        )
        try:
            yield
        finally:
            await http.aclose()
            if active_extractor is not None:
                await active_extractor.aclose()
            logger.info("%s stopped", settings.app_name)

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description="Turns meeting transcripts into action items in Notion.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    # A coarse ceiling for any request; the upload route then applies the exact
    # limit for the kind of file it receives (a transcript or a recording).
    largest_upload = max(settings.max_upload_bytes, settings.max_audio_upload_bytes)
    app.add_middleware(MaxBodySizeMiddleware, max_bytes=largest_upload + MULTIPART_OVERHEAD_BYTES)

    protected = [Depends(require_api_key)]
    app.include_router(ui.router)
    app.include_router(health.router)
    app.include_router(zoom.router)  # authenticated by Zoom's signature instead of the API key
    app.include_router(transcripts.router, prefix="/api/v1", dependencies=protected)
    app.include_router(jobs.router, prefix="/api/v1", dependencies=protected)
    return app


async def _report_notion_setup(notion: NotionTaskWriter) -> None:
    """Check the Notion database at startup so a setup mistake shows up in the logs at once.

    It never blocks startup: Notion being unreachable for a moment must not take
    the whole service down. The check runs again when the first job needs it.
    """
    try:
        columns = await notion.check_database()
    except NotionError as exc:
        logger.warning("Notion is configured but not usable yet: %s", exc)
    else:
        logger.info("Notion database ready (title column %r)", columns.title)


app = create_app()
