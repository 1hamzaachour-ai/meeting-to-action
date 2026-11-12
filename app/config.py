"""Application settings, read from environment variables and an optional .env file."""

import re
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.schemas.transcription import SpokenLanguage

BASE_DIR = Path(__file__).resolve().parent.parent

_NOTION_ID_AT_END = re.compile(r"[0-9a-fA-F]{32}$")


class Settings(BaseSettings):
    """Each field can be overridden by an environment variable of the same name
    (case-insensitive), for example MAX_UPLOAD_BYTES=2097152.

    The three integrations are optional and independent: the app starts without
    them and reports what is missing instead of crashing.
    """

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,  # "KEY=" in .env means "not set"
        extra="ignore",
    )

    app_name: str = "Meeting-to-Action"
    environment: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # --- Access control ---
    # When set, the upload and job endpoints require the header "X-API-Key".
    api_key: SecretStr | None = None

    # --- Ingestion ---
    # Largest transcript accepted, in bytes. 1 MiB is about 150,000 words of plain
    # text, far more than any single meeting.
    max_upload_bytes: int = Field(default=1_048_576, gt=0)
    # How many transcripts are processed at the same time; the rest wait as "queued".
    max_concurrent_jobs: int = Field(default=3, gt=0)
    # How many jobs are kept for status polling before the oldest finished ones are dropped.
    max_jobs_in_memory: int = Field(default=500, gt=0)

    # --- LLM ---
    # "anthropic": Claude (billed). "groq": an open-weight model on Groq, which has a free plan.
    llm_provider: Literal["anthropic", "groq"] = "anthropic"
    groq_api_key: SecretStr | None = None
    groq_model: str = "openai/gpt-oss-120b"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_reasoning_effort: Literal["low", "medium", "high"] = "medium"
    anthropic_api_key: SecretStr | None = None
    anthropic_base_url: str | None = None  # only for a proxy or gateway
    llm_model: str = "claude-opus-5-5"
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    llm_max_tokens: int = Field(default=32_000, gt=0)
    llm_timeout_seconds: float = Field(default=600, gt=0)
    # Language of the tasks written to Notion, e.g. "French". Empty: the transcript's own.
    task_language: str | None = None

    # --- Speech-to-text ---
    # "openai": a hosted API (needs OPENAI_API_KEY). "local": a Hugging Face model on
    # this machine (needs requirements-local-stt.txt, and a GPU to be usable).
    stt_provider: Literal["openai", "local"] = "openai"
    openai_api_key: SecretStr | None = None
    stt_model: str = "gpt-transcribe"
    # Any server that implements OpenAI's transcription API can be used instead.
    stt_base_url: str = "https://api.openai.com/v1"
    stt_timeout_seconds: float = Field(default=900, gt=0)
    local_stt_model: str = "oddadmix/Whisperv3-tunisian-codeswitch"
    # What recordings are assumed to contain when the upload does not say.
    stt_default_language: SpokenLanguage = SpokenLanguage.DERJA
    # Names and terms to spell right in every recording, separated by commas.
    stt_keywords: str = ""
    # Largest recording accepted. 25 MB is the hosted API's own limit.
    max_audio_upload_bytes: int = Field(default=25_000_000, gt=0)

    # --- Notion ---
    notion_token: SecretStr | None = None
    notion_database_id: str | None = None
    # Only needed when the database holds more than one data source.
    notion_data_source_id: str | None = None
    notion_owner_property: str = "Owner"
    notion_due_property: str = "Due"
    notion_priority_property: str = "Priority"
    notion_meeting_property: str = "Meeting"  # optional column, used when it exists
    notion_base_url: str = "https://api.notion.com"

    # --- Zoom ---
    zoom_webhook_secret_token: SecretStr | None = None
    # Server-to-Server OAuth credentials: a second way to download transcripts when
    # the token sent with the webhook is refused.
    zoom_account_id: str | None = None
    zoom_client_id: str | None = None
    zoom_client_secret: SecretStr | None = None
    zoom_oauth_url: str = "https://zoom.us/oauth/token"
    # Transcripts are only downloaded from these domains and their subdomains.
    zoom_download_hosts: str = "zoom.us,zoomgov.com"
    # How old a signed webhook may be. Zoom retries a failed delivery after 5, 20
    # and 60 more minutes, so 90 minutes accepts every retry.
    zoom_timestamp_tolerance_seconds: int = Field(default=5_400, gt=0)

    @field_validator("log_level", mode="before")
    @classmethod
    def _uppercase_log_level(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value

    @field_validator("notion_database_id", "notion_data_source_id", mode="before")
    @classmethod
    def _extract_notion_id(cls, value: object) -> object:
        """Accept a bare id or the full URL copied from the browser."""
        if not isinstance(value, str):
            return value
        last_segment = value.strip().split("?")[0].rstrip("/").rsplit("/", 1)[-1]
        match = _NOTION_ID_AT_END.search(last_segment.replace("-", ""))
        return match.group().lower() if match else value.strip()

    @model_validator(mode="after")
    def _require_api_key_in_production(self) -> Self:
        if self.environment == "production" and self.api_key is None:
            raise ValueError(
                "API_KEY must be set when ENVIRONMENT=production. Without it anyone who "
                "finds the URL could spend your LLM credits and write to your Notion."
            )
        return self

    @property
    def llm_configured(self) -> bool:
        key = self.groq_api_key if self.llm_provider == "groq" else self.anthropic_api_key
        return key is not None

    @property
    def stt_configured(self) -> bool:
        return self.stt_provider == "local" or self.openai_api_key is not None

    @property
    def stt_keyword_list(self) -> tuple[str, ...]:
        return tuple(word.strip() for word in self.stt_keywords.split(",") if word.strip())

    @property
    def notion_configured(self) -> bool:
        return self.notion_token is not None and self.notion_database_id is not None

    @property
    def zoom_oauth_configured(self) -> bool:
        return None not in (self.zoom_account_id, self.zoom_client_id, self.zoom_client_secret)

    @property
    def zoom_download_host_list(self) -> list[str]:
        return [host.strip().lower() for host in self.zoom_download_hosts.split(",") if host.strip()]
