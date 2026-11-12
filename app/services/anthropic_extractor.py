"""The Anthropic (Claude) implementation of ActionItemExtractor.

Kept in its own module because the `anthropic` SDK is large: importing it takes
seconds, so the app only loads this file when an Anthropic key is configured.
"""

import logging
from typing import Self

import anthropic
from pydantic import ValidationError

from app.config import Settings
from app.schemas.extraction import MeetingContext, MeetingExtraction
from app.services.extraction import (
    ActionItemExtractor,
    ExtractionError,
    ExtractionOutcome,
    build_system_prompt,
    build_user_message,
)

logger = logging.getLogger(__name__)

# Structured Outputs: the API constrains the reply to this JSON schema, so the
# answer always parses into MeetingExtraction.
OUTPUT_FORMAT = {"type": "json_schema", "schema": anthropic.transform_schema(MeetingExtraction)}

# Lets the API re-run a request on another model when a safety classifier declines
# it, instead of failing the job. Models without fallbacks simply ignore it.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicExtractor(ActionItemExtractor):
    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        *,
        model: str,
        effort: str,
        max_tokens: int,
        task_language: str | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._effort = effort
        self._max_tokens = max_tokens
        self._task_language = task_language

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        if settings.anthropic_api_key is None:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        client = anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key.get_secret_value(),
            base_url=settings.anthropic_base_url,
            timeout=settings.llm_timeout_seconds,
        )
        return cls(
            client,
            model=settings.llm_model,
            effort=settings.llm_effort,
            max_tokens=settings.llm_max_tokens,
            task_language=settings.task_language,
        )

    async def aclose(self) -> None:
        await self._client.close()

    async def extract(self, transcript: str, context: MeetingContext) -> ExtractionOutcome:
        message = await self._ask(transcript, context)

        if message.stop_reason == "refusal":
            raise ExtractionError("The model declined to process this transcript.")
        if message.stop_reason == "max_tokens":
            raise ExtractionError(
                "The model's answer was cut off before it finished. Raise LLM_MAX_TOKENS and try again."
            )

        # After a fallback the answer is the last text block, not the first.
        answer = next((block.text for block in reversed(message.content) if block.type == "text"), "")
        try:
            extraction = MeetingExtraction.model_validate_json(answer)
        except ValidationError as exc:
            logger.error("Model answer did not match the schema (message %s): %s", message.id, exc)
            raise ExtractionError("The model returned an answer in an unexpected format.") from exc

        logger.info(
            "Extracted %d action items with %s (%d input tokens, %d output tokens)",
            len(extraction.action_items),
            message.model,
            message.usage.input_tokens,
            message.usage.output_tokens,
        )
        return ExtractionOutcome(
            extraction=extraction,
            model=message.model,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
        )

    async def _ask(self, transcript: str, context: MeetingContext):
        try:
            # Streamed so a long transcript cannot run into an idle-connection
            # timeout; only the finished message is used.
            async with self._client.beta.messages.stream(
                model=self._model,
                max_tokens=self._max_tokens,
                system=build_system_prompt(context, self._task_language),
                messages=[{"role": "user", "content": build_user_message(transcript, context)}],
                output_config={"effort": self._effort, "format": OUTPUT_FORMAT},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            ) as stream:
                return await stream.get_final_message()
        except anthropic.AuthenticationError as exc:
            raise ExtractionError("The Anthropic API key was rejected. Check ANTHROPIC_API_KEY.") from exc
        except anthropic.PermissionDeniedError as exc:
            raise ExtractionError(
                f"This Anthropic API key is not allowed to use the model {self._model!r}."
            ) from exc
        except anthropic.NotFoundError as exc:
            raise ExtractionError(f"The model {self._model!r} does not exist. Check LLM_MODEL.") from exc
        except anthropic.RateLimitError as exc:
            raise ExtractionError("The Anthropic API rate limit was reached. Try again in a minute.") from exc
        except anthropic.BadRequestError as exc:
            raise ExtractionError(f"The Anthropic API rejected the request: {_api_message(exc)}") from exc
        except anthropic.APIStatusError as exc:
            raise ExtractionError(
                f"The Anthropic API returned an error (HTTP {exc.status_code}). Try again later."
            ) from exc
        except anthropic.APIConnectionError as exc:  # includes timeouts
            raise ExtractionError("Could not reach the Anthropic API.") from exc


def _api_message(error: anthropic.APIStatusError) -> str:
    """The API's own explanation, e.g. "prompt is too long", without the surrounding JSON."""
    body = error.body
    if isinstance(body, dict) and isinstance(body.get("error"), dict) and body["error"].get("message"):
        return str(body["error"]["message"])
    return error.message
