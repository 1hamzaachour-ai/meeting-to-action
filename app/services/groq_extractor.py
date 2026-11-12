"""Action-item extraction with an open-weight model on Groq.

Groq serves open models behind an OpenAI-compatible chat API, so this talks plain
HTTP with the shared client: no extra dependency. `base_url` and `model` are enough
to point it at any other server that speaks the same API.

Structured Outputs work the same way as with Claude: the request carries a JSON
schema (built from the same Pydantic model) and, in strict mode, the model can only
answer with JSON that matches it.
"""

import asyncio
import copy
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import httpx2
from pydantic import ValidationError

from app.schemas.extraction import MeetingContext, MeetingExtraction
from app.services.extraction import (
    ActionItemExtractor,
    ExtractionError,
    ExtractionOutcome,
    build_system_prompt,
    build_user_message,
)

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
MAX_RETRY_DELAY_SECONDS = 60
RETRY_STATUSES = frozenset({429, 500, 502, 503})  # nothing is written by this call, so retrying is safe


def strict_schema(model: type[MeetingExtraction] = MeetingExtraction) -> dict[str, Any]:
    """The model's JSON schema, reduced to what strict mode accepts.

    Strict mode wants every property required and `additionalProperties: false` on
    every object. Pydantic already makes every field required here, and writes
    "may be null" as an anyOf. Titles, defaults and `format` hints are dropped: the
    response is validated by Pydantic afterwards, which checks dates itself.
    """
    schema = copy.deepcopy(model.model_json_schema())

    def clean(node: Any) -> None:
        if isinstance(node, dict):
            for key in ("title", "default", "format"):
                if key in node and not isinstance(node[key], dict):
                    del node[key]
            if node.get("type") == "object":
                node["additionalProperties"] = False
                node["required"] = list(node.get("properties", {}))
            for value in node.values():
                clean(value)
        elif isinstance(node, list):
            for item in node:
                clean(item)

    clean(schema)
    return schema


class GroqExtractor(ActionItemExtractor):
    def __init__(
        self,
        http: httpx2.AsyncClient,
        *,
        api_key: str,
        model: str = "openai/gpt-oss-120b",
        base_url: str = "https://api.groq.com/openai/v1",
        reasoning_effort: str | None = "medium",
        max_tokens: int = 16_000,
        timeout_seconds: float = 300,
        task_language: str | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._http = http
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._model = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._reasoning_effort = reasoning_effort
        self._max_tokens = max_tokens
        self._timeout = httpx2.Timeout(timeout_seconds, connect=10.0)
        self._task_language = task_language
        self._sleep = sleep
        self._schema = strict_schema()

    async def extract(self, transcript: str, context: MeetingContext) -> ExtractionOutcome:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": build_system_prompt(context, self._task_language)},
                {"role": "user", "content": build_user_message(transcript, context)},
            ],
            "temperature": 0.1,  # extraction should be repeatable, not creative
            "max_completion_tokens": self._max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "meeting_extraction", "strict": True, "schema": self._schema},
            },
        }
        if self._reasoning_effort:
            body["reasoning_effort"] = self._reasoning_effort

        answer = await self._post(body)
        try:
            choice = answer["choices"][0]
            content = choice["message"]["content"] or ""
            usage = answer.get("usage") or {}
        except (KeyError, IndexError, TypeError) as exc:
            raise ExtractionError("The model service sent an answer in an unexpected format.") from exc

        if choice.get("finish_reason") == "length":
            raise ExtractionError(
                "The model's answer was cut off before it finished. Raise LLM_MAX_TOKENS and try again."
            )
        try:
            extraction = MeetingExtraction.model_validate_json(content)
        except ValidationError as exc:
            logger.error("Model answer did not match the schema: %s", exc)
            raise ExtractionError("The model returned an answer in an unexpected format.") from exc

        logger.info("Extracted %d action items with %s", len(extraction.action_items), self._model)
        return ExtractionOutcome(
            extraction=extraction,
            model=str(answer.get("model") or self._model),
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
        )

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            last = attempt == MAX_ATTEMPTS
            try:
                response = await self._http.post(
                    self._url, headers=self._headers, json=body, timeout=self._timeout
                )
            except httpx2.TransportError as exc:
                if last:
                    raise ExtractionError("Could not reach the model service.") from exc
                await self._sleep(float(2**attempt))
                continue

            if response.status_code in RETRY_STATUSES and not last:
                delay = _retry_after(response, attempt)
                if delay is not None:
                    await self._sleep(delay)
                    continue
            if response.is_success:
                try:
                    return response.json()
                except ValueError as exc:
                    raise ExtractionError("The model service sent an answer in an unexpected format.") from exc
            raise _error_from(response, self._model)
        raise AssertionError("unreachable: the last attempt always returns or raises")


def _retry_after(response: httpx2.Response, attempt: int) -> float | None:
    """How long to wait before retrying, or None when waiting would not help."""
    header = response.headers.get("retry-after", "")
    if header.isdigit():
        seconds = int(header)
        # A daily quota can ask for hours: that is a limit to report, not to wait out.
        return float(seconds) if seconds <= MAX_RETRY_DELAY_SECONDS else None
    return float(2**attempt)


def _error_from(response: httpx2.Response, model: str) -> ExtractionError:
    try:
        error = response.json().get("error") or {}
        detail = error.get("message") if isinstance(error, dict) else None
    except (ValueError, AttributeError):
        detail = None
    status = response.status_code

    if status == 401:
        return ExtractionError("The Groq API key was rejected. Check GROQ_API_KEY.")
    if status == 404:
        return ExtractionError(f"The model {model!r} was not found. Check GROQ_MODEL.")
    if status in (413, 429):
        return ExtractionError(
            "Groq refused the request: the transcript is too long for this plan's per-minute "
            "token limit, or a rate limit was reached. Try a shorter meeting, wait a minute, "
            "or use a plan with higher limits."
        )
    if status in (400, 422) and detail:
        return ExtractionError(f"The model service rejected the request: {detail}")
    return ExtractionError(f"The model service returned an error (HTTP {status}).")
