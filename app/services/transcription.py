"""Phase 5: speech-to-text.

`Transcriber` is the interface the pipeline depends on. This file holds the hosted
implementation, which speaks OpenAI's transcription API. The implementation that
runs a Hugging Face model on this machine is in local_whisper.py.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx2

from app.errors import UserFacingError
from app.schemas.transcription import LANGUAGE_CODES, SpeechHints, SpokenLanguage, Transcription

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
MAX_RETRY_DELAY_SECONDS = 60
MAX_KEYWORDS = 50
# Transcribing has no side effects, so every transient failure can simply be retried.
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

AUDIO_CONTENT_TYPES = {".mp3": "audio/mpeg", ".m4a": "audio/mp4"}

# Context for models that take a description of the recording (gpt-transcribe).
DERJA_CONTEXT = (
    "A business meeting in Tunisia. The speakers mix Tunisian Arabic (Derja) with French and "
    "English, often within one sentence. Tunisian Arabic is written in Arabic script; French "
    "and English words stay in Latin script, exactly as they were said, without translation."
)

# Whisper-style models do not follow instructions: their prompt is read as the text
# that came just before, so it shows the wanted style instead of describing it.
DERJA_STYLE_SAMPLE = (
    "اجتماع عمل بالدارجة التونسية. باش نحضرو الـ rapport ونبعثوه للـ client قبل الـ deadline، "
    "وبعد نعملو réunion أخرى على الـ budget والـ planning."
)


class TranscriptionError(UserFacingError):
    """The speech-to-text step failed."""


class Transcriber(ABC):
    """The one thing the pipeline needs from a speech-to-text engine."""

    @abstractmethod
    async def transcribe(self, audio: Path, filename: str, hints: SpeechHints) -> Transcription:
        """Return the text spoken in the audio file, or raise TranscriptionError."""

    async def aclose(self) -> None:
        """Release resources at shutdown. Nothing to do by default."""


class OpenAITranscriber(Transcriber):
    """Calls POST {base_url}/audio/transcriptions.

    OpenAI defines this API, and several other services and self-hosted servers
    implement it too, so `base_url` and `model` are enough to switch between them.
    """

    def __init__(
        self,
        http: httpx2.AsyncClient,
        *,
        api_key: str | None,
        model: str = "gpt-transcribe",
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 900,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._http = http
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._model = model
        self._url = base_url.rstrip("/") + "/audio/transcriptions"
        # A long recording takes minutes to transcribe; only connecting should be quick.
        self._timeout = httpx2.Timeout(timeout_seconds, connect=10.0)
        self._sleep = sleep

    async def transcribe(self, audio: Path, filename: str, hints: SpeechHints) -> Transcription:
        fields = self._form_fields(hints)
        content_type = AUDIO_CONTENT_TYPES.get(Path(filename).suffix.lower(), "application/octet-stream")

        for attempt in range(1, MAX_ATTEMPTS + 1):
            is_last_attempt = attempt == MAX_ATTEMPTS
            try:
                with audio.open("rb") as stream:  # reopened for each attempt, read from the start
                    response = await self._http.post(
                        self._url,
                        headers=self._headers,
                        data=fields,
                        files={"file": (filename, stream, content_type)},
                        timeout=self._timeout,
                    )
            except httpx2.TransportError as exc:
                if is_last_attempt:
                    raise TranscriptionError("Could not reach the speech-to-text service.") from exc
                await self._sleep(_backoff(attempt))
                continue

            if response.status_code in RETRY_STATUSES and not is_last_attempt:
                await self._sleep(_retry_delay(response, attempt))
                continue
            if not response.is_success:
                raise _error_from(response, self._model)
            return self._read(response)

        raise AssertionError("unreachable: the last attempt always returns or raises")

    def _form_fields(self, hints: SpeechHints) -> dict[str, Any]:
        """The multipart fields that go with the file. A list becomes a repeated field."""
        codes = LANGUAGE_CODES[hints.language]
        keywords = _clean_keywords(hints.keywords)
        is_derja = hints.language is SpokenLanguage.DERJA
        fields: dict[str, Any] = {"model": self._model, "response_format": "json"}

        if self._model.startswith("gpt-transcribe"):
            # This model takes every language to expect, which suits speech that
            # switches between them, and a list of terms to spell right.
            if codes:
                fields["languages[]"] = list(codes)
            if keywords:
                fields["keywords[]"] = keywords
            if is_derja:
                fields["prompt"] = DERJA_CONTEXT
        else:
            # Whisper-style API: a single language, and a prompt that primes the
            # style and vocabulary.
            if codes:
                fields["language"] = codes[0]
            prompt = " ".join(part for part in (DERJA_STYLE_SAMPLE if is_derja else "", ", ".join(keywords)) if part)
            if prompt:
                fields["prompt"] = prompt
        return fields

    def _read(self, response: httpx2.Response) -> Transcription:
        try:
            body = response.json()
            text = body["text"].strip()
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise TranscriptionError("The speech-to-text service sent an answer in an unexpected format.") from exc
        if not text:
            raise TranscriptionError("No speech was recognised in the recording.")
        detected = [entry["code"] for entry in body.get("languages") or [] if isinstance(entry, dict) and "code" in entry]
        logger.info("Transcribed %d characters with %s (languages: %s)", len(text), self._model, detected or "n/a")
        return Transcription(engine=self._model, languages=detected, text=text)


def _clean_keywords(keywords: tuple[str, ...]) -> list[str]:
    """The API rejects a request whose keywords contain <, > or a line break."""
    cleaned = []
    for keyword in keywords:
        keyword = " ".join(keyword.replace("<", " ").replace(">", " ").split())[:80]
        if keyword and keyword not in cleaned:
            cleaned.append(keyword)
    return cleaned[:MAX_KEYWORDS]


def _backoff(attempt: int) -> float:
    return float(min(2**attempt, 8))  # 2, 4 seconds


def _retry_delay(response: httpx2.Response, attempt: int) -> float:
    retry_after = response.headers.get("Retry-After", "")
    if retry_after.isdigit():
        return float(min(int(retry_after), MAX_RETRY_DELAY_SECONDS))
    return _backoff(attempt)


def _error_from(response: httpx2.Response, model: str) -> TranscriptionError:
    try:
        error = response.json().get("error") or {}
    except (ValueError, AttributeError):
        error = {}
    detail = error.get("message") if isinstance(error, dict) else None
    status = response.status_code

    if status == 401:
        return TranscriptionError("The speech-to-text API key was rejected. Check OPENAI_API_KEY.")
    if status == 404:
        return TranscriptionError(f"The speech-to-text model {model!r} was not found. Check STT_MODEL and STT_BASE_URL.")
    if status == 413:
        return TranscriptionError("The recording is too large for the speech-to-text service (25 MB at most).")
    if status == 429:
        return TranscriptionError("The speech-to-text service is rate limited or out of credit. Try again later.")
    if status == 400 and detail:
        return TranscriptionError(f"The speech-to-text service rejected the recording: {detail}")
    return TranscriptionError(f"The speech-to-text service returned an error (HTTP {status}).")
