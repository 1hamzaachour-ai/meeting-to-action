"""Stand-ins for the outside world, shared by the tests.

No test touches the network: the LLM is replaced by `FakeExtractor`, and the Notion
and Zoom servers by in-memory handlers plugged into the app's HTTP client.
"""

import hashlib
import hmac
import json
import time
from pathlib import Path

import httpx2

from app.config import Settings
from app.schemas.extraction import ActionItem, MeetingContext, MeetingExtraction, Priority
from app.schemas.transcription import SpeechHints, Transcription
from app.services.extraction import ActionItemExtractor, ExtractionOutcome
from app.services.transcription import Transcriber

TRANSCRIPT = "Amira: I will send the budget by Friday.\nYoussef: I'll review the API draft.\nAmira: Thanks all."

# Every secret is forced to "unset" so a developer's own environment variables
# can never leak into a test run.
ISOLATED_SETTINGS = {
    "environment": "test",
    "max_upload_bytes": 2_000,
    "api_key": None,
    "anthropic_api_key": None,
    "groq_api_key": None,
    "llm_provider": "anthropic",
    "openai_api_key": None,
    "stt_provider": "openai",
    "task_language": None,
    "notion_token": None,
    "notion_database_id": None,
    "notion_data_source_id": None,
    "zoom_webhook_secret_token": None,
    "zoom_account_id": None,
    "zoom_client_id": None,
    "zoom_client_secret": None,
}


def make_settings(**overrides) -> Settings:
    return Settings(_env_file=None, **{**ISOLATED_SETTINGS, **overrides})


def upload(client, content=TRANSCRIPT.encode(), filename="standup.txt", **form_fields):
    return client.post(
        "/api/v1/transcripts",
        files={"file": (filename, content, "text/plain")},
        data=form_fields,
    )


def json_response(status: int, body: dict, headers: dict | None = None) -> httpx2.Response:
    return httpx2.Response(status, json=body, headers=headers)


def route_by_host(handlers: dict) -> httpx2.MockTransport:
    """One transport for several fake services, chosen by the request's host name."""

    def dispatch(request: httpx2.Request) -> httpx2.Response:
        handler = handlers.get(request.url.host)
        if handler is None:
            raise AssertionError(f"Unexpected outbound request to {request.url}")
        return handler(request)

    return httpx2.MockTransport(dispatch)


# --- LLM ---


class FakeExtractor(ActionItemExtractor):
    """Stands in for the LLM: every line containing "I will" or "I'll" becomes an action item."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, MeetingContext]] = []

    async def extract(self, transcript: str, context: MeetingContext) -> ExtractionOutcome:
        self.calls.append((transcript, context))
        items = [
            ActionItem(
                title=f"Follow up on: {line[:30]}",
                description="Created by the fake extractor.",
                owner=line.split(":")[0] if ":" in line else None,
                due_date=None,
                priority=Priority.MEDIUM,
                source_quote=line,
            )
            for line in transcript.splitlines()
            if "I will" in line or "I'll" in line
        ]
        extraction = MeetingExtraction(
            meeting_title=context.title or "Untitled meeting",
            summary="A fake summary.",
            action_items=items,
        )
        return ExtractionOutcome(extraction=extraction, model="fake-model", input_tokens=100, output_tokens=20)


# --- Speech-to-text ---

# The smallest byte strings that start like the real thing: an MP3 with an ID3 tag,
# and an MP4 audio container. The fake engine below never decodes them.
MP3_BYTES = b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\x00" * 200
M4A_BYTES = b"\x00\x00\x00\x20ftypM4A \x00\x00\x00\x00" + b"\x00" * 200

SPOKEN_TRANSCRIPT = "Amira: I will send the rapport ghodwa.\nيوسف: باش نبعثلك الرابور غدوة إن شاء الله."


class FakeTranscriber(Transcriber):
    """Stands in for the speech engine: returns a fixed Derja/French transcript."""

    def __init__(self, text: str = SPOKEN_TRANSCRIPT, error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.calls: list[tuple[Path, str, SpeechHints]] = []
        self.audio_seen: list[bytes] = []  # proves the file was readable when the engine got it

    async def transcribe(self, audio: Path, filename: str, hints: SpeechHints) -> Transcription:
        self.calls.append((audio, filename, hints))
        self.audio_seen.append(audio.read_bytes())
        if self.error is not None:
            raise self.error
        return Transcription(engine="fake-stt", languages=["ar", "fr"], text=self.text)


def upload_audio(client, content=MP3_BYTES, filename="meeting.mp3", **form_fields):
    return client.post(
        "/api/v1/transcripts",
        files={"file": (filename, content, "application/octet-stream")},
        data=form_fields,
    )


# --- Notion ---

NOTION_COLUMNS = {
    "Name": {"id": "title", "name": "Name", "type": "title", "title": {}},
    "Owner": {"id": "a", "name": "Owner", "type": "rich_text", "rich_text": {}},
    "Due": {"id": "b", "name": "Due", "type": "date", "date": {}},
    "Priority": {"id": "c", "name": "Priority", "type": "select", "select": {"options": []}},
    "Meeting": {"id": "d", "name": "Meeting", "type": "rich_text", "rich_text": {}},
}


class FakeNotion:
    """The three Notion endpoints the app calls, backed by a list."""

    def __init__(self, columns: dict | None = None, data_sources: list[dict] | None = None) -> None:
        self.columns = NOTION_COLUMNS if columns is None else columns
        self.data_sources = [{"id": "ds-1", "name": "Tasks"}] if data_sources is None else data_sources
        self.pages: list[dict] = []  # bodies of the pages created
        self.requests: list[httpx2.Request] = []
        self.next_page_responses: list[httpx2.Response] = []  # served first, to simulate failures

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        path = request.url.path
        if request.method == "GET" and path == "/v1/databases/db-1":
            return json_response(200, {"object": "database", "id": "db-1", "data_sources": self.data_sources})
        if request.method == "GET" and path.startswith("/v1/data_sources/"):
            table = {"object": "data_source", "id": path.rsplit("/", 1)[-1], "properties": self.columns}
            return json_response(200, table)
        if request.method == "POST" and path == "/v1/pages":
            if self.next_page_responses:
                return self.next_page_responses.pop(0)
            self.pages.append(json.loads(request.content))
            number = len(self.pages)
            page = {"object": "page", "id": f"page-{number}", "url": f"https://www.notion.so/page-{number}"}
            return json_response(200, page)
        return json_response(
            404, {"object": "error", "status": 404, "code": "object_not_found", "message": "Could not find it."}
        )


# --- Zoom ---

ZOOM_SECRET = "zoom-webhook-secret"
ZOOM_VTT = (
    "WEBVTT\n\n"
    "1\n00:00:01.000 --> 00:00:05.000\nAmira: I will send the budget by Friday.\n\n"
    "2\n00:00:05.500 --> 00:00:08.000\nYoussef: Thanks Amira.\n"
)


class FakeZoom:
    """Zoom's token endpoint and its recording download, with configurable access."""

    def __init__(self, accepted_tokens: tuple[str, ...] = ("webhook-download-token",), vtt: str = ZOOM_VTT) -> None:
        self.accepted_tokens = accepted_tokens
        self.vtt = vtt
        self.token_requests: list[httpx2.Request] = []
        self.download_tokens: list[str] = []  # the bearer token of each download attempt

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        if request.method == "POST" and request.url.path == "/oauth/token":
            self.token_requests.append(request)
            return json_response(200, {"access_token": "oauth-access-token", "token_type": "bearer", "expires_in": 3600})
        if request.method == "GET" and request.url.path.startswith("/rec/download/"):
            token = request.headers.get("authorization", "").removeprefix("Bearer ")
            self.download_tokens.append(token)
            if token not in self.accepted_tokens:
                return httpx2.Response(401)
            return httpx2.Response(200, content=self.vtt.encode())
        return httpx2.Response(404)


def zoom_transport(zoom: FakeZoom, **other_hosts) -> httpx2.MockTransport:
    return route_by_host({"zoom.us": zoom, "us06web.zoom.us": zoom, **other_hosts})


def transcript_event(meeting_uuid: str = "uuid-1", download_url: str = "https://us06web.zoom.us/rec/download/abc") -> dict:
    """A recording.transcript_completed event, reduced to the fields Zoom documents."""
    return {
        "event": "recording.transcript_completed",
        "event_ts": 1_794_000_000_000,
        "download_token": "webhook-download-token",
        "payload": {
            "account_id": "acct-1",
            "object": {
                "id": 81234567890,
                "uuid": meeting_uuid,
                "topic": "Weekly Product Sync",
                "start_time": "2026-11-09T23:30:00Z",
                "timezone": "Africa/Tunis",
                "recording_files": [
                    {"file_type": "MP4", "recording_type": "shared_screen_with_speaker_view", "download_url": "https://us06web.zoom.us/rec/download/video"},
                    {"file_type": "TRANSCRIPT", "file_extension": "VTT", "recording_type": "audio_transcript", "download_url": download_url},
                ],
            },
        },
    }


def post_zoom_event(client, event: dict, *, secret: str = ZOOM_SECRET, sent_at: float | None = None, signature: str | None = None):
    """Send an event the way Zoom does: raw JSON body, timestamp header, HMAC signature header."""
    body = json.dumps(event, separators=(",", ":")).encode()
    timestamp = str(int(time.time() if sent_at is None else sent_at))
    if signature is None:
        digest = hmac.new(secret.encode(), b"v0:" + timestamp.encode() + b":" + body, hashlib.sha256).hexdigest()
        signature = f"v0={digest}"
    return client.post(
        "/webhooks/zoom",
        content=body,
        headers={"content-type": "application/json", "x-zm-request-timestamp": timestamp, "x-zm-signature": signature},
    )
