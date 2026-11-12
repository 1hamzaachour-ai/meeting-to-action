"""Phase 4: Zoom webhooks - checking that a request is from Zoom, and downloading transcripts."""

import asyncio
import hashlib
import hmac
import time
from datetime import UTC, date
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx2

from app.errors import UserFacingError
from app.schemas.zoom import ZoomRecording, ZoomRecordingFile

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class ZoomError(UserFacingError):
    """Downloading a transcript from Zoom failed."""


# --- Webhook authentication -------------------------------------------------


def _hmac_sha256(secret: str, message: bytes) -> str:
    return hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()


def expected_signature(secret: str, timestamp: str, body: bytes) -> str:
    """The value Zoom puts in x-zm-signature: HMAC-SHA256 of "v0:{timestamp}:{raw body}"."""
    return "v0=" + _hmac_sha256(secret, b"v0:" + timestamp.encode() + b":" + body)


def is_from_zoom(
    *,
    secret: str,
    timestamp: str | None,
    signature: str | None,
    body: bytes,
    tolerance_seconds: int,
    now: float | None = None,
) -> bool:
    """True when the request was signed with our webhook secret, recently enough."""
    if not timestamp or not signature:
        return False
    try:
        sent_at = float(timestamp)
    except ValueError:
        return False
    if sent_at > 1e11:  # a timestamp in milliseconds rather than seconds
        sent_at /= 1000
    current_time = time.time() if now is None else now
    if abs(current_time - sent_at) > tolerance_seconds:
        return False  # too old: possibly a captured request being replayed
    # compare_digest takes the same time whatever the input, so the comparison
    # leaks nothing about the expected value.
    return hmac.compare_digest(expected_signature(secret, timestamp, body), signature)


def url_validation_response(secret: str, plain_token: str) -> dict[str, str]:
    """Answer to Zoom's endpoint.url_validation challenge, which proves we hold the secret."""
    return {"plainToken": plain_token, "encryptedToken": _hmac_sha256(secret, plain_token.encode())}


# --- Reading a recording event ----------------------------------------------


def find_transcript_file(recording: ZoomRecording) -> ZoomRecordingFile | None:
    """The audio transcript among the recording's files, if there is one to download."""
    transcripts = [
        file
        for file in recording.recording_files
        if (file.file_type or "").upper() == "TRANSCRIPT" and file.download_url
    ]
    preferred = [file for file in transcripts if file.recording_type == "audio_transcript"]
    return (preferred or transcripts or [None])[0]


def meeting_date(recording: ZoomRecording) -> date | None:
    """The calendar day of the meeting where the host was, used to resolve "by Friday"."""
    if recording.start_time is None:
        return None
    started = recording.start_time
    if started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    try:
        zone = ZoneInfo(recording.timezone) if recording.timezone else UTC
    except (ZoneInfoNotFoundError, ValueError):
        zone = UTC
    return started.astimezone(zone).date()


# --- Downloading the transcript ---------------------------------------------


class ZoomClient:
    def __init__(
        self,
        http: httpx2.AsyncClient,
        *,
        allowed_hosts: list[str],
        max_bytes: int,
        account_id: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        oauth_url: str = "https://zoom.us/oauth/token",
    ) -> None:
        self._http = http
        self._allowed_hosts = allowed_hosts
        self._max_bytes = max_bytes
        self._account_id = account_id
        self._client_id = client_id
        self._client_secret = client_secret
        self._oauth_url = oauth_url
        self._token: str | None = None
        self._token_expires_at = 0.0
        self._token_lock = asyncio.Lock()

    @property
    def has_oauth_credentials(self) -> bool:
        return None not in (self._account_id, self._client_id, self._client_secret)

    async def download_transcript(self, url: str, download_token: str | None) -> bytes:
        """Fetch a transcript file.

        The token Zoom sends with the webhook is tried first. If Zoom refuses it and
        Server-to-Server OAuth credentials are configured, an access token is tried.
        """
        self._check_url(url)

        async def webhook_token() -> str:
            return download_token or ""

        token_sources = []
        if download_token:
            token_sources.append(webhook_token)
        if self.has_oauth_credentials:
            token_sources.append(self._oauth_token)
        if not token_sources:
            raise ZoomError(
                "Cannot download the transcript: the event carried no download token and "
                "the Zoom OAuth credentials are not set."
            )

        refused_status = 0
        for get_token in token_sources:
            status, content = await self._get(url, await get_token())
            if status in (401, 403):
                refused_status = status
                continue
            if 200 <= status < 300:
                return content
            raise ZoomError(f"Zoom answered HTTP {status} to the transcript download.")
        raise ZoomError(
            f"Zoom refused the transcript download (HTTP {refused_status}). "
            "Check that the Zoom app has the cloud recording scopes."
        )

    def _check_url(self, url: str) -> None:
        """Only fetch from Zoom, and only over HTTPS.

        The URL comes from the request body. Even though the request is signed, the
        server should never be steerable into calling an arbitrary address.
        """
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        is_allowed = any(host == allowed or host.endswith("." + allowed) for allowed in self._allowed_hosts)
        # Plain HTTP is accepted for this machine only, to test against a local file server.
        is_secure = parts.scheme == "https" or (parts.scheme == "http" and host in LOOPBACK_HOSTS)
        if not (is_allowed and is_secure):
            raise ZoomError(
                f"Refused to download from {host or 'an invalid address'}: it is not an allowed "
                "Zoom host (ZOOM_DOWNLOAD_HOSTS)."
            )

    async def _get(self, url: str, token: str) -> tuple[int, bytes]:
        """Download with a size cap, so a huge file cannot exhaust memory."""
        try:
            async with self._http.stream(
                "GET", url, headers={"Authorization": f"Bearer {token}"}, follow_redirects=True
            ) as response:
                if not response.is_success:
                    return response.status_code, b""
                chunks: list[bytes] = []
                size = 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > self._max_bytes:
                        raise ZoomError(
                            f"The Zoom transcript is larger than the {self._max_bytes:,} byte limit."
                        )
                    chunks.append(chunk)
                return response.status_code, b"".join(chunks)
        except httpx2.HTTPError as exc:
            raise ZoomError("Could not download the transcript from Zoom (network error).") from exc

    async def _oauth_token(self) -> str:
        """A Server-to-Server OAuth access token, reused until shortly before it expires."""
        async with self._token_lock:  # concurrent jobs share one token request
            if self._token is None or time.monotonic() >= self._token_expires_at:
                try:
                    response = await self._http.post(
                        self._oauth_url,
                        data={"grant_type": "account_credentials", "account_id": self._account_id},
                        auth=(self._client_id or "", self._client_secret or ""),
                    )
                except httpx2.HTTPError as exc:
                    raise ZoomError("Could not reach Zoom to get an access token.") from exc
                if not response.is_success:
                    raise ZoomError(
                        f"Zoom rejected the Server-to-Server OAuth credentials (HTTP {response.status_code}). "
                        "Check ZOOM_ACCOUNT_ID, ZOOM_CLIENT_ID and ZOOM_CLIENT_SECRET."
                    )
                body = response.json()
                self._token = body["access_token"]
                lifetime = float(body.get("expires_in", 3600))
                self._token_expires_at = time.monotonic() + max(lifetime - 60, 0)  # renew a minute early
            return self._token
