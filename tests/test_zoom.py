"""Phase 4 tests: Zoom's webhook signature, the events, and the transcript download.

`FakeZoom` (see helpers.py) plays Zoom's token endpoint and file server.
"""

import asyncio
import time
from datetime import date

import httpx2
import pytest
from helpers import (
    ZOOM_SECRET,
    FakeNotion,
    FakeZoom,
    post_zoom_event,
    route_by_host,
    transcript_event,
    upload,
    zoom_transport,
)

from app.schemas.zoom import ZoomRecording
from app.services.zoom import (
    ZoomClient,
    ZoomError,
    expected_signature,
    find_transcript_file,
    is_from_zoom,
    meeting_date,
    url_validation_response,
)

ZOOM_SETTINGS = {"zoom_webhook_secret_token": ZOOM_SECRET}
OAUTH_SETTINGS = {"zoom_account_id": "acct-1", "zoom_client_id": "client-id", "zoom_client_secret": "client-secret"}

# Reference values computed with Node's crypto module, the way Zoom's own sample
# code does it, so these tests do not just compare the implementation with itself.
VALIDATION_BODY = (
    b'{"event":"endpoint.url_validation","payload":{"plainToken":"qgg8vlvZRS6UYooatFL8Aw"},"event_ts":1794000000000}'
)
VALIDATION_TIMESTAMP = "1794000000"
VALIDATION_SIGNATURE = "v0=d91f5e05872fe87739a0140fe8ba65f3d64a49a54140eca64e52f7dd8c1494e1"
ENCRYPTED_TOKEN = "652a0188f06bf15dec80150c3da2729148fdcb8439d1f19ff7eba53901875baf"


# --- Signature ---


def test_signature_matches_the_reference_value():
    assert expected_signature(ZOOM_SECRET, VALIDATION_TIMESTAMP, VALIDATION_BODY) == VALIDATION_SIGNATURE


def test_url_validation_answer_matches_the_reference_value():
    assert url_validation_response(ZOOM_SECRET, "qgg8vlvZRS6UYooatFL8Aw") == {
        "plainToken": "qgg8vlvZRS6UYooatFL8Aw",
        "encryptedToken": ENCRYPTED_TOKEN,
    }


def check(**changes) -> bool:
    arguments = {
        "secret": ZOOM_SECRET,
        "timestamp": VALIDATION_TIMESTAMP,
        "signature": VALIDATION_SIGNATURE,
        "body": VALIDATION_BODY,
        "tolerance_seconds": 300,
        "now": 1_794_000_100.0,
    }
    return is_from_zoom(**{**arguments, **changes})


def test_genuine_request_is_accepted():
    assert check() is True


@pytest.mark.parametrize(
    "changes",
    [
        {"secret": "another-secret"},
        {"body": VALIDATION_BODY + b" "},  # one byte added
        {"signature": VALIDATION_SIGNATURE[:-1] + "0"},
        {"signature": None},
        {"timestamp": None},
        {"timestamp": "not-a-number"},
        {"timestamp": "1794000001"},  # the timestamp is part of what is signed
        {"now": 1_794_000_000.0 + 301},  # older than the tolerance
        {"now": 1_794_000_000.0 - 301},  # dated in the future
    ],
)
def test_forged_altered_or_stale_requests_are_rejected(changes):
    assert check(**changes) is False


# --- Reading the event ---


def test_the_audio_transcript_is_picked_among_the_recording_files():
    recording = ZoomRecording.model_validate(transcript_event()["payload"]["object"])
    chosen = find_transcript_file(recording)
    assert chosen is not None
    assert chosen.download_url == "https://us06web.zoom.us/rec/download/abc"


def test_recording_without_a_transcript_has_nothing_to_download():
    recording = ZoomRecording(recording_files=[{"file_type": "MP4", "download_url": "https://zoom.us/rec/download/v"}])
    assert find_transcript_file(recording) is None


@pytest.mark.parametrize(
    ("start_time", "timezone", "expected"),
    [
        ("2026-11-09T23:30:00Z", "Africa/Tunis", date(2026, 11, 10)),  # already the next day for the host
        ("2026-11-09T23:30:00Z", "America/New_York", date(2026, 11, 9)),
        ("2026-11-09T23:30:00Z", "Not/AZone", date(2026, 11, 9)),  # unknown zone: UTC
        ("2026-11-09T23:30:00Z", None, date(2026, 11, 9)),
        (None, "Africa/Tunis", None),
    ],
)
def test_meeting_date_is_the_hosts_calendar_day(start_time, timezone, expected):
    assert meeting_date(ZoomRecording(start_time=start_time, timezone=timezone)) == expected


# --- Download rules ---


@pytest.mark.parametrize(
    ("url", "allowed_hosts", "accepted"),
    [
        ("https://zoom.us/rec/download/abc", ["zoom.us"], True),
        ("https://us06web.zoom.us/rec/download/abc", ["zoom.us"], True),
        ("https://example.zoomgov.com/rec/download/abc", ["zoom.us", "zoomgov.com"], True),
        ("http://zoom.us/rec/download/abc", ["zoom.us"], False),  # not HTTPS
        ("https://evilzoom.us/rec/download/abc", ["zoom.us"], False),  # only looks like zoom.us
        ("https://zoom.us.evil.example/rec/download/abc", ["zoom.us"], False),
        ("https://169.254.169.254/latest/meta-data", ["zoom.us"], False),  # an internal address
        ("http://127.0.0.1:9000/sample.vtt", ["zoom.us"], False),  # this machine, but not allowed
        ("http://127.0.0.1:9000/sample.vtt", ["zoom.us", "127.0.0.1"], True),  # explicitly allowed for local tests
        ("not a url", ["zoom.us"], False),
    ],
)
def test_transcripts_are_only_fetched_from_allowed_zoom_hosts(url, allowed_hosts, accepted):
    async def attempt() -> bool:
        transport = httpx2.MockTransport(lambda request: httpx2.Response(200, content=b"WEBVTT"))
        async with httpx2.AsyncClient(transport=transport) as http:
            client = ZoomClient(http, allowed_hosts=allowed_hosts, max_bytes=1_000)
            try:
                await client.download_transcript(url, "token")
            except ZoomError:
                return False
            return True

    assert asyncio.run(attempt()) is accepted


# --- The webhook endpoint ---


@pytest.fixture
def zoom() -> FakeZoom:
    return FakeZoom()


@pytest.fixture
def zoom_client(make_client, zoom):
    return make_client(transport=zoom_transport(zoom), **ZOOM_SETTINGS)


def test_webhook_is_unavailable_until_the_secret_is_configured(client):
    assert post_zoom_event(client, transcript_event()).status_code == 503
    assert client.get("/health").json()["integrations"]["zoom_webhook"] is False


def test_request_with_a_bad_signature_is_refused(zoom_client, extractor):
    assert post_zoom_event(zoom_client, transcript_event(), secret="wrong-secret").status_code == 401
    assert post_zoom_event(zoom_client, transcript_event(), signature="v0=abc").status_code == 401
    assert extractor.calls == []


def test_request_older_than_the_tolerance_is_refused(zoom_client):
    two_hours_ago = time.time() - 7_200
    assert post_zoom_event(zoom_client, transcript_event(), sent_at=two_hours_ago).status_code == 401


def test_zoom_retry_an_hour_later_is_still_accepted(zoom_client):
    an_hour_ago = time.time() - 3_600  # Zoom's last retry comes 85 minutes after the first delivery
    assert post_zoom_event(zoom_client, transcript_event(), sent_at=an_hour_ago).status_code == 200


def test_signed_request_that_is_not_an_event_is_a_bad_request(zoom_client):
    assert post_zoom_event(zoom_client, {"hello": "world"}).status_code == 400


def test_url_validation_challenge_is_answered(zoom_client):
    event = {"event": "endpoint.url_validation", "payload": {"plainToken": "qgg8vlvZRS6UYooatFL8Aw"}, "event_ts": 1}
    response = post_zoom_event(zoom_client, event)
    assert response.status_code == 200
    assert response.json() == {"plainToken": "qgg8vlvZRS6UYooatFL8Aw", "encryptedToken": ENCRYPTED_TOKEN}


def test_other_events_are_acknowledged_and_ignored(zoom_client, extractor):
    response = post_zoom_event(zoom_client, {"event": "meeting.started", "payload": {"object": {"id": 1}}})
    assert response.status_code == 200
    assert response.json()["status"] == "ignored"
    assert extractor.calls == []


def test_event_without_a_transcript_file_is_ignored(zoom_client):
    event = transcript_event()
    event["payload"]["object"]["recording_files"] = event["payload"]["object"]["recording_files"][:1]  # video only
    assert post_zoom_event(zoom_client, event).json()["status"] == "ignored"


def test_transcript_event_downloads_and_processes_the_transcript(zoom_client, zoom, extractor):
    response = post_zoom_event(zoom_client, transcript_event())
    assert response.status_code == 200
    assert response.json()["status"] == "accepted"

    job = zoom_client.get(f"/api/v1/jobs/{response.json()['job_id']}").json()
    assert job["status"] == "completed"
    assert job["source"] == "zoom"
    assert job["meeting_title"] == "Weekly Product Sync"
    assert job["meeting_date"] == "2026-11-10"  # 23:30 UTC is the next day in Tunis
    assert job["sha256"] is not None
    assert len(job["result"]["action_items"]) == 1

    # The file was fetched with the token from the event, and the LLM got plain lines.
    assert zoom.download_tokens == ["webhook-download-token"]
    assert zoom.token_requests == []
    transcript, context = extractor.calls[0]
    assert transcript == "Amira: I will send the budget by Friday.\nYoussef: Thanks Amira."
    assert context.meeting_date == date(2026, 11, 10)


def test_repeated_delivery_of_the_same_meeting_is_not_processed_twice(zoom_client, zoom, extractor):
    first = post_zoom_event(zoom_client, transcript_event()).json()
    second = post_zoom_event(zoom_client, transcript_event()).json()
    assert second == {"status": "duplicate", "job_id": first["job_id"]}
    assert len(zoom.download_tokens) == 1
    assert len(extractor.calls) == 1

    other_meeting = post_zoom_event(zoom_client, transcript_event(meeting_uuid="uuid-2")).json()
    assert other_meeting["status"] == "accepted"


def test_oauth_token_is_used_when_the_event_token_is_refused(make_client, extractor):
    zoom = FakeZoom(accepted_tokens=("oauth-access-token",))
    client = make_client(transport=zoom_transport(zoom), **ZOOM_SETTINGS, **OAUTH_SETTINGS)

    first = post_zoom_event(client, transcript_event()).json()
    assert client.get(f"/api/v1/jobs/{first['job_id']}").json()["status"] == "completed"
    assert zoom.download_tokens == ["webhook-download-token", "oauth-access-token"]

    # The token request follows Zoom's Server-to-Server OAuth rules.
    [token_request] = zoom.token_requests
    assert str(token_request.url) == "https://zoom.us/oauth/token"
    assert token_request.headers["authorization"] == "Basic Y2xpZW50LWlkOmNsaWVudC1zZWNyZXQ="  # client-id:client-secret
    assert token_request.content == b"grant_type=account_credentials&account_id=acct-1"

    # A second meeting reuses the cached access token instead of asking again.
    post_zoom_event(client, transcript_event(meeting_uuid="uuid-2"))
    assert len(zoom.token_requests) == 1
    assert len(extractor.calls) == 2


def test_refused_download_fails_the_job_with_a_hint(make_client):
    zoom = FakeZoom(accepted_tokens=())
    client = make_client(transport=zoom_transport(zoom), **ZOOM_SETTINGS)
    job_id = post_zoom_event(client, transcript_event()).json()["job_id"]
    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "failed"
    assert "refused the transcript download (HTTP 401)" in job["error"]


def test_download_url_outside_zoom_is_never_fetched(make_client, extractor):
    # route_by_host raises on any request to a host it does not know, so reaching
    # the network here would turn into an unexpected error instead of this message.
    client = make_client(transport=route_by_host({}), **ZOOM_SETTINGS)
    event = transcript_event(download_url="https://attacker.example/steal")
    job_id = post_zoom_event(client, event).json()["job_id"]
    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "failed"
    assert "not an allowed Zoom host" in job["error"]
    assert extractor.calls == []


def test_oversized_zoom_transcript_is_refused(make_client):
    zoom = FakeZoom(vtt="WEBVTT\n\n" + "x" * 5_000)
    client = make_client(transport=zoom_transport(zoom), **ZOOM_SETTINGS)  # limit: 2,000 bytes
    job_id = post_zoom_event(client, transcript_event()).json()["job_id"]
    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "failed"
    assert "byte limit" in job["error"]


def test_event_gets_a_retryable_answer_when_no_llm_is_configured(make_client):
    client = make_client(transport=zoom_transport(FakeZoom()), with_llm=False, **ZOOM_SETTINGS)
    assert post_zoom_event(client, transcript_event()).status_code == 503  # Zoom retries on 5xx


def test_webhook_does_not_need_the_api_key_but_job_status_does(make_client):
    zoom = FakeZoom()
    client = make_client(transport=zoom_transport(zoom), api_key="s3cret", **ZOOM_SETTINGS)
    response = post_zoom_event(client, transcript_event())  # authenticated by its signature
    assert response.status_code == 200
    assert client.get(f"/api/v1/jobs/{response.json()['job_id']}").status_code == 401


def test_zoom_meeting_ends_with_tasks_in_notion(make_client):
    zoom, notion = FakeZoom(), FakeNotion()
    client = make_client(
        transport=zoom_transport(zoom, **{"api.notion.com": notion}),
        notion_token="secret-token",
        notion_database_id="db-1",
        **ZOOM_SETTINGS,
    )
    job_id = post_zoom_event(client, transcript_event()).json()["job_id"]
    job = client.get(f"/api/v1/jobs/{job_id}").json()

    assert job["status"] == "completed"
    assert job["result"]["notion"]["status"] == "created"
    [page] = notion.pages
    assert page["properties"]["Meeting"]["rich_text"][0]["text"]["content"] == "Weekly Product Sync"


def test_upload_and_zoom_paths_share_the_same_pipeline(zoom_client, extractor):
    upload(zoom_client)
    post_zoom_event(zoom_client, transcript_event())
    assert len(extractor.calls) == 2
