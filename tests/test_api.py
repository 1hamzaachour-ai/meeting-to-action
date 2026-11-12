"""End-to-end tests of the upload and job endpoints.

TestClient runs background tasks before it returns the response, so a job has
already finished by the time a test polls it.
"""

from datetime import date

import pytest
from helpers import TRANSCRIPT, upload

from app.services.pipeline import TranscriptPipeline


def max_upload_bytes(client) -> int:
    return client.app.state.settings.max_upload_bytes


def test_health_reports_what_is_configured(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["auth_required"] is False
    assert body["integrations"] == {"llm": True, "speech_to_text": False, "notion": False, "zoom_webhook": False}


def test_upload_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_upload_is_accepted_then_processed_in_the_background(client, extractor):
    response = upload(client, meeting_title="  Weekly standup ", meeting_date="2026-11-09")
    assert response.status_code == 202
    accepted = response.json()
    assert accepted["status"] == "queued"
    assert accepted["duplicate"] is False
    assert response.headers["location"] == accepted["status_url"]

    job = client.get(accepted["status_url"]).json()
    assert job["id"] == accepted["job_id"]
    assert job["status"] == "completed"
    assert job["error"] is None
    assert job["filename"] == "standup.txt"
    assert job["meeting_title"] == "Weekly standup"
    assert job["meeting_date"] == "2026-11-09"

    result = job["result"]
    assert result["transcript"] == {"characters": len(TRANSCRIPT), "words": 17, "lines": 3}
    assert result["meeting_title"] == "Weekly standup"
    assert [item["owner"] for item in result["action_items"]] == ["Amira", "Youssef"]
    assert all(item["quote_verified"] for item in result["action_items"])
    assert result["llm"]["model"] == "fake-model"
    assert result["notion"]["status"] == "skipped"

    # The LLM received the transcript and what is known about the meeting.
    transcript, context = extractor.calls[0]
    assert transcript == TRANSCRIPT
    assert (context.title, context.meeting_date) == ("Weekly standup", date(2026, 11, 9))


def test_blank_title_and_date_are_stored_as_null(client):
    accepted = upload(client, meeting_title="   ", meeting_date="").json()
    job = client.get(accepted["status_url"]).json()
    assert job["meeting_title"] is None
    assert job["meeting_date"] is None
    assert job["result"]["meeting_title"] == "Untitled meeting"  # the extractor's own title is used


def test_client_supplied_path_is_reduced_to_a_file_name(client):
    accepted = upload(client, filename="C:\\Users\\someone\\Desktop\\notes.TXT").json()
    assert client.get(accepted["status_url"]).json()["filename"] == "notes.TXT"


def test_zoom_vtt_upload_is_converted_to_plain_lines(client, extractor):
    vtt = (
        "WEBVTT\n\n"
        "1\n00:00:01.000 --> 00:00:04.000\nAmira: I will send the budget.\n\n"
        "2\n00:00:04.500 --> 00:00:07.000\nYoussef: Thanks.\n"
    )
    assert upload(client, content=vtt.encode(), filename="meeting.vtt").status_code == 202
    assert extractor.calls[0][0] == "Amira: I will send the budget.\nYoussef: Thanks."


def test_same_file_twice_returns_the_first_job(client, extractor):
    first = upload(client)
    second = upload(client)
    assert second.status_code == 200
    assert second.json()["duplicate"] is True
    assert second.json()["job_id"] == first.json()["job_id"]
    assert len(extractor.calls) == 1


def test_force_processes_the_same_file_again(client, extractor):
    first = upload(client)
    again = upload(client, force="true")
    assert again.status_code == 202
    assert again.json()["job_id"] != first.json()["job_id"]
    assert len(extractor.calls) == 2


@pytest.mark.parametrize(
    ("filename", "content", "expected_status"),
    [
        ("notes.pdf", b"some text", 415),  # wrong extension
        ("notes", b"some text", 415),  # no extension
        ("notes.txt", b"  \r\n  ", 422),  # nothing left after trimming
        ("notes.txt", b"\x89PNG\r\n\x1a\n\x00\x00", 422),  # binary file renamed to .txt
        ("notes.txt", b"caf\xe9", 422),  # legacy Windows encoding, not UTF-8
        ("notes.vtt", b"WEBVTT\n\nNOTE nothing was said\n", 422),  # captions file without captions
    ],
)
def test_unusable_files_are_rejected(client, filename, content, expected_status):
    response = upload(client, content=content, filename=filename)
    assert response.status_code == expected_status
    assert response.json()["detail"]


def test_missing_file_is_a_validation_error(client):
    response = client.post("/api/v1/transcripts", data={"meeting_title": "No file attached"})
    assert response.status_code == 422


def test_invalid_meeting_date_is_a_validation_error(client):
    assert upload(client, meeting_date="next Friday").status_code == 422


def test_file_one_byte_over_the_limit_is_rejected(client):
    response = upload(client, content=b"x" * (max_upload_bytes(client) + 1))
    assert response.status_code == 413
    assert "byte limit" in response.json()["detail"]


def test_file_exactly_at_the_limit_is_accepted(client):
    assert upload(client, content=b"x" * max_upload_bytes(client)).status_code == 202


@pytest.fixture
def small_client(make_client):
    """An app whose largest accepted upload of any kind is 5,000 bytes."""
    return make_client(max_audio_upload_bytes=5_000)


def test_oversized_request_is_refused_from_its_content_length(small_client):
    response = upload(small_client, content=b"x" * 300_000)
    assert response.status_code == 413
    assert response.json()["detail"] == "Request body too large."


def test_oversized_chunked_request_is_refused_while_reading(small_client):
    client = small_client

    def chunks():  # a generator body is sent chunked, without a Content-Length header
        for _ in range(300):
            yield b"x" * 1024

    response = client.post(
        "/api/v1/transcripts",
        content=chunks(),
        headers={"content-type": "multipart/form-data; boundary=boundary"},
    )
    assert "content-length" not in response.request.headers
    assert response.status_code == 413
    assert response.json()["detail"] == "Request body too large."


def test_unknown_job_returns_404(client):
    assert client.get("/api/v1/jobs/does-not-exist").status_code == 404


def test_upload_is_refused_when_no_llm_is_configured(make_client):
    client = make_client(with_llm=False)
    response = upload(client)
    assert response.status_code == 503
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]
    assert client.get("/health").json()["integrations"]["llm"] is False


def test_unexpected_failure_is_recorded_without_leaking_details(client, monkeypatch):
    async def explode(self, *args):
        raise RuntimeError("boom")

    monkeypatch.setattr(TranscriptPipeline, "_process", explode)

    job = client.get(upload(client).json()["status_url"]).json()
    assert job["status"] == "failed"
    assert "RuntimeError" in job["error"]
    assert "boom" not in job["error"]  # internal messages stay in the logs
    assert job["result"] is None


def test_a_failed_job_does_not_block_a_new_attempt(client, monkeypatch):
    async def explode(self, *args):
        raise RuntimeError("boom")

    with monkeypatch.context() as patch:
        patch.setattr(TranscriptPipeline, "_process", explode)
        failed = upload(client).json()

    retry = upload(client)
    assert retry.status_code == 202
    assert retry.json()["job_id"] != failed["job_id"]
    assert client.get(retry.json()["status_url"]).json()["status"] == "completed"


# --- API key ---


def test_api_key_is_required_when_the_server_sets_one(make_client):
    client = make_client(api_key="s3cret")
    assert client.get("/health").json()["auth_required"] is True  # health stays public
    assert client.get("/").status_code == 200  # so does the page

    assert upload(client).status_code == 401
    assert client.get("/api/v1/jobs/anything").status_code == 401

    client.headers["X-API-Key"] = "wrong"
    assert upload(client).status_code == 401

    client.headers["X-API-Key"] = "s3cret"
    accepted = upload(client)
    assert accepted.status_code == 202
    assert client.get(accepted.json()["status_url"]).status_code == 200
