"""Phase 5 tests: audio uploads, the hosted speech-to-text client, and the local one.

No audio is decoded and no model is loaded here: the speech engine is replaced by
`FakeTranscriber`, or its HTTP layer by a handler that answers like the API.
"""

import asyncio
import tempfile
from pathlib import Path

import httpx2
import pytest
from helpers import M4A_BYTES, MP3_BYTES, SPOKEN_TRANSCRIPT, FakeTranscriber, json_response, upload, upload_audio

from app.schemas.transcription import SpeechHints, SpokenLanguage, Transcription
from app.services import local_whisper
from app.services.audio import looks_like_audio
from app.services.local_whisper import LocalWhisperTranscriber
from app.services.transcription import DERJA_CONTEXT, DERJA_STYLE_SAMPLE, OpenAITranscriber, TranscriptionError


@pytest.fixture
def audio_client(make_client, tmp_path, monkeypatch):
    """An app with a (fake) speech engine, whose temporary files land in tmp_path."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return make_client(with_stt=True)


def leftover_files(tmp_path: Path) -> list[str]:
    return [path.name for path in tmp_path.iterdir() if path.name.startswith("meeting-audio-")]


# --- Uploading a recording ---


@pytest.mark.parametrize(("content", "filename"), [(MP3_BYTES, "weekly.mp3"), (M4A_BYTES, "Voice Memo.M4A")])
def test_recording_is_transcribed_then_processed_like_a_transcript(
    audio_client, extractor, transcriber, tmp_path, content, filename
):
    response = upload_audio(audio_client, content=content, filename=filename, meeting_date="2026-11-09")
    assert response.status_code == 202

    job = audio_client.get(response.json()["status_url"]).json()
    assert job["status"] == "completed"
    assert job["stage"] is None
    assert job["media"] == "audio"
    assert job["size_bytes"] == len(content)

    # The speech engine received the uploaded bytes, and its text went to the LLM.
    assert transcriber.audio_seen == [content]
    assert extractor.calls[0][0] == SPOKEN_TRANSCRIPT
    result = job["result"]
    assert result["transcription"] == {"engine": "fake-stt", "languages": ["ar", "fr"], "text": SPOKEN_TRANSCRIPT}
    assert [item["owner"] for item in result["action_items"]] == ["Amira"]

    assert leftover_files(tmp_path) == []  # the recording is deleted once it has been read


def test_recording_is_assumed_to_be_derja_and_the_llm_is_told_it_is_speech(audio_client, extractor, transcriber):
    upload_audio(audio_client)
    assert transcriber.calls[0][2].language is SpokenLanguage.DERJA
    assert extractor.calls[0][1].spoken_language is SpokenLanguage.DERJA


def test_spoken_language_and_participants_reach_both_the_engine_and_the_llm(audio_client, extractor, transcriber):
    upload_audio(audio_client, spoken_language="fr", participants=" Amira Ben Salah ,Youssef;; Amira Ben Salah\nSarah ")

    hints = transcriber.calls[0][2]
    assert hints.language is SpokenLanguage.FRENCH
    assert hints.keywords == ("Amira Ben Salah", "Youssef", "Sarah")
    context = extractor.calls[0][1]
    assert context.spoken_language is SpokenLanguage.FRENCH
    assert context.participants == ("Amira Ben Salah", "Youssef", "Sarah")


def test_server_wide_keywords_are_added_to_the_participants(make_client, transcriber):
    client = make_client(with_stt=True, stt_keywords="Notion, Zitouna Bank")
    upload_audio(client, participants="Amira")
    assert transcriber.calls[0][2].keywords == ("Amira", "Notion", "Zitouna Bank")


def test_text_upload_is_unchanged_and_can_carry_participants(audio_client, extractor, transcriber):
    job = audio_client.get(upload(audio_client, participants="Amira, Youssef").json()["status_url"]).json()
    assert job["media"] == "text"
    assert job["result"]["transcription"] is None
    assert transcriber.calls == []
    context = extractor.calls[0][1]
    assert context.participants == ("Amira", "Youssef")
    assert context.spoken_language is None


def test_recording_is_refused_when_speech_to_text_is_not_configured(client):
    response = upload_audio(client)
    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]


def test_health_reports_speech_to_text(audio_client):
    assert audio_client.get("/health").json()["integrations"]["speech_to_text"] is True


def test_recording_over_the_limit_is_refused_and_not_left_on_disk(make_client, transcriber, tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    client = make_client(with_stt=True, max_audio_upload_bytes=1_000)
    response = upload_audio(client, content=MP3_BYTES + b"\x00" * 1_000)
    assert response.status_code == 413
    assert "Compress it" in response.json()["detail"]
    assert transcriber.calls == []
    assert leftover_files(tmp_path) == []


def test_recording_limit_is_separate_from_the_transcript_limit(make_client):
    client = make_client(with_stt=True, max_audio_upload_bytes=50_000)  # transcripts stay capped at 2,000 bytes
    assert upload_audio(client, content=MP3_BYTES + b"\x00" * 10_000).status_code == 202
    assert upload(client, content=b"x" * 10_000).status_code == 413


@pytest.mark.parametrize(
    ("content", "filename"),
    [
        (b"Amira: this is really a text file", "notes.mp3"),  # a renamed text file
        (MP3_BYTES, "meeting.m4a"),  # an MP3 under the wrong name
        (b"", "empty.mp3"),
    ],
)
def test_file_that_is_not_the_audio_it_claims_to_be_is_refused(audio_client, transcriber, tmp_path, content, filename):
    response = upload_audio(audio_client, content=content, filename=filename)
    assert response.status_code == 422
    assert transcriber.calls == []
    assert leftover_files(tmp_path) == []


def test_other_audio_formats_are_not_accepted_yet(audio_client):
    assert upload_audio(audio_client, content=b"RIFF....WAVE", filename="meeting.wav").status_code == 415


def test_same_recording_twice_returns_the_first_job(audio_client, transcriber, tmp_path):
    first = upload_audio(audio_client)
    second = upload_audio(audio_client)
    assert second.status_code == 200
    assert second.json() == {**first.json(), "status": "completed", "duplicate": True}
    assert len(transcriber.calls) == 1
    assert leftover_files(tmp_path) == []

    assert upload_audio(audio_client, force="true").status_code == 202
    assert len(transcriber.calls) == 2


def test_failed_transcription_fails_the_job_with_the_engines_explanation(make_client, transcriber, tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    transcriber.error = TranscriptionError("No speech was recognised in the recording.")
    client = make_client(with_stt=True)

    job = client.get(upload_audio(client).json()["status_url"]).json()
    assert job["status"] == "failed"
    assert job["error"] == "No speech was recognised in the recording."
    assert job["result"] is None
    assert leftover_files(tmp_path) == []  # deleted even when transcription fails


@pytest.mark.parametrize(
    ("extension", "head", "expected"),
    [
        (".mp3", b"ID3\x04\x00", True),
        (".mp3", b"\xff\xfb\x90\x64", True),  # no tag: starts with an MPEG audio frame
        (".mp3", b"\xff\x00\x00\x00", False),
        (".m4a", b"\x00\x00\x00\x20ftypM4A ", True),
        (".m4a", b"ID3\x04\x00\x00\x00\x00", False),
        (".wav", b"RIFF", False),
    ],
)
def test_audio_is_recognised_by_its_first_bytes(extension, head, expected):
    assert looks_like_audio(extension, head) is expected


# --- The hosted engine (OpenAI's transcription API) ---


def transcribe(handler, tmp_path, *, hints=SpeechHints(), model="gpt-transcribe", filename="meeting.m4a", **options):
    """Run OpenAITranscriber against a fake API; returns (transcription, delays slept)."""
    audio = tmp_path / "upload.bin"
    audio.write_bytes(M4A_BYTES)
    delays: list[float] = []

    async def record_sleep(seconds: float) -> None:
        delays.append(seconds)

    async def scenario() -> Transcription:
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http:
            engine = OpenAITranscriber(http, api_key="sk-test", model=model, sleep=record_sleep, **options)
            return await engine.transcribe(audio, filename, hints)

    return asyncio.run(scenario()), delays


def capture(requests: list, response: httpx2.Response):
    def handler(request: httpx2.Request) -> httpx2.Response:
        request.read()  # the multipart body is a stream until it is read
        requests.append(request)
        return response

    return handler


def field(name: str, value: str) -> bytes:
    """How one text field appears inside a multipart body."""
    return f'name="{name}"\r\n\r\n{value}\r\n'.encode()


def test_gpt_transcribe_gets_every_language_to_expect_and_the_names(tmp_path):
    requests: list[httpx2.Request] = []
    answer = json_response(200, {"text": " باش نبعثو الـ rapport غدوة ", "languages": [{"code": "ar"}, {"code": "fr"}]})
    hints = SpeechHints(language=SpokenLanguage.DERJA, keywords=("Amira Ben Salah", "Zitouna <Bank>\n"))

    transcription, _ = transcribe(capture(requests, answer), tmp_path, hints=hints)

    assert transcription == Transcription(engine="gpt-transcribe", languages=["ar", "fr"], text="باش نبعثو الـ rapport غدوة")
    request = requests[0]
    assert str(request.url) == "https://api.openai.com/v1/audio/transcriptions"
    assert request.headers["authorization"] == "Bearer sk-test"
    assert request.headers["content-type"].startswith("multipart/form-data")

    body = request.content
    assert field("model", "gpt-transcribe") in body
    for code in ("ar", "fr", "en"):
        assert field("languages[]", code) in body
    assert field("keywords[]", "Amira Ben Salah") in body
    assert field("keywords[]", "Zitouna Bank") in body  # characters the API rejects are removed
    assert field("prompt", DERJA_CONTEXT) in body
    assert b'name="language"\r\n' not in body
    assert b'name="file"; filename="meeting.m4a"\r\nContent-Type: audio/mp4' in body
    assert M4A_BYTES in body


def test_whisper_style_models_get_one_language_and_a_priming_prompt(tmp_path):
    requests: list[httpx2.Request] = []
    hints = SpeechHints(language=SpokenLanguage.DERJA, keywords=("Amira", "Youssef"))
    transcribe(
        capture(requests, json_response(200, {"text": "ok"})),
        tmp_path,
        hints=hints,
        model="whisper-large-v3",
        base_url="http://127.0.0.1:9000/v1/",
        filename="meeting.mp3",
    )

    request = requests[0]
    assert str(request.url) == "http://127.0.0.1:9000/v1/audio/transcriptions"
    body = request.content
    assert field("language", "ar") in body
    assert field("prompt", f"{DERJA_STYLE_SAMPLE} Amira, Youssef") in body
    assert b"languages[]" not in body
    assert b"keywords[]" not in body
    assert b"Content-Type: audio/mpeg" in body


def test_detect_mode_sends_no_language_at_all(tmp_path):
    requests: list[httpx2.Request] = []
    transcribe(capture(requests, json_response(200, {"text": "ok"})), tmp_path, hints=SpeechHints(language=SpokenLanguage.AUTO))
    assert b"languages[]" not in requests[0].content
    assert b'name="prompt"' not in requests[0].content


@pytest.mark.parametrize(
    ("response", "expected_message"),
    [
        (json_response(200, {"text": "   "}), "No speech was recognised"),
        (json_response(200, {"unexpected": True}), "unexpected format"),
        (json_response(401, {"error": {"message": "Incorrect API key provided"}}), "OPENAI_API_KEY"),
        (json_response(404, {"error": {"message": "model not found"}}), "STT_MODEL"),
        (json_response(413, {"error": {"message": "too big"}}), "too large"),
        (json_response(400, {"error": {"message": "Unsupported language code: xx"}}), "rejected the recording: Unsupported language code: xx"),
        (httpx2.Response(500), "HTTP 500"),
    ],
)
def test_engine_failures_become_clear_errors(tmp_path, response, expected_message):
    with pytest.raises(TranscriptionError, match=expected_message):
        transcribe(lambda request: response, tmp_path)


def test_rate_limit_is_retried_after_the_delay_the_api_asks_for(tmp_path):
    responses = [httpx2.Response(429, headers={"Retry-After": "3"}), json_response(200, {"text": "ok"})]
    transcription, delays = transcribe(lambda request: responses.pop(0), tmp_path)
    assert transcription.text == "ok"
    assert delays == [3.0]


def test_network_failures_are_retried_then_reported(tmp_path):
    attempts = 0

    def unreachable(request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        attempts += 1
        raise httpx2.ConnectError("no route", request=request)

    with pytest.raises(TranscriptionError, match="Could not reach"):
        transcribe(unreachable, tmp_path)
    assert attempts == 3


# --- The local engine (a Hugging Face model on this machine) ---


def test_local_engine_explains_which_packages_to_install(tmp_path, monkeypatch):
    monkeypatch.setattr(local_whisper, "missing_packages", lambda: ["torch", "av"])
    engine = LocalWhisperTranscriber("some/model")
    with pytest.raises(TranscriptionError, match=r"torch, av.*requirements-local-stt.txt"):
        asyncio.run(engine.transcribe(tmp_path / "a.mp3", "a.mp3", SpeechHints()))


def test_local_engine_decodes_then_runs_the_model_with_the_right_language(tmp_path, monkeypatch):
    calls: list[dict] = []

    def fake_recognizer(audio, **options):
        calls.append({"audio": audio, **options})
        return {"text": "  باش نبعثو الرابور غدوة  "}

    loaded: list[str] = []
    monkeypatch.setattr(local_whisper, "missing_packages", lambda: [])
    monkeypatch.setattr(local_whisper, "load_recognizer", lambda model_id: loaded.append(model_id) or fake_recognizer)
    monkeypatch.setattr(local_whisper, "decode_audio", lambda path: [0.0] * 16_000)

    engine = LocalWhisperTranscriber("oddadmix/Whisperv3-tunisian-codeswitch")

    async def twice() -> list[Transcription]:
        hints = SpeechHints(language=SpokenLanguage.DERJA)
        return [await engine.transcribe(tmp_path / "a.mp3", "a.mp3", hints) for _ in range(2)]

    first, _ = asyncio.run(twice())
    assert first == Transcription(engine="oddadmix/Whisperv3-tunisian-codeswitch", languages=["ar"], text="باش نبعثو الرابور غدوة")
    assert loaded == ["oddadmix/Whisperv3-tunisian-codeswitch"]  # the model is loaded once, then reused
    assert calls[0]["audio"]["sampling_rate"] == 16_000
    assert calls[0]["generate_kwargs"] == {"task": "transcribe", "language": "ar"}
    assert calls[0]["chunk_length_s"] == 30
    assert calls[0]["batch_size"] == 1  # no GPU here
