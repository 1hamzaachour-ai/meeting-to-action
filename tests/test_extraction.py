"""Phase 2 tests: the prompt, the quote check, and the Anthropic call.

The real `anthropic` SDK is used, but its HTTP layer is replaced by a handler
that answers like the Messages API (a server-sent event stream).
"""

import asyncio
import json
from datetime import date

import anthropic
import httpx2
import pytest

from app.schemas.extraction import ActionItem, MeetingContext, Priority
from app.services.anthropic_extractor import OUTPUT_FORMAT, AnthropicExtractor
from app.schemas.transcription import SpokenLanguage
from app.services.extraction import (
    DERJA_GUIDANCE,
    SPEECH_GUIDANCE,
    SYSTEM_PROMPT,
    ExtractionError,
    ExtractionOutcome,
    build_system_prompt,
    build_user_message,
    verify_quotes,
)

TRANSCRIPT = "Omar: I'll ship a fix by end of day tomorrow.\nKarim: Someone should write that spec."
CONTEXT = MeetingContext(title="Weekly Product Sync", meeting_date=date(2026, 11, 9))

ANSWER = {
    "meeting_title": "Weekly Product Sync",
    "summary": "The team agreed to fix the billing bug first.",
    "action_items": [
        {
            "title": "Ship the billing fix",
            "description": "Annual customers are charged twice.",
            "owner": "Omar",
            "due_date": "2026-11-10",
            "priority": "high",
            "source_quote": "I'll ship a fix by end of day tomorrow.",
        },
        {
            "title": "Write the tracking spec",
            "description": "Nobody owns it yet.",
            "owner": None,
            "due_date": None,
            "priority": "medium",
            "source_quote": "Someone should write that spec.",
        },
    ],
}


def streamed_message(
    text: str, *, stop_reason: str = "end_turn", model: str = "claude-opus-5-5", with_thinking: bool = False
) -> httpx2.Response:
    """A Messages API reply in its streaming wire format."""
    events: list[dict] = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 1200, "output_tokens": 1},
            },
        }
    ]
    index = 0
    if with_thinking:
        events += [
            {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig"}},
            {"type": "content_block_stop", "index": 0},
        ]
        index = 1
    middle = len(text) // 2  # sent in two pieces, as a real stream would
    events += [
        {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": text[:middle]}},
        {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": text[middle:]}},
        {"type": "content_block_stop", "index": index},
        {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None}, "usage": {"output_tokens": 350}},
        {"type": "message_stop"},
    ]
    body = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
    return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())


def api_error(status: int, error_type: str, message: str) -> httpx2.Response:
    return httpx2.Response(status, json={"type": "error", "error": {"type": error_type, "message": message}})


def extract(handler) -> ExtractionOutcome:
    """Run AnthropicExtractor.extract against a fake API."""

    async def scenario() -> ExtractionOutcome:
        client = anthropic.AsyncAnthropic(
            api_key="test-key",
            max_retries=0,
            http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)),
        )
        try:
            extractor = AnthropicExtractor(client, model="claude-opus-5-5", effort="medium", max_tokens=32_000)
            return await extractor.extract(TRANSCRIPT, CONTEXT)
        finally:
            await client.close()

    return asyncio.run(scenario())


def test_request_asks_for_structured_output():
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return streamed_message(json.dumps(ANSWER))

    extract(handler)

    request = seen[0]
    assert request.url.path == "/v1/messages"
    assert request.headers["authorization"] == "Bearer test-key"
    assert "server-side-fallback-2026-07-01" in request.headers["anthropic-beta"]

    body = json.loads(request.content)
    assert body["model"] == "claude-opus-5-5"
    assert body["max_tokens"] == 32_000
    assert body["stream"] is True
    assert body["system"] == build_system_prompt(CONTEXT)
    assert body["output_config"] == {"effort": "medium", "format": OUTPUT_FORMAT}
    assert body["fallbacks"] == "default"
    assert "temperature" not in body  # sampling parameters are rejected by current models

    user_message = body["messages"][0]["content"]
    assert "Meeting title: Weekly Product Sync" in user_message
    assert "Meeting date: 2026-11-09 (Monday)" in user_message
    assert f"<transcript>\n{TRANSCRIPT}\n</transcript>" in user_message


@pytest.mark.parametrize("with_thinking", [False, True])
def test_answer_becomes_typed_action_items(with_thinking):
    outcome = extract(lambda request: streamed_message(json.dumps(ANSWER), with_thinking=with_thinking))

    assert outcome.model == "claude-opus-5-5"
    assert (outcome.input_tokens, outcome.output_tokens) == (1200, 350)
    assert outcome.extraction.meeting_title == "Weekly Product Sync"

    first, second = outcome.extraction.action_items
    assert first.owner == "Omar"
    assert first.due_date == date(2026, 11, 10)
    assert first.priority is Priority.HIGH
    assert second.owner is None
    assert second.due_date is None


@pytest.mark.parametrize(
    ("response", "expected_message"),
    [
        (streamed_message("", stop_reason="refusal"), "declined"),
        (streamed_message('{"meeting_title": "Cut', stop_reason="max_tokens"), "cut off"),
        (streamed_message('{"unexpected": true}'), "unexpected format"),
        (api_error(401, "authentication_error", "invalid x-api-key"), "ANTHROPIC_API_KEY"),
        (api_error(403, "permission_error", "no access"), "not allowed"),
        (api_error(404, "not_found_error", "model: nope"), "LLM_MODEL"),
        (api_error(429, "rate_limit_error", "slow down"), "rate limit"),
        (api_error(400, "invalid_request_error", "prompt is too long"), "rejected the request: prompt is too long"),
        (api_error(500, "api_error", "internal"), "HTTP 500"),
    ],
)
def test_failures_become_clear_errors(response, expected_message):
    with pytest.raises(ExtractionError, match=expected_message):
        extract(lambda request: response)


def test_network_failure_becomes_a_clear_error():
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("no route to host", request=request)

    with pytest.raises(ExtractionError, match="Could not reach"):
        extract(handler)


# --- Prompt ---


def test_user_message_says_when_the_date_is_unknown():
    message = build_user_message("Hello", MeetingContext())
    assert "Meeting title: not provided" in message
    assert "Meeting date: not provided" in message


def test_user_message_carries_participants_and_the_transcripts_origin():
    context = MeetingContext(participants=("Amira Ben Salah", "Youssef"), spoken_language=SpokenLanguage.DERJA)
    message = build_user_message("...", context)
    assert "Participants: Amira Ben Salah, Youssef" in message
    assert "Transcript source: automatic speech recognition" in message


def test_instructions_for_a_text_transcript_have_no_speech_guidance():
    prompt = build_system_prompt(MeetingContext())
    assert prompt.startswith(SYSTEM_PROMPT)
    assert SPEECH_GUIDANCE not in prompt
    assert DERJA_GUIDANCE not in prompt
    assert prompt.endswith("in the main language of the transcript.")


@pytest.mark.parametrize(
    ("language", "expects_derja_guidance"),
    [(SpokenLanguage.DERJA, True), (SpokenLanguage.FRENCH, False), (SpokenLanguage.AUTO, False)],
)
def test_instructions_for_a_recording_add_speech_guidance_and_derja_only_when_spoken(language, expects_derja_guidance):
    prompt = build_system_prompt(MeetingContext(spoken_language=language))
    assert SPEECH_GUIDANCE in prompt
    assert (DERJA_GUIDANCE in prompt) is expects_derja_guidance


def test_task_language_setting_overrides_the_transcripts_language():
    prompt = build_system_prompt(MeetingContext(spoken_language=SpokenLanguage.DERJA), task_language="French")
    assert "Write titles, descriptions and the summary in French" in prompt
    assert "main language of the transcript" not in prompt


def test_schema_sent_to_the_model_is_strict():
    schema = OUTPUT_FORMAT["schema"]
    item = schema["$defs"]["ActionItem"]
    for object_schema in (schema, item):
        assert object_schema["additionalProperties"] is False
        assert set(object_schema["required"]) == set(object_schema["properties"])  # nothing optional
    assert {"type": "string", "format": "date"}.items() <= item["properties"]["due_date"]["anyOf"][0].items()
    assert schema["$defs"]["Priority"]["enum"] == ["high", "medium", "low"]


# --- Quote check ---


def item_quoting(quote: str) -> ActionItem:
    return ActionItem(
        title="Do it", description="Because.", owner=None, due_date=None, priority=Priority.LOW, source_quote=quote
    )


@pytest.mark.parametrize(
    ("quote", "expected"),
    [
        ("I'll ship a fix by end of day tomorrow.", True),
        ("i'll SHIP a fix by   end of day\ntomorrow", True),  # case and spacing do not matter
        ("I’ll ship a fix", True),  # typographic apostrophe
        ('"Someone should write that spec."', True),  # the model wrapped it in quotes
        ("Omar promised to ship the fix soon.", False),  # a paraphrase, not a quote
        ("", False),
    ],
)
def test_quote_check(quote, expected):
    [verified] = verify_quotes([item_quoting(quote)], TRANSCRIPT)
    assert verified.quote_verified is expected
    assert verified.source_quote == quote  # the item itself is unchanged
