"""Tests for the Groq extractor, against a fake chat-completions server."""

import asyncio
import json
from datetime import date

import httpx2
import pytest
from helpers import TRANSCRIPT, json_response, route_by_host, upload

from app.schemas.extraction import MeetingContext, Priority
from app.services.extraction import ExtractionError, build_system_prompt
from app.services.groq_extractor import GroqExtractor, strict_schema

CONTEXT = MeetingContext(title="Weekly Sync", meeting_date=date(2026, 11, 9))

ANSWER = {
    "meeting_title": "Weekly Sync",
    "summary": "The budget report goes out tomorrow.",
    "action_items": [
        {
            "title": "Send the budget report",
            "description": "Amira sends it.",
            "owner": "Amira",
            "due_date": "2026-11-10",
            "priority": "high",
            "source_quote": "I will send the budget by Friday.",
        }
    ],
}


def completion(content: str, *, finish_reason: str = "stop") -> httpx2.Response:
    return json_response(
        200,
        {
            "model": "openai/gpt-oss-120b",
            "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 900, "completion_tokens": 120},
        },
    )


def run(handler, **options):
    delays: list[float] = []

    async def sleep(seconds: float) -> None:
        delays.append(seconds)

    async def scenario():
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http:
            extractor = GroqExtractor(http, api_key="gsk-test", sleep=sleep, **options)
            return await extractor.extract(TRANSCRIPT, CONTEXT)

    return asyncio.run(scenario()), delays


def test_request_uses_strict_json_schema_and_the_shared_prompts():
    seen: list[httpx2.Request] = []

    def handler(request):
        seen.append(request)
        return completion(json.dumps(ANSWER))

    outcome, _ = run(handler, task_language="French")
    request = seen[0]
    assert str(request.url) == "https://api.groq.com/openai/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer gsk-test"
    body = json.loads(request.content)
    assert body["model"] == "openai/gpt-oss-120b"
    assert body["reasoning_effort"] == "medium"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["response_format"]["json_schema"]["schema"] == strict_schema()
    assert body["messages"][0]["content"] == build_system_prompt(CONTEXT, "French")
    assert "<transcript>" in body["messages"][1]["content"]

    assert (outcome.model, outcome.input_tokens, outcome.output_tokens) == ("openai/gpt-oss-120b", 900, 120)
    item = outcome.extraction.action_items[0]
    assert (item.owner, item.due_date, item.priority) == ("Amira", date(2026, 11, 10), Priority.HIGH)


def test_schema_meets_strict_mode_rules():
    schema = strict_schema()

    def objects(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                yield node
            for value in node.values():
                yield from objects(value)
        elif isinstance(node, list):
            for item in node:
                yield from objects(item)

    found = list(objects(schema))
    assert len(found) == 2  # the meeting and the action item
    for obj in found:
        assert obj["additionalProperties"] is False
        assert set(obj["required"]) == set(obj["properties"])
    assert "format" not in json.dumps(schema)
    assert "anyOf" in schema["$defs"]["ActionItem"]["properties"]["owner"]  # "may be null" is kept


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (completion("{}", finish_reason="stop"), "unexpected format"),
        (completion('{"meeting_title": "Cut', finish_reason="length"), "cut off"),
        (json_response(200, {"unexpected": True}), "unexpected format"),
        (json_response(401, {"error": {"message": "Invalid API Key"}}), "GROQ_API_KEY"),
        (json_response(404, {"error": {"message": "no model"}}), "GROQ_MODEL"),
        (json_response(413, {"error": {"message": "Request too large"}}), "per-minute token limit"),
        (json_response(400, {"error": {"message": "Generated JSON does not match"}}), "rejected the request: Generated JSON"),
        (httpx2.Response(418), "HTTP 418"),
    ],
)
def test_failures_become_clear_errors(response, message):
    with pytest.raises(ExtractionError, match=message):
        run(lambda request: response)


def test_rate_limit_is_retried_after_the_delay_groq_asks_for():
    responses = [httpx2.Response(429, headers={"retry-after": "4"}), completion(json.dumps(ANSWER))]
    outcome, delays = run(lambda request: responses.pop(0))
    assert outcome.extraction.meeting_title == "Weekly Sync"
    assert delays == [4.0]


def test_a_daily_quota_is_reported_instead_of_waited_out():
    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1
        return httpx2.Response(429, headers={"retry-after": "3600"})

    with pytest.raises(ExtractionError, match="rate limit"):
        run(handler)
    assert attempts == 1


def test_network_failure_is_retried_then_reported():
    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1
        raise httpx2.ConnectError("down", request=request)

    with pytest.raises(ExtractionError, match="Could not reach"):
        run(handler)
    assert attempts == 3


def test_whole_app_with_groq_provider(make_client):
    seen: list[httpx2.Request] = []

    def groq(request):
        seen.append(request)
        return completion(json.dumps(ANSWER))

    client = make_client(
        with_llm=False,
        transport=route_by_host({"api.groq.com": groq}),
        llm_provider="groq",
        groq_api_key="gsk-test",
    )
    assert client.get("/health").json()["integrations"]["llm"] is True
    job = client.get(upload(client).json()["status_url"]).json()
    assert job["status"] == "completed"
    assert job["result"]["llm"]["model"] == "openai/gpt-oss-120b"
    assert job["result"]["action_items"][0]["quote_verified"] is True  # quote is in TRANSCRIPT
    assert len(seen) == 1


def test_groq_provider_without_a_key_leaves_the_llm_off(make_client):
    client = make_client(with_llm=False, llm_provider="groq", anthropic_api_key="would-be-ignored")
    assert client.get("/health").json()["integrations"]["llm"] is False
