"""Phase 3 tests: mapping action items to Notion pages, and talking to the Notion API.

`FakeNotion` (see helpers.py) plays the Notion server.
"""

import asyncio
import json
from datetime import date

import httpx2
import pytest
from helpers import NOTION_COLUMNS, FakeNotion, json_response, make_settings, route_by_host, upload

from app.schemas.extraction import Priority, VerifiedActionItem
from app.schemas.jobs import NotionSync
from app.services.notion import NOTION_VERSION, NotionColumns, NotionError, NotionTaskWriter, build_page

COLUMNS = NotionColumns(
    data_source_id="ds-1", title="Name", owner="Owner", due="Due", priority="Priority", meeting="Meeting"
)


def make_item(**changes) -> VerifiedActionItem:
    fields = {
        "title": "Ship the billing fix",
        "description": "Annual customers are charged twice.",
        "owner": "Omar Haddad",
        "due_date": date(2026, 11, 10),
        "priority": Priority.HIGH,
        "source_quote": "I'll ship a fix by end of day tomorrow.",
        "quote_verified": True,
    }
    return VerifiedActionItem(**{**fields, **changes})


def text_of(rich_text: list[dict]) -> str:
    return "".join(part["text"]["content"] for part in rich_text)


# --- Settings ---


@pytest.mark.parametrize(
    "given",
    [
        "1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d",
        "1a2b3c4d-5e6f-7a8b-9c0d-1e2f3a4b5c6d",
        "https://www.notion.so/acme/Tasks-Cafe-1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d?v=ffffffffffffffffffffffffffffffff",
        "  1A2B3C4D5E6F7A8B9C0D1E2F3A4B5C6D  ",
    ],
)
def test_database_id_can_be_pasted_in_any_form(given):
    assert make_settings(notion_database_id=given).notion_database_id == "1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d"


def test_notion_counts_as_configured_only_with_both_token_and_database():
    assert make_settings(notion_token="secret").notion_configured is False
    assert make_settings(notion_token="secret", notion_database_id="db-1").notion_configured is True


# --- Mapping ---


def test_action_item_maps_to_page_properties_and_body():
    page = build_page(make_item(), COLUMNS, meeting_title="Weekly Sync", meeting_date=date(2026, 11, 9))

    assert page["parent"] == {"type": "data_source_id", "data_source_id": "ds-1"}
    properties = page["properties"]
    assert text_of(properties["Name"]["title"]) == "Ship the billing fix"
    assert text_of(properties["Owner"]["rich_text"]) == "Omar Haddad"
    assert properties["Due"] == {"date": {"start": "2026-11-10"}}
    assert properties["Priority"] == {"select": {"name": "High"}}
    assert text_of(properties["Meeting"]["rich_text"]) == "Weekly Sync"

    description, quote, origin = page["children"]
    assert text_of(description["paragraph"]["rich_text"]) == "Annual customers are charged twice."
    assert text_of(quote["quote"]["rich_text"]) == "I'll ship a fix by end of day tomorrow."
    assert text_of(origin["paragraph"]["rich_text"]) == 'From the meeting "Weekly Sync" on 2026-11-09.'


def test_unknown_owner_and_date_are_left_out_rather_than_sent_empty():
    page = build_page(make_item(owner=None, due_date=None), COLUMNS, meeting_title="Sync", meeting_date=None)
    assert "Owner" not in page["properties"]
    assert "Due" not in page["properties"]


def test_optional_meeting_column_is_skipped_when_the_database_lacks_it():
    columns = NotionColumns(data_source_id="ds-1", title="Task", owner="Owner", due="Due", priority="Priority", meeting=None)
    page = build_page(make_item(), columns, meeting_title="Sync", meeting_date=None)
    assert set(page["properties"]) == {"Task", "Owner", "Due", "Priority"}


def test_unverified_quote_is_flagged_in_the_page():
    page = build_page(make_item(quote_verified=False), COLUMNS, meeting_title="Sync", meeting_date=None)
    assert "Needs review" in text_of(page["children"][2]["paragraph"]["rich_text"])


def test_long_text_is_split_at_notions_2000_character_limit():
    page = build_page(make_item(description="x" * 4500), COLUMNS, meeting_title="Sync", meeting_date=None)
    parts = page["children"][0]["paragraph"]["rich_text"]
    assert [len(part["text"]["content"]) for part in parts] == [2000, 2000, 500]


# --- Talking to the API ---


def run_writer(notion: FakeNotion, items=None, *, check_only: bool = False, handler=None, **writer_options):
    """Use a NotionTaskWriter against the fake server; returns (result, delays slept)."""
    delays: list[float] = []

    async def record_sleep(seconds: float) -> None:
        delays.append(seconds)

    async def scenario():
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler or notion)) as http:
            writer = NotionTaskWriter(
                http, token="secret-token", database_id="db-1", sleep=record_sleep, **writer_options
            )
            if check_only:
                return await writer.check_database()
            return await writer.create_tasks(
                items if items is not None else [make_item()],
                meeting_title="Weekly Sync",
                meeting_date=date(2026, 11, 9),
            )

    return asyncio.run(scenario()), delays


def paths(notion: FakeNotion) -> list[str]:
    return [f"{request.method} {request.url.path}" for request in notion.requests]


def test_database_is_resolved_to_its_table_and_columns():
    notion = FakeNotion()
    columns, _ = run_writer(notion, check_only=True)

    assert columns == COLUMNS
    assert paths(notion) == ["GET /v1/databases/db-1", "GET /v1/data_sources/ds-1"]
    request = notion.requests[0]
    assert request.headers["authorization"] == "Bearer secret-token"
    assert request.headers["notion-version"] == NOTION_VERSION


def test_title_column_is_found_whatever_its_name():
    columns = {**NOTION_COLUMNS, "Task": {"id": "title", "name": "Task", "type": "title", "title": {}}}
    del columns["Name"]
    resolved, _ = run_writer(FakeNotion(columns=columns), check_only=True)
    assert resolved.title == "Task"


def test_configured_data_source_skips_the_database_lookup():
    notion = FakeNotion()
    run_writer(notion, check_only=True, data_source_id="ds-9")
    assert paths(notion) == ["GET /v1/data_sources/ds-9"]


def test_database_with_several_tables_asks_which_one():
    notion = FakeNotion(data_sources=[{"id": "ds-1", "name": "Tasks"}, {"id": "ds-2", "name": "Archive"}])
    with pytest.raises(NotionError, match=r"NOTION_DATA_SOURCE_ID.*Tasks \(ds-1\), Archive \(ds-2\)"):
        run_writer(notion, check_only=True)


def test_wrong_columns_are_explained():
    columns = {
        "Name": NOTION_COLUMNS["Name"],
        "Owner": {"id": "a", "name": "Owner", "type": "people", "people": {}},
        "Priority": NOTION_COLUMNS["Priority"],
    }
    with pytest.raises(NotionError) as error:
        run_writer(FakeNotion(columns=columns), check_only=True)
    message = str(error.value)
    assert 'change the "Owner" column to type Text (it is people)' in message
    assert 'add a "Due" column of type Date' in message


def test_tasks_are_created_one_page_each():
    notion = FakeNotion()
    items = [make_item(), make_item(title="Write the tracking spec", owner=None)]
    sync, _ = run_writer(notion, items)

    assert sync.status == "created"
    assert [task.url for task in sync.tasks] == ["https://www.notion.so/page-1", "https://www.notion.so/page-2"]
    assert [text_of(page["properties"]["Name"]["title"]) for page in notion.pages] == [
        "Ship the billing fix",
        "Write the tracking spec",
    ]


def test_no_action_items_means_no_notion_call():
    notion = FakeNotion()
    sync, _ = run_writer(notion, [])
    assert sync.status == "skipped"
    assert notion.requests == []


def test_rate_limit_waits_as_long_as_notion_asks_then_retries():
    notion = FakeNotion()
    notion.next_page_responses = [
        json_response(429, {"object": "error", "status": 429, "code": "rate_limited", "message": "slow"}, {"Retry-After": "7"})
    ]
    sync, delays = run_writer(notion)
    assert sync.status == "created"
    assert delays == [7.0]
    assert len(notion.pages) == 1


def test_a_write_is_not_repeated_when_it_may_have_gone_through():
    notion = FakeNotion()
    notion.next_page_responses = [json_response(500, {"object": "error", "status": 500, "code": "internal_server_error", "message": "oops"})]
    sync, delays = run_writer(notion)
    assert sync.status == "failed"
    assert delays == []  # no retry: the page might exist already
    assert "HTTP 500" in sync.tasks[0].error


def test_a_write_is_retried_when_it_never_reached_notion():
    notion = FakeNotion()
    attempts = 0

    def flaky(request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        if request.method == "POST":
            attempts += 1
            if attempts == 1:
                raise httpx2.ConnectError("connection refused", request=request)
        return notion(request)

    sync, delays = run_writer(notion, handler=flaky)
    assert sync.status == "created"
    assert delays == [1.0]


def test_a_read_is_retried_on_a_server_error():
    notion = FakeNotion()
    failed_once = False

    def flaky(request: httpx2.Request) -> httpx2.Response:
        nonlocal failed_once
        if request.method == "GET" and not failed_once:
            failed_once = True
            return httpx2.Response(502)
        return notion(request)

    columns, delays = run_writer(notion, check_only=True, handler=flaky)
    assert columns == COLUMNS
    assert delays == [1.0]


def test_one_rejected_task_does_not_stop_the_others():
    notion = FakeNotion()
    notion.next_page_responses = [
        json_response(400, {"object": "error", "status": 400, "code": "validation_error", "message": "Due is expected to be date."})
    ]
    sync, _ = run_writer(notion, [make_item(), make_item(title="Second task")])

    assert sync.status == "partial"
    assert sync.detail == "1 of 2 tasks were created in Notion."
    assert sync.tasks[0].error == "Notion rejected the request: Due is expected to be date."
    assert sync.tasks[1].url == "https://www.notion.so/page-1"


def test_a_setup_problem_stops_after_the_first_failure():
    notion = FakeNotion()
    notion.next_page_responses = [json_response(401, {"object": "error", "status": 401, "code": "unauthorized", "message": "bad token"})]
    sync, _ = run_writer(notion, [make_item(), make_item(title="Second"), make_item(title="Third")])

    assert sync.status == "failed"
    assert [task.error for task in sync.tasks] == ["Notion rejected the token. Check NOTION_TOKEN."] * 3
    assert paths(notion).count("POST /v1/pages") == 1  # the other two were not attempted


def test_unshared_database_explains_how_to_connect_it():
    def not_found(request: httpx2.Request) -> httpx2.Response:
        return json_response(404, {"object": "error", "status": 404, "code": "object_not_found", "message": "nope"})

    sync, _ = run_writer(FakeNotion(), handler=not_found)
    assert sync == NotionSync(
        status="failed",
        detail="Notion could not find the database. Check NOTION_DATABASE_ID and connect the "
        "database to your integration (database menu > Connections).",
    )


# --- Through the whole app ---

NOTION_SETTINGS = {"notion_token": "secret-token", "notion_database_id": "db-1"}


def test_upload_ends_with_tasks_in_notion(make_client):
    notion = FakeNotion()
    client = make_client(transport=route_by_host({"api.notion.com": notion}), **NOTION_SETTINGS)
    assert client.get("/health").json()["integrations"]["notion"] is True

    accepted = upload(client, meeting_title="Weekly standup", meeting_date="2026-11-09").json()
    job = client.get(accepted["status_url"]).json()

    assert job["status"] == "completed"
    notion_result = job["result"]["notion"]
    assert notion_result["status"] == "created"
    assert [task["url"] for task in notion_result["tasks"]] == ["https://www.notion.so/page-1", "https://www.notion.so/page-2"]
    assert len(notion.pages) == 2
    assert text_of(notion.pages[0]["properties"]["Meeting"]["rich_text"]) == "Weekly standup"
    assert text_of(notion.pages[0]["properties"]["Owner"]["rich_text"]) == "Amira"


def test_notion_failure_fails_the_job_but_keeps_the_extracted_items(make_client):
    notion = FakeNotion()
    notion.next_page_responses = [
        json_response(400, {"object": "error", "status": 400, "code": "validation_error", "message": "Bad property."})
    ]
    client = make_client(transport=route_by_host({"api.notion.com": notion}), **NOTION_SETTINGS)

    job = client.get(upload(client).json()["status_url"]).json()
    assert job["status"] == "failed"
    assert job["error"] == "Notion: 1 of 2 tasks were created in Notion."
    assert len(job["result"]["action_items"]) == 2  # nothing has to be extracted again
    assert job["result"]["notion"]["status"] == "partial"


def test_app_starts_even_when_the_notion_database_is_misconfigured(make_client):
    notion = FakeNotion(columns={"Name": NOTION_COLUMNS["Name"]})  # none of the required columns
    client = make_client(transport=route_by_host({"api.notion.com": notion}), **NOTION_SETTINGS)
    assert client.get("/health").status_code == 200

    job = client.get(upload(client).json()["status_url"]).json()
    assert job["status"] == "failed"
    assert 'add a "Owner" column of type Text' in job["error"]
    assert notion.pages == []


def test_page_bodies_sent_to_notion_are_valid_json_with_the_expected_parent(make_client):
    notion = FakeNotion()
    client = make_client(transport=route_by_host({"api.notion.com": notion}), **NOTION_SETTINGS)
    upload(client)

    post = next(request for request in notion.requests if request.method == "POST")
    assert post.headers["content-type"] == "application/json"
    assert json.loads(post.content)["parent"] == {"type": "data_source_id", "data_source_id": "ds-1"}
