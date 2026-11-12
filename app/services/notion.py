"""Phase 3: writing action items to a Notion database.

Three calls of the Notion REST API are used:
- GET  /v1/databases/{id}     which data source (table) the database holds
- GET  /v1/data_sources/{id}  that table's columns, checked once
- POST /v1/pages              one page (row) per action item
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

import httpx2

from app.errors import UserFacingError
from app.schemas.extraction import VerifiedActionItem
from app.schemas.jobs import NotionSync, NotionTask

logger = logging.getLogger(__name__)

NOTION_VERSION = "2026-03-11"
TEXT_LIMIT = 2000  # Notion's maximum characters in one rich-text object
MAX_ATTEMPTS = 4
MAX_RETRY_DELAY_SECONDS = 60

# The request was certainly not processed, so repeating it is always safe.
RETRY_ALWAYS = frozenset({429, 503, 529})
# The request may have been processed. Repeating a GET is harmless; repeating a
# POST could create the same page twice, so a POST is not retried on these.
RETRY_READS_ONLY = frozenset({500, 502, 504})


class NotionError(UserFacingError):
    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status

    @property
    def is_setup_problem(self) -> bool:
        """Bad token, database not shared, missing permission: every call would fail."""
        return self.status in (401, 403, 404)


@dataclass(frozen=True)
class NotionColumns:
    """Where to write, resolved from the user's database."""

    data_source_id: str
    title: str  # the database's title column, whatever it is called
    owner: str
    due: str
    priority: str
    meeting: str | None  # optional column


class NotionTaskWriter:
    def __init__(
        self,
        http: httpx2.AsyncClient,
        *,
        token: str,
        database_id: str,
        data_source_id: str | None = None,
        base_url: str = "https://api.notion.com",
        owner_property: str = "Owner",
        due_property: str = "Due",
        priority_property: str = "Priority",
        meeting_property: str = "Meeting",
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._http = http
        self._headers = {"Authorization": f"Bearer {token}", "Notion-Version": NOTION_VERSION}
        self._database_id = database_id
        self._data_source_id = data_source_id
        self._base_url = base_url.rstrip("/")
        self._owner_property = owner_property
        self._due_property = due_property
        self._priority_property = priority_property
        self._meeting_property = meeting_property
        self._sleep = sleep
        self._columns: NotionColumns | None = None

    async def check_database(self) -> NotionColumns:
        """Locate the table and verify its columns. The answer is cached once it succeeds."""
        if self._columns is None:
            self._columns = await self._load_columns()
        return self._columns

    async def create_tasks(
        self, items: list[VerifiedActionItem], *, meeting_title: str, meeting_date: date | None
    ) -> NotionSync:
        """Create one page per action item and report the outcome of each."""
        if not items:
            return NotionSync(status="skipped", detail="The meeting produced no action items.")
        try:
            columns = await self.check_database()
        except NotionError as exc:
            return NotionSync(status="failed", detail=str(exc))

        tasks: list[NotionTask] = []
        setup_problem: str | None = None
        for item in items:
            if setup_problem is not None:  # the next call would fail for the same reason
                tasks.append(NotionTask(title=item.title, error=setup_problem))
                continue
            page_request = build_page(item, columns, meeting_title=meeting_title, meeting_date=meeting_date)
            try:
                page = await self._request("POST", "/v1/pages", json=page_request)
            except NotionError as exc:
                logger.warning("Notion refused the task %r: %s", item.title, exc)
                tasks.append(NotionTask(title=item.title, error=str(exc)))
                if exc.is_setup_problem:
                    setup_problem = str(exc)
                continue
            tasks.append(NotionTask(title=item.title, page_id=page.get("id"), url=page.get("url")))

        created = sum(task.error is None for task in tasks)
        if created == len(tasks):
            return NotionSync(status="created", tasks=tasks)
        return NotionSync(
            status="partial" if created else "failed",
            detail=f"{created} of {len(tasks)} tasks were created in Notion.",
            tasks=tasks,
        )

    async def _load_columns(self) -> NotionColumns:
        data_source_id = self._data_source_id or await self._find_data_source()
        table = await self._request("GET", f"/v1/data_sources/{data_source_id}")
        columns: dict[str, Any] = table.get("properties") or {}

        title = next((name for name, column in columns.items() if column.get("type") == "title"), None)
        if title is None:
            raise NotionError("The Notion database has no title column.")

        required = (
            (self._owner_property, "rich_text", "Text"),
            (self._due_property, "date", "Date"),
            (self._priority_property, "select", "Select"),
        )
        changes = []
        for name, expected_type, label in required:
            actual_type = columns.get(name, {}).get("type")
            if actual_type is None:
                changes.append(f'add a "{name}" column of type {label}')
            elif actual_type != expected_type:
                changes.append(f'change the "{name}" column to type {label} (it is {actual_type})')
        if changes:
            raise NotionError("The Notion database needs these changes: " + "; ".join(changes) + ".")

        has_meeting_column = columns.get(self._meeting_property, {}).get("type") == "rich_text"
        return NotionColumns(
            data_source_id=data_source_id,
            title=title,
            owner=self._owner_property,
            due=self._due_property,
            priority=self._priority_property,
            meeting=self._meeting_property if has_meeting_column else None,
        )

    async def _find_data_source(self) -> str:
        database = await self._request("GET", f"/v1/databases/{self._database_id}")
        sources = database.get("data_sources") or []
        if len(sources) == 1:
            return sources[0]["id"]
        if not sources:
            raise NotionError("The Notion database has no data source to write to.")
        choices = ", ".join(f"{source.get('name') or 'untitled'} ({source['id']})" for source in sources)
        raise NotionError(
            f"The Notion database holds several data sources. Set NOTION_DATA_SOURCE_ID to one of: {choices}."
        )

    async def _request(self, method: str, path: str, json: dict[str, Any] | None = None) -> dict[str, Any]:
        is_read = method == "GET"
        retry_statuses = RETRY_ALWAYS | RETRY_READS_ONLY if is_read else RETRY_ALWAYS

        for attempt in range(1, MAX_ATTEMPTS + 1):
            is_last_attempt = attempt == MAX_ATTEMPTS
            try:
                response = await self._http.request(
                    method, self._base_url + path, headers=self._headers, json=json
                )
            except httpx2.TransportError as exc:
                # A request that never connected was not processed, so it is safe to
                # repeat. Any other network failure on a write is ambiguous.
                never_sent = isinstance(exc, (httpx2.ConnectError, httpx2.ConnectTimeout))
                if is_last_attempt or not (is_read or never_sent):
                    raise NotionError("Could not reach the Notion API.") from exc
                await self._sleep(_backoff(attempt))
                continue

            if response.status_code in retry_statuses and not is_last_attempt:
                await self._sleep(_retry_delay(response, attempt))
                continue
            if response.is_success:
                return response.json()
            raise _error_from(response)

        raise AssertionError("unreachable: the last attempt always returns or raises")


def rich_text(text: str) -> list[dict[str, Any]]:
    """Notion's rich-text format. Text above the 2,000-character limit is split."""
    return [
        {"type": "text", "text": {"content": text[start : start + TEXT_LIMIT]}}
        for start in range(0, len(text), TEXT_LIMIT)
    ]


def build_page(
    item: VerifiedActionItem, columns: NotionColumns, *, meeting_title: str, meeting_date: date | None
) -> dict[str, Any]:
    """Map one action item to the body of POST /v1/pages."""
    properties: dict[str, Any] = {
        columns.title: {"title": rich_text(item.title)},
        # Notion adds the option to the Select column if it does not exist yet.
        columns.priority: {"select": {"name": item.priority.value.capitalize()}},
    }
    if item.owner:
        properties[columns.owner] = {"rich_text": rich_text(item.owner)}
    if item.due_date:
        properties[columns.due] = {"date": {"start": item.due_date.isoformat()}}
    if columns.meeting:
        properties[columns.meeting] = {"rich_text": rich_text(meeting_title)}

    origin = f'From the meeting "{meeting_title}"'
    if meeting_date:
        origin += f" on {meeting_date.isoformat()}"
    origin += "."
    if not item.quote_verified:
        origin += " Needs review: this quote was not found word for word in the transcript."

    return {
        "parent": {"type": "data_source_id", "data_source_id": columns.data_source_id},
        "properties": properties,
        "children": [
            {"object": "block", "type": "paragraph", "paragraph": {"rich_text": rich_text(item.description)}},
            {"object": "block", "type": "quote", "quote": {"rich_text": rich_text(item.source_quote)}},
            {"object": "block", "type": "paragraph", "paragraph": {"rich_text": rich_text(origin)}},
        ],
    }


def _backoff(attempt: int) -> float:
    return float(min(2 ** (attempt - 1), 8))  # 1, 2, 4 seconds


def _retry_delay(response: httpx2.Response, attempt: int) -> float:
    """Wait as long as Notion asks (Retry-After, in seconds), within a ceiling."""
    retry_after = response.headers.get("Retry-After", "")
    if retry_after.isdigit():
        return float(min(int(retry_after), MAX_RETRY_DELAY_SECONDS))
    return _backoff(attempt)


def _error_from(response: httpx2.Response) -> NotionError:
    try:
        body = response.json()
    except ValueError:
        body = {}
    if not isinstance(body, dict):
        body = {}
    status = response.status_code

    if status == 401:
        message = "Notion rejected the token. Check NOTION_TOKEN."
    elif status == 403:
        message = "The Notion integration is not allowed to do this. Give it the 'Insert content' capability."
    elif status == 404:
        message = (
            "Notion could not find the database. Check NOTION_DATABASE_ID and connect the "
            "database to your integration (database menu > Connections)."
        )
    elif status == 400 and body.get("message"):
        message = f"Notion rejected the request: {body['message']}"
    elif status == 429:
        message = "Notion's rate limit was reached. Try again in a minute."
    else:
        code = f", {body['code']}" if body.get("code") else ""
        message = f"The Notion API returned an error (HTTP {status}{code})."
    return NotionError(message, status=status)
