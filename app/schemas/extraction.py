"""What the LLM extracts from a transcript.

`MeetingExtraction` is sent to the model as a JSON schema (Structured Outputs), so
the field descriptions below are part of the prompt: the model reads them.
Every field is required and nothing has a default, which keeps the schema strict;
"unknown" is expressed with null rather than by leaving a key out.
"""

from dataclasses import dataclass
from datetime import date
from enum import StrEnum

from pydantic import BaseModel, Field

from app.schemas.transcription import SpokenLanguage


class Priority(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ActionItem(BaseModel):
    title: str = Field(
        description="The task as a short instruction that starts with a verb, at most about 12 words."
    )
    description: str = Field(
        description="One or two sentences of context: what exactly has to be done, and why."
    )
    owner: str | None = Field(
        description="Name of the person who took the task or was given it, as written in the "
        "transcript. null when nobody was named."
    )
    due_date: date | None = Field(
        description="The deadline as a calendar date (YYYY-MM-DD). null when no deadline was "
        "stated or it cannot be turned into a date."
    )
    priority: Priority = Field(
        description="high when it was called urgent, blocking or top priority; low when it was "
        "called non-urgent; medium otherwise."
    )
    source_quote: str = Field(
        description="One or two sentences copied word for word from the transcript, in its "
        "original language and script, that show this task was agreed."
    )


class MeetingExtraction(BaseModel):
    meeting_title: str = Field(
        description="The meeting title if one was provided or stated; otherwise a short "
        "descriptive title."
    )
    summary: str = Field(
        description="Two to four sentences on what was discussed and decided, including "
        "decisions that are not tasks."
    )
    action_items: list[ActionItem] = Field(
        description="Every action item, in the order it came up. Empty when there are none."
    )


class VerifiedActionItem(ActionItem):
    """An action item after the app checked its quote against the transcript."""

    quote_verified: bool


@dataclass(frozen=True)
class MeetingContext:
    """What is known about the meeting besides the transcript itself."""

    title: str | None = None
    meeting_date: date | None = None
    participants: tuple[str, ...] = ()
    # Set when the transcript came out of speech recognition rather than a text file.
    spoken_language: SpokenLanguage | None = None
