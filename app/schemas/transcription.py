"""Speech-to-text models: what is asked of an engine, and what it returns."""

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel


class SpokenLanguage(StrEnum):
    """What is spoken in a recording, chosen by the person uploading it."""

    DERJA = "derja"  # Tunisian Arabic mixed with French and English
    ARABIC = "ar"
    FRENCH = "fr"
    ENGLISH = "en"
    AUTO = "auto"  # let the engine detect it


# The ISO 639-1 codes sent to a speech engine for each choice. Derja has no code
# of its own, so it is described as Arabic with French and English mixed in.
LANGUAGE_CODES: dict[SpokenLanguage, tuple[str, ...]] = {
    SpokenLanguage.DERJA: ("ar", "fr", "en"),
    SpokenLanguage.ARABIC: ("ar",),
    SpokenLanguage.FRENCH: ("fr",),
    SpokenLanguage.ENGLISH: ("en",),
    SpokenLanguage.AUTO: (),
}


@dataclass(frozen=True)
class SpeechHints:
    """What the engine is told about a recording before it listens."""

    language: SpokenLanguage = SpokenLanguage.DERJA
    # Names and terms worth spelling right: participants, products, clients.
    keywords: tuple[str, ...] = ()


class Transcription(BaseModel):
    """The text an engine produced from a recording. Kept on the job so it can be reviewed."""

    engine: str  # the model that produced it
    languages: list[str] = []  # languages detected, when the engine reports them
    text: str
