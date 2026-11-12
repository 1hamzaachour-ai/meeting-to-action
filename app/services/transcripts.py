"""Turning an uploaded or downloaded file into clean transcript text."""

import codecs
import html
import re
from pathlib import PurePosixPath

from app.errors import UserFacingError
from app.schemas.jobs import TranscriptStats

ALLOWED_EXTENSIONS = frozenset({".txt", ".vtt"})

_VTT_VOICE_TAG = re.compile(r"<v(?:\.[^\s>]*)?\s+([^>]+)>")  # <v Speaker Name>
_VTT_ANY_TAG = re.compile(r"<[^>]+>")


class TranscriptError(UserFacingError):
    """The content cannot be used as a transcript."""


def safe_filename(filename: str | None) -> str:
    """Reduce a client-supplied name to its base name.

    The name is untrusted input and some clients send a full path. Keeping the last
    255 characters bounds its length without cutting off the extension.
    """
    name = PurePosixPath((filename or "").replace("\\", "/")).name
    return name[-255:]


def load_transcript(raw: bytes) -> str:
    """Bytes of a .txt or .vtt file -> transcript text ready for the LLM."""
    text = decode_transcript(raw)
    return vtt_to_text(text) if text.startswith("WEBVTT") else text


def decode_transcript(raw: bytes) -> str:
    """Decode bytes into normalized text, or raise TranscriptError.

    Accepts UTF-8 (with or without a byte-order mark) and BOM-marked UTF-16, which
    covers what Windows Notepad and Zoom produce. Any other encoding is rejected
    rather than guessed: a wrong guess would silently feed garbage to the LLM.
    """
    is_utf16 = raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE))
    try:
        text = raw.decode("utf-16" if is_utf16 else "utf-8-sig")
    except UnicodeDecodeError as exc:
        raise TranscriptError(
            "The file is not valid UTF-8 text. Save it as UTF-8 and upload it again."
        ) from exc
    if "\x00" in text:
        raise TranscriptError("The file looks like binary data, not a text transcript.")

    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise TranscriptError("The transcript is empty.")
    return text


def vtt_to_text(vtt: str) -> str:
    """Reduce a WebVTT caption file (Zoom's transcript format) to its spoken lines.

    A cue looks like this, and only its last part is kept:

        12
        00:01:22.000 --> 00:01:30.500
        Omar Haddad: Yes, but first I have to fix the billing bug.
    """
    spoken: list[str] = []
    for block in re.split(r"\n\s*\n", vtt):
        lines = block.strip().split("\n")
        timing_line = next((index for index, line in enumerate(lines) if "-->" in line), None)
        if timing_line is None:
            continue  # the WEBVTT header, a NOTE or a STYLE block
        cue = " ".join(line.strip() for line in lines[timing_line + 1 :])
        cue = _VTT_VOICE_TAG.sub(r"\1: ", cue)  # "<v Omar>Yes" -> "Omar: Yes"
        cue = html.unescape(_VTT_ANY_TAG.sub("", cue)).strip()
        if cue:
            spoken.append(cue)
    if not spoken:
        raise TranscriptError("The VTT file contains no captions.")
    return "\n".join(spoken)


def transcript_stats(text: str) -> TranscriptStats:
    return TranscriptStats(
        characters=len(text),
        words=len(text.split()),
        lines=text.count("\n") + 1,
    )
