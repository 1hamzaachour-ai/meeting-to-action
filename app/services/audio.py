"""Receiving an audio upload safely: bounded size, on disk rather than in memory."""

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from fastapi import UploadFile

AUDIO_EXTENSIONS = frozenset({".mp3", ".m4a"})
CHUNK_BYTES = 1024 * 1024


class AudioTooLarge(Exception):
    pass


@dataclass(frozen=True)
class SavedAudio:
    path: Path
    size_bytes: int
    sha256: str
    head: bytes  # the first bytes, to check what kind of file it really is


async def save_audio_upload(file: UploadFile, extension: str, max_bytes: int) -> SavedAudio:
    """Copy the upload to a temporary file, at most `max_bytes` of it.

    A recording is tens of megabytes, so it is streamed to disk in pieces instead of
    being held in memory. The caller owns the file and must delete it.
    """
    digest = hashlib.sha256()
    size = 0
    head = b""
    descriptor, name = tempfile.mkstemp(prefix="meeting-audio-", suffix=extension)
    path = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as target:
            while chunk := await file.read(CHUNK_BYTES):
                size += len(chunk)
                if size > max_bytes:
                    raise AudioTooLarge
                if not head:
                    head = chunk[:16]
                digest.update(chunk)
                target.write(chunk)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return SavedAudio(path=path, size_bytes=size, sha256=digest.hexdigest(), head=head)


def looks_like_audio(extension: str, head: bytes) -> bool:
    """Check the file's first bytes against its extension: a renamed file is caught here."""
    if extension == ".mp3":
        # Either an ID3 tag, or straight into an MPEG audio frame (11 set bits).
        return head.startswith(b"ID3") or (len(head) >= 2 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0)
    if extension == ".m4a":
        return head[4:8] == b"ftyp"  # the MP4 container's opening box
    return False
