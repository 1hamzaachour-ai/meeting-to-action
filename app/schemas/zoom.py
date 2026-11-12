"""The parts of a Zoom webhook request this app reads. Unknown fields are ignored."""

from datetime import datetime

from pydantic import BaseModel


class ZoomRecordingFile(BaseModel):
    file_type: str | None = None  # "TRANSCRIPT" for the audio transcript
    file_extension: str | None = None  # "VTT"
    recording_type: str | None = None  # "audio_transcript"
    download_url: str | None = None


class ZoomRecording(BaseModel):
    uuid: str | None = None  # unique per meeting instance
    id: int | str | None = None  # the meeting number
    topic: str | None = None
    start_time: datetime | None = None  # UTC
    timezone: str | None = None  # the host's time zone, e.g. "Africa/Tunis"
    recording_files: list[ZoomRecordingFile] = []


class ZoomEventPayload(BaseModel):
    plainToken: str | None = None  # only on endpoint.url_validation
    object: ZoomRecording | None = None


class ZoomEvent(BaseModel):
    event: str
    download_token: str | None = None  # lets the receiver download the files for 24 hours
    payload: ZoomEventPayload = ZoomEventPayload()
