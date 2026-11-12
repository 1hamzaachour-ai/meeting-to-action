"""Speech-to-text with a Hugging Face Whisper model running on this machine.

Why it exists: stock Whisper does badly on Tunisian Arabic, while models fine-tuned
on Tunisian speech do far better, and those are only available as downloadable
weights. This runs one of them, on the GPU when there is one.

It needs packages that the rest of the app does not (see requirements-local-stt.txt).
They are imported inside the functions below, so the app starts normally without them.
"""

import asyncio
import importlib.util
import logging
from pathlib import Path
from typing import Any

from app.schemas.transcription import LANGUAGE_CODES, SpeechHints, Transcription
from app.services.transcription import Transcriber, TranscriptionError

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000  # what Whisper models expect
REQUIRED_PACKAGES = ("torch", "transformers", "av")


def missing_packages() -> list[str]:
    """Which of the extra packages are not installed (checked without importing them)."""
    return [name for name in REQUIRED_PACKAGES if importlib.util.find_spec(name) is None]


class LocalWhisperTranscriber(Transcriber):
    def __init__(self, model_id: str) -> None:
        self._model_id = model_id
        # Loaded on first use: it takes a while, and gigabytes of memory.
        self._recognizer: Any = None
        # One recording at a time: two would compete for the same GPU memory.
        self._lock = asyncio.Lock()

    async def transcribe(self, audio: Path, filename: str, hints: SpeechHints) -> Transcription:
        missing = missing_packages()
        if missing:
            raise TranscriptionError(
                f"The local speech model needs extra packages ({', '.join(missing)}): "
                "pip install -r requirements-local-stt.txt"
            )
        async with self._lock:
            # The model call blocks for a long time, so it runs in a worker thread
            # and the server keeps answering other requests meanwhile.
            return await asyncio.to_thread(self._transcribe_blocking, audio, hints)

    def _transcribe_blocking(self, audio: Path, hints: SpeechHints) -> Transcription:
        if self._recognizer is None:
            self._recognizer = load_recognizer(self._model_id)
        samples = decode_audio(audio)

        generate_kwargs: dict[str, Any] = {"task": "transcribe"}
        codes = LANGUAGE_CODES[hints.language]
        if codes:
            generate_kwargs["language"] = codes[0]

        # Whisper reads 30 seconds at a time; the pipeline cuts longer audio into
        # overlapping windows and stitches the pieces back together. Whisper's own
        # long-form mode is not used because it steers by timestamp tokens, which
        # fine-tuned models have often stopped producing; hence ignore_warning.
        on_gpu = getattr(getattr(self._recognizer, "device", None), "type", "cpu") == "cuda"
        result = self._recognizer(
            {"raw": samples, "sampling_rate": SAMPLE_RATE},
            chunk_length_s=30,
            batch_size=4 if on_gpu else 1,
            ignore_warning=True,
            generate_kwargs=generate_kwargs,
        )
        text = str(result["text"]).strip()
        if not text:
            raise TranscriptionError("No speech was recognised in the recording.")
        logger.info("Transcribed %.0f seconds of audio with %s", len(samples) / SAMPLE_RATE, self._model_id)
        return Transcription(engine=self._model_id, languages=list(codes[:1]), text=text)


def load_recognizer(model_id: str) -> Any:
    """Download the model if needed (several gigabytes the first time) and load it."""
    import torch
    from transformers import pipeline

    on_gpu = torch.cuda.is_available()
    logger.info("Loading speech model %s on %s", model_id, "GPU" if on_gpu else "CPU (slow)")
    try:
        return pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=0 if on_gpu else -1,
            # Half precision halves the GPU memory needed; CPUs want full precision.
            dtype=torch.float16 if on_gpu else torch.float32,
        )
    except OSError as exc:  # unknown model id, no network, gated repository
        raise TranscriptionError(f"Could not load the speech model {model_id!r}: {exc}") from exc


def decode_audio(path: Path) -> Any:
    """Read any audio file into the form Whisper wants: mono, 16 kHz, floats in [-1, 1].

    PyAV carries its own copy of FFmpeg, so no separate FFmpeg install is needed.
    """
    import av
    import numpy as np

    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    pieces = []
    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise TranscriptionError("The file contains no audio track.")
            for frame in container.decode(container.streams.audio[0]):
                pieces.extend(resampled.to_ndarray().reshape(-1) for resampled in resampler.resample(frame))
            pieces.extend(resampled.to_ndarray().reshape(-1) for resampled in resampler.resample(None))  # flush
    except av.FFmpegError as exc:
        raise TranscriptionError("The recording could not be decoded. Is it a valid audio file?") from exc

    if not pieces:
        raise TranscriptionError("The recording is empty.")
    return np.concatenate(pieces).astype(np.float32) / 32768.0
