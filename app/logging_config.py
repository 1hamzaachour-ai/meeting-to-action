"""Console logging setup."""

import logging.config
import sys


def configure_logging(level: str = "INFO") -> None:
    # A Windows console (cp1252) cannot encode every character, for example a
    # non-Latin file name, and the log line would be lost. Escape such characters
    # instead of failing.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,  # keep uvicorn's own loggers working
            "formatters": {
                "default": {"format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"},
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {"level": level, "handlers": ["console"]},
            # The HTTP client logs every outbound request at INFO, full URL included.
            # The app logs what matters about those calls itself.
            "loggers": {"httpx2": {"level": "WARNING"}},
        }
    )
