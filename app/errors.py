"""Errors whose message is safe and useful to show to the person using the app."""


class UserFacingError(Exception):
    """An expected failure: a rejected key, a refused download, an unusable file.

    `str(error)` says what went wrong and how to fix it, without internal details,
    so the pipeline stores it on the job as-is. Any other exception is treated as a
    bug: it is logged in full and reported to the user only by its type.
    """
