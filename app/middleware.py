"""ASGI middleware."""

from starlette.datastructures import Headers
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

TOO_LARGE_DETAIL = "Request body too large."


class MaxBodySizeMiddleware:
    """Refuse request bodies above `max_bytes` instead of buffering them.

    Without it the server accepts an upload of any size and spools it to disk before
    the endpoint gets a chance to reject it. Written as plain ASGI so it can wrap
    `receive` and count the bytes as they arrive.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Fast path: the client announced the size, so refuse before reading anything.
        declared = Headers(scope=scope).get("content-length", "")
        if declared.isdigit() and int(declared) > self.max_bytes:
            response = JSONResponse({"detail": TOO_LARGE_DETAIL}, status_code=413)
            await response(scope, receive, send)
            return

        # Slow path: a chunked upload carries no Content-Length, so count while reading.
        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    # Raised while FastAPI reads the body; it becomes a normal 413 response.
                    raise HTTPException(status_code=413, detail=TOO_LARGE_DETAIL)
            return message

        await self.app(scope, limited_receive, send)
