"""The upload page."""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter(include_in_schema=False)

INDEX_HTML = Path(__file__).resolve().parents[2] / "static" / "index.html"


@router.get("/")
async def index() -> FileResponse:
    return FileResponse(INDEX_HTML)
