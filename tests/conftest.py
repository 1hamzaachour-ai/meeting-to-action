"""Shared test fixtures. The fakes they use live in helpers.py."""

from contextlib import ExitStack

import pytest
from fastapi.testclient import TestClient
from helpers import FakeExtractor, FakeTranscriber, make_settings

from app.main import create_app


@pytest.fixture
def extractor() -> FakeExtractor:
    return FakeExtractor()


@pytest.fixture
def transcriber() -> FakeTranscriber:
    return FakeTranscriber()


@pytest.fixture
def make_client(extractor, transcriber):
    """Build a test client for an app with the given settings and fake services."""
    with ExitStack() as stack:

        def build(*, transport=None, with_llm: bool = True, with_stt: bool = False, **settings_overrides) -> TestClient:
            app = create_app(
                make_settings(**settings_overrides),
                extractor=extractor if with_llm else None,
                transcriber=transcriber if with_stt else None,
                http_transport=transport,
            )
            return stack.enter_context(TestClient(app))  # entering runs startup; leaving runs shutdown

        yield build


@pytest.fixture
def client(make_client) -> TestClient:
    """An app with a working (fake) LLM, and no speech-to-text, Notion, Zoom or API key."""
    return make_client()
