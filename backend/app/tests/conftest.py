"""
tests/conftest.py — Shared pytest fixtures for the TruthLens verification tests.

Provides:
    - async_client: An httpx AsyncClient configured for the FastAPI test app.
    - Environment variable isolation via monkeypatch so tests never require
      real API keys or read from an on-disk .env file.

Test isolation strategy:
    Settings are loaded once when app.core.config is imported. To override them
    per-test, we patch settings attributes directly on the already-loaded singleton
    using monkeypatch.setattr(). This avoids the complexity of re-importing
    modules mid-session while ensuring tests remain hermetic.
"""

from __future__ import annotations

from typing import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.core import config as config_module
from app.main import create_app


@pytest.fixture(autouse=True)
def isolate_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Ensure tests run against clean, predictable settings values.

    This fixture runs automatically for every test (autouse=True).
    It patches the module-level 'settings' singleton so tests are
    never affected by values in an on-disk .env file.
    """
    monkeypatch.setattr(config_module.settings, "APP_ENV", "test")
    monkeypatch.setattr(config_module.settings, "LOG_LEVEL", "WARNING")
    monkeypatch.setattr(config_module.settings, "MAX_CLAIM_LENGTH", 2000)
    monkeypatch.setattr(config_module.settings, "ALLOWED_ORIGINS", ["*"])
    monkeypatch.setattr(config_module.settings, "FACTCHECK_TIMEOUT_SECONDS", 5.0)
    monkeypatch.setattr(config_module.settings, "FACTCHECK_MAX_RESULTS", 5)
    # NOTE: GOOGLE_FACTCHECK_API_KEY is intentionally NOT set here.
    # It defaults to SecretStr("") from Settings, which causes
    # FactCheckConfigError → route falls through to placeholder.
    # This keeps all Phase 1 test assertions valid without any mocking.
    # Tests that exercise real factcheck behavior must set the key explicitly
    # via monkeypatch and provide a respx_mock for the HTTP call.


@pytest_asyncio.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """
    Yield an httpx AsyncClient that drives the FastAPI app via ASGI transport.

    No real network connections are made. The test app is created fresh for
    each test function that requests this fixture, ensuring full isolation.
    """
    test_app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=test_app),
        base_url="http://testserver",
    ) as client:
        yield client
