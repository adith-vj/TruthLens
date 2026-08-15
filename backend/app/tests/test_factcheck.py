"""
tests/test_factcheck.py — Service interface tests for the factcheck module.

SCAFFOLDING PHASE:
    This file documents the test contracts that will be written in Phase 2
    when the Google Fact Check Tools API integration is implemented.

    No test functions are active here yet. The commented-out test skeletons
    serve as a specification for the Phase 2 implementor.

What to implement in Phase 2:
    - Replace all comments below with real async tests
    - Use respx (https://lundberg.github.io/respx/) to mock httpx calls
      so tests never make real network requests
    - Add respx to requirements.txt alongside the Phase 2 implementation
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Future test contracts — implement in Phase 2
# ---------------------------------------------------------------------------
#
# from unittest.mock import AsyncMock, patch
# import respx
# import httpx
#
# from app.services.factcheck import query_factcheck_api, FactCheckMatch
#
#
# @pytest.mark.anyio
# async def test_factcheck_returns_match_when_api_has_result():
#     """
#     When the API returns a relevant fact-check, query_factcheck_api()
#     returns a FactCheckMatch with a normalized verdict.
#     """
#     # Arrange: mock the Google Fact Check API response
#     # Act: call query_factcheck_api("The Earth is flat.")
#     # Assert: returns FactCheckMatch with verdict == "false"
#     ...
#
#
# @pytest.mark.anyio
# async def test_factcheck_returns_none_when_no_result():
#     """
#     When the API returns zero results, query_factcheck_api() returns None.
#     The route handler must fall through to the LLM layer.
#     """
#     # Arrange: mock the API to return {"claims": []}
#     # Act: call query_factcheck_api("some obscure claim")
#     # Assert: returns None
#     ...
#
#
# @pytest.mark.anyio
# async def test_factcheck_raises_on_api_timeout():
#     """
#     When the API times out, the service raises httpx.TimeoutException.
#     The route handler converts this to a 503 response.
#     """
#     ...
#
#
# @pytest.mark.anyio
# async def test_factcheck_raises_on_missing_api_key():
#     """
#     When GOOGLE_FACTCHECK_API_KEY is empty, the service raises ConfigurationError
#     before making any network request.
#     """
#     ...
#
#
# @pytest.mark.anyio
# async def test_verdict_normalization_maps_false_ratings():
#     """
#     Raw ratings like "False", "Pants on Fire", "Incorrect" should map
#     to normalized verdict "false".
#     """
#     ...
#
#
# @pytest.mark.anyio
# async def test_verdict_normalization_maps_true_ratings():
#     """
#     Raw ratings like "True", "Correct", "Verified" should map
#     to normalized verdict "true".
#     """
#     ...
#
#
# @pytest.mark.anyio
# async def test_verdict_normalization_maps_ambiguous_ratings():
#     """
#     Raw ratings like "Mostly True", "Half True", "Misleading" should map
#     to normalized verdict "misleading" with reduced confidence_score.
#     """
#     ...


# ---------------------------------------------------------------------------
# Placeholder: ensures pytest collects this file without errors
# ---------------------------------------------------------------------------

def test_factcheck_service_interface_placeholder() -> None:
    """
    Placeholder test confirming this file is collected by pytest.

    This test will be removed and replaced by real async tests in Phase 2
    when query_factcheck_api() is implemented.
    """
    # Import the interface module to verify it is importable with no side effects.
    from app.services import factcheck  # noqa: F401

    assert hasattr(factcheck, "FactCheckMatch"), (
        "FactCheckMatch model must be defined in services/factcheck.py"
    )
