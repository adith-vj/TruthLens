"""
tests/test_verify.py — Route-level tests for POST /api/verify.

Tests cover:
    1. Valid claim          → 200 with correct placeholder response
    2. Missing text field   → 422
    3. Empty text           → 422
    4. Whitespace-only text → 422
    5. Text too long        → 422
    6. Required fields      → all three fields present in response
    7. Verdict enum value   → one of the four allowed values
    8. Confidence range     → 0.0 <= confidence_score <= 1.0
    9. Sources type         → sources is a JSON array
    10. Health endpoint     → 200 with {"status": "ok"}

During the scaffolding phase, all 200-level responses return the placeholder:
    { "verdict": "unverifiable", "confidence_score": 0.0, "sources": [] }

These assertions will be updated in Phase 2 once real verification is wired.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

VERIFY_URL = "/api/verify"
HEALTH_URL = "/health"

# The four valid verdict strings defined in the VerifyResponse schema.
VALID_VERDICTS = {"true", "false", "misleading", "unverifiable"}

# A well-formed, normal-length factual claim used as the standard valid input.
SAMPLE_CLAIM = "The Earth has one moon."


# ---------------------------------------------------------------------------
# Parametrize markers
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def post_verify(client: AsyncClient, payload: dict) -> "httpx.Response":  # type: ignore[name-defined]
    return await client.post(VERIFY_URL, json=payload)


# ---------------------------------------------------------------------------
# Test: valid request
# ---------------------------------------------------------------------------

async def test_valid_claim(async_client: AsyncClient) -> None:
    """A well-formed claim returns 200 with a valid VerifyResponse."""
    response = await post_verify(async_client, {"text": SAMPLE_CLAIM})
    assert response.status_code == 200, response.text


async def test_placeholder_verdict_is_unverifiable(async_client: AsyncClient) -> None:
    """
    During scaffolding, the verdict is always 'unverifiable'.
    This assertion documents the scaffold-phase behavior and will be updated
    in Phase 2 when real verification is implemented.
    """
    response = await post_verify(async_client, {"text": SAMPLE_CLAIM})
    data = response.json()
    assert data["verdict"] == "unverifiable", (
        "Scaffold should return 'unverifiable'. "
        "Update this test in Phase 2 when real verification is wired."
    )


async def test_placeholder_confidence_is_zero(async_client: AsyncClient) -> None:
    """
    During scaffolding, confidence_score is always 0.0.
    Will be updated in Phase 2.
    """
    response = await post_verify(async_client, {"text": SAMPLE_CLAIM})
    data = response.json()
    assert data["confidence_score"] == 0.0


async def test_placeholder_sources_is_empty(async_client: AsyncClient) -> None:
    """
    During scaffolding, sources is always an empty list.
    Will be updated in Phase 2.
    """
    response = await post_verify(async_client, {"text": SAMPLE_CLAIM})
    data = response.json()
    assert data["sources"] == []


# ---------------------------------------------------------------------------
# Test: input validation — 422 cases
# ---------------------------------------------------------------------------

async def test_missing_text_field(async_client: AsyncClient) -> None:
    """An empty JSON body (missing 'text') returns 422."""
    response = await post_verify(async_client, {})
    assert response.status_code == 422


async def test_empty_text(async_client: AsyncClient) -> None:
    """An empty string for 'text' returns 422."""
    response = await post_verify(async_client, {"text": ""})
    assert response.status_code == 422


async def test_whitespace_only_text(async_client: AsyncClient) -> None:
    """A whitespace-only string for 'text' returns 422."""
    response = await post_verify(async_client, {"text": "   "})
    assert response.status_code == 422


async def test_tab_and_newline_whitespace(async_client: AsyncClient) -> None:
    """Tabs and newlines without visible characters count as whitespace-only."""
    response = await post_verify(async_client, {"text": "\t\n\r"})
    assert response.status_code == 422


async def test_text_too_long(async_client: AsyncClient) -> None:
    """A claim exceeding MAX_CLAIM_LENGTH (2000 chars) returns 422."""
    long_text = "a" * 2001
    response = await post_verify(async_client, {"text": long_text})
    assert response.status_code == 422


async def test_text_at_exact_max_length(async_client: AsyncClient) -> None:
    """A claim of exactly MAX_CLAIM_LENGTH characters is accepted."""
    exact_text = "a" * 2000
    response = await post_verify(async_client, {"text": exact_text})
    assert response.status_code == 200


async def test_non_string_text_field(async_client: AsyncClient) -> None:
    """A non-string value for 'text' returns 422."""
    response = await post_verify(async_client, {"text": 12345})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Test: response schema validation
# ---------------------------------------------------------------------------

async def test_response_has_required_fields(async_client: AsyncClient) -> None:
    """Response body contains exactly the three required top-level fields."""
    response = await post_verify(async_client, {"text": SAMPLE_CLAIM})
    data = response.json()
    assert "verdict" in data, "Response must contain 'verdict'"
    assert "confidence_score" in data, "Response must contain 'confidence_score'"
    assert "sources" in data, "Response must contain 'sources'"


async def test_verdict_is_valid_enum_value(async_client: AsyncClient) -> None:
    """The 'verdict' field is one of the four allowed values."""
    response = await post_verify(async_client, {"text": SAMPLE_CLAIM})
    data = response.json()
    assert data["verdict"] in VALID_VERDICTS, (
        f"verdict '{data['verdict']}' is not a valid value. "
        f"Must be one of: {VALID_VERDICTS}"
    )


async def test_confidence_score_in_range(async_client: AsyncClient) -> None:
    """The 'confidence_score' is a float in [0.0, 1.0]."""
    response = await post_verify(async_client, {"text": SAMPLE_CLAIM})
    data = response.json()
    score = data["confidence_score"]
    assert isinstance(score, (int, float)), "confidence_score must be numeric"
    assert 0.0 <= score <= 1.0, f"confidence_score {score} is out of [0.0, 1.0] range"


async def test_sources_is_list(async_client: AsyncClient) -> None:
    """The 'sources' field is a JSON array."""
    response = await post_verify(async_client, {"text": SAMPLE_CLAIM})
    data = response.json()
    assert isinstance(data["sources"], list), "sources must be a JSON array"


async def test_response_content_type_is_json(async_client: AsyncClient) -> None:
    """Response Content-Type is application/json."""
    response = await post_verify(async_client, {"text": SAMPLE_CLAIM})
    assert "application/json" in response.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# Test: health endpoint
# ---------------------------------------------------------------------------

async def test_health_endpoint(async_client: AsyncClient) -> None:
    """GET /health returns 200."""
    response = await async_client.get(HEALTH_URL)
    assert response.status_code == 200


async def test_health_response_has_status(async_client: AsyncClient) -> None:
    """GET /health returns a body with 'status': 'ok'."""
    response = await async_client.get(HEALTH_URL)
    data = response.json()
    assert data.get("status") == "ok"


async def test_health_response_has_env(async_client: AsyncClient) -> None:
    """GET /health returns a body with an 'env' field."""
    response = await async_client.get(HEALTH_URL)
    data = response.json()
    assert "env" in data, "Health response must include 'env' field"
