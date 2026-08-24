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
    When no API key is configured (the test default via isolate_settings),
    the route falls through to the placeholder and returns verdict='unverifiable'.

    This tests the no-match / no-key code path, which remains the fallback
    behavior after Phase 2. The isolate_settings autouse fixture resets
    GOOGLE_FACTCHECK_API_KEY to empty, so no real API call is made.
    """
    response = await post_verify(async_client, {"text": SAMPLE_CLAIM})
    data = response.json()
    assert data["verdict"] == "unverifiable"


async def test_placeholder_confidence_is_zero(async_client: AsyncClient) -> None:
    """
    When no API key is configured (the test default via isolate_settings),
    the placeholder response has confidence_score=0.0.
    """
    response = await post_verify(async_client, {"text": SAMPLE_CLAIM})
    data = response.json()
    assert data["confidence_score"] == 0.0


async def test_placeholder_sources_is_empty(async_client: AsyncClient) -> None:
    """
    When no API key is configured (the test default via isolate_settings),
    the placeholder response has an empty sources list.
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


# =============================================================================
# Phase 2 — Route-level integration tests (Google Fact Check wired in)
# =============================================================================
#
# Isolation strategy:
#   - The `isolate_settings` autouse fixture leaves GOOGLE_FACTCHECK_API_KEY
#     as SecretStr("") by default.  Phase 1 tests pass because FactCheckConfigError
#     causes a fallthrough to the placeholder (200, unverifiable).
#   - Phase 2 tests that exercise the factcheck path must:
#       1. Set a non-empty fake key via monkeypatch.
#       2. Provide a respx_mock for the Google API call.
#   - Tests that specifically test the no-key fallthrough only need step 1
#     (explicitly setting an empty key to document the behavior).
#
# respx_mock intercepts ALL httpx requests for the test's duration, preventing
# any accidental real network calls to the Google Fact Check API.

import pytest as _pytest  # noqa: E402 — local import to avoid polluting Phase 1 namespace

from pydantic import SecretStr  # noqa: E402

from app.core import config as config_module  # noqa: E402
from app.services.factcheck import FACTCHECK_API_URL  # noqa: E402

_PHASE2_CLAIM = "The Earth is flat."

# A minimal valid Google API response used across multiple Phase 2 tests.
_GOOGLE_FALSE_RESPONSE = {
    "claims": [
        {
            "text": "The Earth is flat.",
            "claimReview": [
                {
                    "publisher": {
                        "name": "SciCheck",
                        "site": "factcheck.org",
                    },
                    "url": "https://www.factcheck.org/earth-is-not-flat",
                    "title": "Earth Is Not Flat",
                    "textualRating": "False",
                }
            ],
        }
    ]
}


async def test_route_returns_real_verdict_when_factcheck_found(
    async_client: AsyncClient,
    respx_mock,
    monkeypatch: _pytest.MonkeyPatch,
) -> None:
    """
    When Google returns a valid fact-check, the route returns the normalized verdict
    (not the placeholder).  The response still conforms to VerifyResponse schema.
    """
    monkeypatch.setattr(
        config_module.settings, "GOOGLE_FACTCHECK_API_KEY", SecretStr("test-key")
    )
    respx_mock.get(FACTCHECK_API_URL).respond(json=_GOOGLE_FALSE_RESPONSE)

    response = await async_client.post(VERIFY_URL, json={"text": _PHASE2_CLAIM})

    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "false"
    assert data["confidence_score"] == _pytest.approx(0.85)
    assert len(data["sources"]) == 1
    assert data["sources"][0]["publisher"] == "SciCheck"
    assert "factcheck.org" in data["sources"][0]["url"]


async def test_route_response_is_valid_verify_response_schema(
    async_client: AsyncClient,
    respx_mock,
    monkeypatch: _pytest.MonkeyPatch,
) -> None:
    """
    The normalized result passes through VerifyResponse Pydantic model validation
    before being returned.  Required fields are present and types are correct.
    """
    monkeypatch.setattr(
        config_module.settings, "GOOGLE_FACTCHECK_API_KEY", SecretStr("test-key")
    )
    respx_mock.get(FACTCHECK_API_URL).respond(json=_GOOGLE_FALSE_RESPONSE)

    response = await async_client.post(VERIFY_URL, json={"text": _PHASE2_CLAIM})

    assert response.status_code == 200
    data = response.json()
    # Verify all three required schema fields are present
    assert "verdict" in data
    assert "confidence_score" in data
    assert "sources" in data
    # Verify types
    assert data["verdict"] in ("true", "false", "misleading", "unverifiable")
    assert isinstance(data["confidence_score"], float)
    assert isinstance(data["sources"], list)


async def test_route_falls_through_to_placeholder_when_no_match(
    async_client: AsyncClient,
    respx_mock,
    monkeypatch: _pytest.MonkeyPatch,
) -> None:
    """
    When Google returns {} (no fact-check found), the route falls through to the
    Phase 1 placeholder: verdict='unverifiable', confidence=0.0, sources=[].
    """
    monkeypatch.setattr(
        config_module.settings, "GOOGLE_FACTCHECK_API_KEY", SecretStr("test-key")
    )
    respx_mock.get(FACTCHECK_API_URL).respond(json={})

    response = await async_client.post(VERIFY_URL, json={"text": _PHASE2_CLAIM})

    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "unverifiable"
    assert data["confidence_score"] == _pytest.approx(0.0)
    assert data["sources"] == []


async def test_route_falls_through_when_no_api_key(
    async_client: AsyncClient,
    monkeypatch: _pytest.MonkeyPatch,
) -> None:
    """
    When GOOGLE_FACTCHECK_API_KEY is empty, FactCheckConfigError is raised
    before any HTTP call.  The route falls through to the placeholder (200).
    No respx_mock needed: no HTTP request is made.
    """
    monkeypatch.setattr(
        config_module.settings, "GOOGLE_FACTCHECK_API_KEY", SecretStr("")
    )

    response = await async_client.post(VERIFY_URL, json={"text": _PHASE2_CLAIM})

    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "unverifiable"
    assert data["sources"] == []


async def test_route_502_on_google_auth_failure_401(
    async_client: AsyncClient,
    respx_mock,
    monkeypatch: _pytest.MonkeyPatch,
) -> None:
    """HTTP 401 from Google → route returns 502 Bad Gateway."""
    monkeypatch.setattr(
        config_module.settings, "GOOGLE_FACTCHECK_API_KEY", SecretStr("bad-key")
    )
    respx_mock.get(FACTCHECK_API_URL).respond(status_code=401)

    response = await async_client.post(VERIFY_URL, json={"text": _PHASE2_CLAIM})

    assert response.status_code == 502
    # Error detail must be a static safe string, not a raw upstream message
    assert response.json()["detail"] == "upstream service error"


async def test_route_503_on_google_quota_exceeded(
    async_client: AsyncClient,
    respx_mock,
    monkeypatch: _pytest.MonkeyPatch,
) -> None:
    """HTTP 429 from Google → route returns 503 Service Unavailable."""
    monkeypatch.setattr(
        config_module.settings, "GOOGLE_FACTCHECK_API_KEY", SecretStr("test-key")
    )
    respx_mock.get(FACTCHECK_API_URL).respond(status_code=429)

    response = await async_client.post(VERIFY_URL, json={"text": _PHASE2_CLAIM})

    assert response.status_code == 503
    assert response.json()["detail"] == "upstream service temporarily unavailable"


async def test_route_503_on_google_server_error(
    async_client: AsyncClient,
    respx_mock,
    monkeypatch: _pytest.MonkeyPatch,
) -> None:
    """HTTP 503 from Google → route returns 503 Service Unavailable."""
    monkeypatch.setattr(
        config_module.settings, "GOOGLE_FACTCHECK_API_KEY", SecretStr("test-key")
    )
    respx_mock.get(FACTCHECK_API_URL).respond(status_code=503)

    response = await async_client.post(VERIFY_URL, json={"text": _PHASE2_CLAIM})

    assert response.status_code == 503
    assert response.json()["detail"] == "upstream service temporarily unavailable"


async def test_route_503_on_google_timeout(
    async_client: AsyncClient,
    respx_mock,
    monkeypatch: _pytest.MonkeyPatch,
) -> None:
    """Google API timeout → route returns 503 Service Unavailable."""
    import httpx

    monkeypatch.setattr(
        config_module.settings, "GOOGLE_FACTCHECK_API_KEY", SecretStr("test-key")
    )
    respx_mock.get(FACTCHECK_API_URL).mock(
        side_effect=httpx.ConnectTimeout("timed out")
    )

    response = await async_client.post(VERIFY_URL, json={"text": _PHASE2_CLAIM})

    assert response.status_code == 503
    assert response.json()["detail"] == "upstream service temporarily unavailable"


# =============================================================================
# Phase 3 — Route-level integration tests (Classifier wired in)
# =============================================================================
#
# Isolation strategy:
#   - All tests mock `classify_claim` in the verify module via monkeypatch so
#     tests are hermetic and don't depend on Gemini or rule-based behavior.
#   - `verify_claim_factcheck` is also mocked where needed to avoid Gemini/
#     Google API calls and to assert call/no-call behavior.
#   - Phase 1 / Phase 2 tests continue to pass because `isolate_settings`
#     resets GEMINI_API_KEY to empty → classify_claim returns FACTUAL_CLAIM
#     via ConfigError fallthrough → factcheck also has no key → placeholder.
#
# Requirements covered:
#   13. Route does NOT call factcheck for OPINION
#   14. Route does NOT call factcheck for ADVERTISEMENT
#   15. FACTUAL_CLAIM reaches factcheck
#   16. AMBIGUOUS reaches factcheck
#   17. AMBIGUOUS confidence is multiplied by 0.7
#   18. Phase 1/2 route behavior still works (implicitly — full suite passes)

from unittest.mock import AsyncMock as _AsyncMock  # noqa: E402

import app.api.verify as _verify_module  # noqa: E402
from app.services.classifier import ClaimType as _ClaimType  # noqa: E402

_PHASE3_CLAIM = "Scientists claim that coffee prevents cancer."


async def test_route_opinion_returns_placeholder_without_calling_factcheck(
    async_client: AsyncClient,
    monkeypatch: _pytest.MonkeyPatch,
) -> None:
    """
    Requirement 13: OPINION claim → route returns placeholder immediately.
    verify_claim_factcheck must NOT be called.
    """
    monkeypatch.setattr(
        _verify_module, "classify_claim", _AsyncMock(return_value=_ClaimType.OPINION)
    )
    factcheck_mock = _AsyncMock(side_effect=AssertionError("factcheck must not be called for OPINION"))
    monkeypatch.setattr(_verify_module, "verify_claim_factcheck", factcheck_mock)

    response = await async_client.post(VERIFY_URL, json={"text": "I think coffee is healthy."})

    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "unverifiable"
    assert data["confidence_score"] == 0.0
    assert data["sources"] == []


async def test_route_advertisement_returns_placeholder_without_calling_factcheck(
    async_client: AsyncClient,
    monkeypatch: _pytest.MonkeyPatch,
) -> None:
    """
    Requirement 14: ADVERTISEMENT claim → route returns placeholder immediately.
    verify_claim_factcheck must NOT be called.
    """
    monkeypatch.setattr(
        _verify_module, "classify_claim", _AsyncMock(return_value=_ClaimType.ADVERTISEMENT)
    )
    factcheck_mock = _AsyncMock(side_effect=AssertionError("factcheck must not be called for ADVERTISEMENT"))
    monkeypatch.setattr(_verify_module, "verify_claim_factcheck", factcheck_mock)

    response = await async_client.post(VERIFY_URL, json={"text": "Buy now — limited time offer!"})

    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "unverifiable"
    assert data["confidence_score"] == 0.0
    assert data["sources"] == []


async def test_route_factual_claim_calls_factcheck(
    async_client: AsyncClient,
    respx_mock,
    monkeypatch: _pytest.MonkeyPatch,
) -> None:
    """
    Requirement 15: FACTUAL_CLAIM → fact-check is called and result is returned.
    """
    monkeypatch.setattr(
        _verify_module, "classify_claim", _AsyncMock(return_value=_ClaimType.FACTUAL_CLAIM)
    )
    monkeypatch.setattr(
        config_module.settings, "GOOGLE_FACTCHECK_API_KEY", SecretStr("test-key")
    )
    respx_mock.get(FACTCHECK_API_URL).respond(json=_GOOGLE_FALSE_RESPONSE)

    response = await async_client.post(VERIFY_URL, json={"text": _PHASE3_CLAIM})

    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "false"
    assert data["confidence_score"] == _pytest.approx(0.85)


async def test_route_ambiguous_calls_factcheck(
    async_client: AsyncClient,
    respx_mock,
    monkeypatch: _pytest.MonkeyPatch,
) -> None:
    """
    Requirement 16: AMBIGUOUS claim → fact-check is still called.
    """
    monkeypatch.setattr(
        _verify_module, "classify_claim", _AsyncMock(return_value=_ClaimType.AMBIGUOUS)
    )
    monkeypatch.setattr(
        config_module.settings, "GOOGLE_FACTCHECK_API_KEY", SecretStr("test-key")
    )
    respx_mock.get(FACTCHECK_API_URL).respond(json=_GOOGLE_FALSE_RESPONSE)

    response = await async_client.post(VERIFY_URL, json={"text": _PHASE3_CLAIM})

    assert response.status_code == 200
    # Confidence must be reduced (× 0.7), so it won't be 0.85
    data = response.json()
    assert data["verdict"] == "false"
    assert data["confidence_score"] != _pytest.approx(0.85), (
        "AMBIGUOUS confidence must be reduced from the raw factcheck value"
    )


async def test_route_ambiguous_confidence_reduced_by_factor(
    async_client: AsyncClient,
    respx_mock,
    monkeypatch: _pytest.MonkeyPatch,
) -> None:
    """
    Requirement 17: AMBIGUOUS claim confidence is multiplied by 0.7.
    Factcheck returns confidence=0.85 → final must be 0.85 × 0.7 = 0.595.
    """
    monkeypatch.setattr(
        _verify_module, "classify_claim", _AsyncMock(return_value=_ClaimType.AMBIGUOUS)
    )
    monkeypatch.setattr(
        config_module.settings, "GOOGLE_FACTCHECK_API_KEY", SecretStr("test-key")
    )
    respx_mock.get(FACTCHECK_API_URL).respond(json=_GOOGLE_FALSE_RESPONSE)

    response = await async_client.post(VERIFY_URL, json={"text": _PHASE3_CLAIM})

    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "false"
    expected_confidence = _pytest.approx(0.85 * 0.7, abs=1e-6)
    assert data["confidence_score"] == expected_confidence, (
        f"Expected confidence 0.85 × 0.7 = {0.85 * 0.7:.4f}, "
        f"got {data['confidence_score']}"
    )


async def test_route_ambiguous_no_factcheck_match_returns_placeholder(
    async_client: AsyncClient,
    respx_mock,
    monkeypatch: _pytest.MonkeyPatch,
) -> None:
    """
    AMBIGUOUS claim where factcheck returns no match → placeholder (confidence
    NOT reduced, since there is no match to reduce).
    """
    monkeypatch.setattr(
        _verify_module, "classify_claim", _AsyncMock(return_value=_ClaimType.AMBIGUOUS)
    )
    monkeypatch.setattr(
        config_module.settings, "GOOGLE_FACTCHECK_API_KEY", SecretStr("test-key")
    )
    respx_mock.get(FACTCHECK_API_URL).respond(json={})

    response = await async_client.post(VERIFY_URL, json={"text": _PHASE3_CLAIM})

    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "unverifiable"
    assert data["confidence_score"] == _pytest.approx(0.0)
    assert data["sources"] == []


async def test_route_classifier_failure_falls_through_to_factcheck(
    async_client: AsyncClient,
    respx_mock,
    monkeypatch: _pytest.MonkeyPatch,
) -> None:
    """
    If classify_claim raises unexpectedly (should not happen in practice, but
    defensively: the route wraps classify_claim in a try/except too), the
    pipeline continues.  This tests the resilience contract.

    Here we verify that when classifier returns FACTUAL_CLAIM (the fallback
    behavior for any classifier error), the factcheck layer still runs.
    """
    # Simulate the fallback scenario: classify_claim returns FACTUAL_CLAIM
    # (which is what it returns on any internal error).
    monkeypatch.setattr(
        _verify_module, "classify_claim", _AsyncMock(return_value=_ClaimType.FACTUAL_CLAIM)
    )
    monkeypatch.setattr(
        config_module.settings, "GOOGLE_FACTCHECK_API_KEY", SecretStr("test-key")
    )
    respx_mock.get(FACTCHECK_API_URL).respond(json=_GOOGLE_FALSE_RESPONSE)

    response = await async_client.post(VERIFY_URL, json={"text": _PHASE3_CLAIM})

    assert response.status_code == 200
    data = response.json()
    # Factcheck was called and returned a real verdict
    assert data["verdict"] == "false"
    assert data["confidence_score"] == _pytest.approx(0.85)


async def test_route_opinion_schema_is_valid_verify_response(
    async_client: AsyncClient,
    monkeypatch: _pytest.MonkeyPatch,
) -> None:
    """
    OPINION early-exit returns a response that conforms to VerifyResponse schema.
    All three required fields are present with correct types.
    """
    monkeypatch.setattr(
        _verify_module, "classify_claim", _AsyncMock(return_value=_ClaimType.OPINION)
    )
    monkeypatch.setattr(
        _verify_module, "verify_claim_factcheck",
        _AsyncMock(side_effect=AssertionError("factcheck must not be called"))
    )

    response = await async_client.post(VERIFY_URL, json={"text": "I believe coffee is healthy."})

    assert response.status_code == 200
    data = response.json()
    assert "verdict" in data
    assert "confidence_score" in data
    assert "sources" in data
    assert data["verdict"] in ("true", "false", "misleading", "unverifiable")
    assert isinstance(data["confidence_score"], float)
    assert isinstance(data["sources"], list)
    # ClaimType must NOT be present in the response
    assert "claim_type" not in data
    assert "type" not in data


# =============================================================================
# Phase 4 — Route-level integration tests (Tavily + LLM fallback)
# =============================================================================
#
# Isolation strategy:
#   - All Phase 4 tests use the FACTUAL_CLAIM classifier path (simplest path
#     to reach Phase 4).  classify_claim is mocked to return FACTUAL_CLAIM.
#   - The factcheck API mock returns {} (no match) so the route falls through
#     to Phase 4.  GOOGLE_FACTCHECK_API_KEY is set to a fake key.
#   - TAVILY_API_KEY and GEMINI_API_KEY are set as needed per test.
#   - All three HTTP endpoints (Google, Tavily, Gemini) are intercepted by
#     respx_mock so no real network calls occur.
#
# Test naming:
#   Phase 4 tests are numbered as requirements 19–30.

import json as _json  # noqa: E402

import app.api.verify as _verify_module_p4  # noqa: E402
from app.services.llm import GEMINI_VERIFY_URL as _GEMINI_VERIFY_URL  # noqa: E402
from app.services.search import TAVILY_SEARCH_URL as _TAVILY_SEARCH_URL  # noqa: E402

_PHASE4_CLAIM = "Coffee has been proven to cause cancer."
_FAKE_TAVILY_KEY = "tvly-phase4-test-key"
_FAKE_GEMINI_KEY = "AIzaSy-phase4-test-key"

# Tavily response with one valid result
_TAVILY_ONE_RESULT = {
    "results": [
        {
            "title": "Reuters: Coffee study",
            "url": "https://reuters.com/health/coffee-cancer",
            "content": "A major study found coffee does not cause cancer.",
            "score": 0.91,
        }
    ]
}

# Tavily response with two valid results
_TAVILY_TWO_RESULTS = {
    "results": [
        {
            "title": "Reuters: Coffee study",
            "url": "https://reuters.com/health/coffee-cancer",
            "content": "A major study found coffee does not cause cancer.",
            "score": 0.91,
        },
        {
            "title": "WHO: Cancer classification",
            "url": "https://who.int/cancer/coffee",
            "content": "WHO removed coffee from its carcinogen list.",
            "score": 0.85,
        },
    ]
}


def _llm_body(verdict: str = "false", confidence: float = 0.87, indices: list = None):
    """Build a minimal Gemini generateContent response body for Phase 4 tests."""
    payload = {
        "verdict": verdict,
        "confidence_score": confidence,
        "source_indices": indices if indices is not None else [0],
    }
    return {
        "candidates": [
            {"content": {"parts": [{"text": _json.dumps(payload)}]}}
        ]
    }


def _setup_factual_no_match(monkeypatch, respx_mock):
    """
    Helper: configure the test so that classify_claim returns FACTUAL_CLAIM,
    Google Fact Check finds no match, and both HTTP endpoints are available.
    The caller still needs to set TAVILY_API_KEY and mock the Tavily + Gemini
    HTTP calls.
    """
    monkeypatch.setattr(
        _verify_module_p4, "classify_claim",
        _AsyncMock(return_value=_ClaimType.FACTUAL_CLAIM),
    )
    monkeypatch.setattr(
        config_module.settings, "GOOGLE_FACTCHECK_API_KEY", SecretStr("test-key")
    )
    respx_mock.get(FACTCHECK_API_URL).respond(json={})  # no match


# ---------------------------------------------------------------------------
# Requirement 19: Factcheck match found → Phase 4 NOT invoked
# ---------------------------------------------------------------------------

async def test_phase4_factcheck_match_skips_search_and_llm(
    async_client: AsyncClient,
    respx_mock,
    monkeypatch: _pytest.MonkeyPatch,
) -> None:
    """
    Requirement 19: When Google Fact Check returns a match, Tavily and Gemini
    must NOT be called.  Any call to search or LLM would raise an AssertionError.
    """
    monkeypatch.setattr(
        _verify_module_p4, "classify_claim",
        _AsyncMock(return_value=_ClaimType.FACTUAL_CLAIM),
    )
    monkeypatch.setattr(
        config_module.settings, "GOOGLE_FACTCHECK_API_KEY", SecretStr("test-key")
    )
    respx_mock.get(FACTCHECK_API_URL).respond(json=_GOOGLE_FALSE_RESPONSE)

    search_mock = _AsyncMock(
        side_effect=AssertionError("search_evidence must not be called when factcheck matches")
    )
    llm_mock = _AsyncMock(
        side_effect=AssertionError("verify_with_llm must not be called when factcheck matches")
    )
    monkeypatch.setattr(_verify_module_p4, "search_evidence", search_mock)
    monkeypatch.setattr(_verify_module_p4, "verify_with_llm", llm_mock)

    response = await async_client.post(VERIFY_URL, json={"text": _PHASE4_CLAIM})
    assert response.status_code == 200
    assert response.json()["verdict"] == "false"


# ---------------------------------------------------------------------------
# Requirement 20: Factcheck None → Tavily + LLM invoked
# ---------------------------------------------------------------------------

async def test_phase4_no_factcheck_match_invokes_search_and_llm(
    async_client: AsyncClient,
    respx_mock,
    monkeypatch: _pytest.MonkeyPatch,
) -> None:
    """
    Requirement 20: When factcheck returns None, both search_evidence and
    verify_with_llm must be called.
    """
    monkeypatch.setattr(
        config_module.settings, "TAVILY_API_KEY", SecretStr(_FAKE_TAVILY_KEY)
    )
    monkeypatch.setattr(
        config_module.settings, "GEMINI_API_KEY", SecretStr(_FAKE_GEMINI_KEY)
    )
    _setup_factual_no_match(monkeypatch, respx_mock)
    respx_mock.post(_TAVILY_SEARCH_URL).respond(json=_TAVILY_ONE_RESULT)
    respx_mock.post(_GEMINI_VERIFY_URL).respond(json=_llm_body("false", 0.87, [0]))

    response = await async_client.post(VERIFY_URL, json={"text": _PHASE4_CLAIM})
    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "false"
    assert data["confidence_score"] == _pytest.approx(0.87)


# ---------------------------------------------------------------------------
# Requirement 21: Tavily returns evidence → VerifyResponse includes sources
# ---------------------------------------------------------------------------

async def test_phase4_sources_originate_from_tavily_not_llm(
    async_client: AsyncClient,
    respx_mock,
    monkeypatch: _pytest.MonkeyPatch,
) -> None:
    """
    Requirement 21: VerifyResponse.sources must be populated from the original
    Tavily SearchResult objects, not from LLM-supplied text.
    """
    monkeypatch.setattr(
        config_module.settings, "TAVILY_API_KEY", SecretStr(_FAKE_TAVILY_KEY)
    )
    monkeypatch.setattr(
        config_module.settings, "GEMINI_API_KEY", SecretStr(_FAKE_GEMINI_KEY)
    )
    _setup_factual_no_match(monkeypatch, respx_mock)
    respx_mock.post(_TAVILY_SEARCH_URL).respond(json=_TAVILY_ONE_RESULT)
    respx_mock.post(_GEMINI_VERIFY_URL).respond(json=_llm_body("false", 0.87, [0]))

    response = await async_client.post(VERIFY_URL, json={"text": _PHASE4_CLAIM})
    assert response.status_code == 200
    data = response.json()
    assert len(data["sources"]) == 1
    assert data["sources"][0]["title"] == "Reuters: Coffee study"
    assert "reuters.com" in data["sources"][0]["url"]
    assert data["sources"][0]["publisher"] == "reuters.com"


# ---------------------------------------------------------------------------
# Requirement 22: Empty Tavily results → unverifiable, Gemini NOT called
# ---------------------------------------------------------------------------

async def test_phase4_empty_tavily_results_returns_unverifiable_without_llm(
    async_client: AsyncClient,
    respx_mock,
    monkeypatch: _pytest.MonkeyPatch,
) -> None:
    """
    Requirement 22: Empty Tavily results → 200 unverifiable, Gemini NOT called.
    """
    monkeypatch.setattr(
        config_module.settings, "TAVILY_API_KEY", SecretStr(_FAKE_TAVILY_KEY)
    )
    _setup_factual_no_match(monkeypatch, respx_mock)
    respx_mock.post(_TAVILY_SEARCH_URL).respond(json={"results": []})

    llm_mock = _AsyncMock(
        side_effect=AssertionError("verify_with_llm must not be called with empty evidence")
    )
    monkeypatch.setattr(_verify_module_p4, "verify_with_llm", llm_mock)

    response = await async_client.post(VERIFY_URL, json={"text": _PHASE4_CLAIM})
    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "unverifiable"
    assert data["confidence_score"] == _pytest.approx(0.0)
    assert data["sources"] == []


# ---------------------------------------------------------------------------
# Requirement 23–26: Search failure → 200 unverifiable (not 5xx)
# ---------------------------------------------------------------------------

async def test_phase4_search_config_error_returns_unverifiable(
    async_client: AsyncClient,
    respx_mock,
    monkeypatch: _pytest.MonkeyPatch,
) -> None:
    """Requirement 23: Missing TAVILY_API_KEY → 200 unverifiable."""
    # TAVILY_API_KEY is empty (isolate_settings default)
    _setup_factual_no_match(monkeypatch, respx_mock)

    response = await async_client.post(VERIFY_URL, json={"text": _PHASE4_CLAIM})
    assert response.status_code == 200
    assert response.json()["verdict"] == "unverifiable"


async def test_phase4_search_quota_error_returns_unverifiable(
    async_client: AsyncClient,
    respx_mock,
    monkeypatch: _pytest.MonkeyPatch,
) -> None:
    """Requirement 24: Tavily 429 → 200 unverifiable, not 503."""
    monkeypatch.setattr(
        config_module.settings, "TAVILY_API_KEY", SecretStr(_FAKE_TAVILY_KEY)
    )
    _setup_factual_no_match(monkeypatch, respx_mock)
    respx_mock.post(_TAVILY_SEARCH_URL).respond(status_code=429)

    response = await async_client.post(VERIFY_URL, json={"text": _PHASE4_CLAIM})
    assert response.status_code == 200
    assert response.json()["verdict"] == "unverifiable"


async def test_phase4_search_timeout_returns_unverifiable(
    async_client: AsyncClient,
    respx_mock,
    monkeypatch: _pytest.MonkeyPatch,
) -> None:
    """Requirement 25: Tavily timeout → 200 unverifiable, not 503."""
    import httpx as _httpx

    monkeypatch.setattr(
        config_module.settings, "TAVILY_API_KEY", SecretStr(_FAKE_TAVILY_KEY)
    )
    _setup_factual_no_match(monkeypatch, respx_mock)
    respx_mock.post(_TAVILY_SEARCH_URL).mock(
        side_effect=_httpx.ConnectTimeout("timed out")
    )

    response = await async_client.post(VERIFY_URL, json={"text": _PHASE4_CLAIM})
    assert response.status_code == 200
    assert response.json()["verdict"] == "unverifiable"


async def test_phase4_search_service_error_returns_unverifiable(
    async_client: AsyncClient,
    respx_mock,
    monkeypatch: _pytest.MonkeyPatch,
) -> None:
    """Requirement 26: Tavily 500 → 200 unverifiable, not 503."""
    monkeypatch.setattr(
        config_module.settings, "TAVILY_API_KEY", SecretStr(_FAKE_TAVILY_KEY)
    )
    _setup_factual_no_match(monkeypatch, respx_mock)
    respx_mock.post(_TAVILY_SEARCH_URL).respond(status_code=500)

    response = await async_client.post(VERIFY_URL, json={"text": _PHASE4_CLAIM})
    assert response.status_code == 200
    assert response.json()["verdict"] == "unverifiable"


# ---------------------------------------------------------------------------
# Requirement 27–30: LLM failure → 200 unverifiable (not 5xx)
# ---------------------------------------------------------------------------

async def test_phase4_llm_config_error_returns_unverifiable(
    async_client: AsyncClient,
    respx_mock,
    monkeypatch: _pytest.MonkeyPatch,
) -> None:
    """Requirement 27: Missing GEMINI_API_KEY → 200 unverifiable."""
    monkeypatch.setattr(
        config_module.settings, "TAVILY_API_KEY", SecretStr(_FAKE_TAVILY_KEY)
    )
    # GEMINI_API_KEY is empty (isolate_settings default)
    _setup_factual_no_match(monkeypatch, respx_mock)
    respx_mock.post(_TAVILY_SEARCH_URL).respond(json=_TAVILY_ONE_RESULT)

    response = await async_client.post(VERIFY_URL, json={"text": _PHASE4_CLAIM})
    assert response.status_code == 200
    assert response.json()["verdict"] == "unverifiable"


async def test_phase4_llm_quota_error_returns_unverifiable(
    async_client: AsyncClient,
    respx_mock,
    monkeypatch: _pytest.MonkeyPatch,
) -> None:
    """Requirement 28: Gemini 429 → 200 unverifiable."""
    monkeypatch.setattr(
        config_module.settings, "TAVILY_API_KEY", SecretStr(_FAKE_TAVILY_KEY)
    )
    monkeypatch.setattr(
        config_module.settings, "GEMINI_API_KEY", SecretStr(_FAKE_GEMINI_KEY)
    )
    _setup_factual_no_match(monkeypatch, respx_mock)
    respx_mock.post(_TAVILY_SEARCH_URL).respond(json=_TAVILY_ONE_RESULT)
    respx_mock.post(_GEMINI_VERIFY_URL).respond(status_code=429)

    response = await async_client.post(VERIFY_URL, json={"text": _PHASE4_CLAIM})
    assert response.status_code == 200
    assert response.json()["verdict"] == "unverifiable"


async def test_phase4_llm_timeout_returns_unverifiable(
    async_client: AsyncClient,
    respx_mock,
    monkeypatch: _pytest.MonkeyPatch,
) -> None:
    """Requirement 29: Gemini timeout → 200 unverifiable."""
    import httpx as _httpx

    monkeypatch.setattr(
        config_module.settings, "TAVILY_API_KEY", SecretStr(_FAKE_TAVILY_KEY)
    )
    monkeypatch.setattr(
        config_module.settings, "GEMINI_API_KEY", SecretStr(_FAKE_GEMINI_KEY)
    )
    _setup_factual_no_match(monkeypatch, respx_mock)
    respx_mock.post(_TAVILY_SEARCH_URL).respond(json=_TAVILY_ONE_RESULT)
    respx_mock.post(_GEMINI_VERIFY_URL).mock(
        side_effect=_httpx.ConnectTimeout("timed out")
    )

    response = await async_client.post(VERIFY_URL, json={"text": _PHASE4_CLAIM})
    assert response.status_code == 200
    assert response.json()["verdict"] == "unverifiable"


async def test_phase4_llm_parse_error_returns_unverifiable(
    async_client: AsyncClient,
    respx_mock,
    monkeypatch: _pytest.MonkeyPatch,
) -> None:
    """Requirement 30: Gemini returns invalid JSON → 200 unverifiable."""
    monkeypatch.setattr(
        config_module.settings, "TAVILY_API_KEY", SecretStr(_FAKE_TAVILY_KEY)
    )
    monkeypatch.setattr(
        config_module.settings, "GEMINI_API_KEY", SecretStr(_FAKE_GEMINI_KEY)
    )
    _setup_factual_no_match(monkeypatch, respx_mock)
    respx_mock.post(_TAVILY_SEARCH_URL).respond(json=_TAVILY_ONE_RESULT)
    bad_body = {
        "candidates": [
            {"content": {"parts": [{"text": "I cannot determine this."}]}}
        ]
    }
    respx_mock.post(_GEMINI_VERIFY_URL).respond(json=bad_body)

    response = await async_client.post(VERIFY_URL, json={"text": _PHASE4_CLAIM})
    assert response.status_code == 200
    assert response.json()["verdict"] == "unverifiable"


# ---------------------------------------------------------------------------
# Requirement 31: Out-of-range LLM source index is silently discarded
# ---------------------------------------------------------------------------

async def test_phase4_out_of_range_source_index_discarded(
    async_client: AsyncClient,
    respx_mock,
    monkeypatch: _pytest.MonkeyPatch,
) -> None:
    """
    Requirement 31: LLM returns source_indices=[0, 99]; index 99 is out of range
    (only 1 evidence item exists) and must be silently discarded.
    The response must still return index 0 as a valid source.
    """
    monkeypatch.setattr(
        config_module.settings, "TAVILY_API_KEY", SecretStr(_FAKE_TAVILY_KEY)
    )
    monkeypatch.setattr(
        config_module.settings, "GEMINI_API_KEY", SecretStr(_FAKE_GEMINI_KEY)
    )
    _setup_factual_no_match(monkeypatch, respx_mock)
    respx_mock.post(_TAVILY_SEARCH_URL).respond(json=_TAVILY_ONE_RESULT)
    respx_mock.post(_GEMINI_VERIFY_URL).respond(
        json=_llm_body("false", 0.8, [0, 99])
    )

    response = await async_client.post(VERIFY_URL, json={"text": _PHASE4_CLAIM})
    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "false"
    # Only index 0 is valid; index 99 must be discarded
    assert len(data["sources"]) == 1
    assert "reuters.com" in data["sources"][0]["url"]


# ---------------------------------------------------------------------------
# Requirement 32: Duplicate LLM source indices are deduplicated
# ---------------------------------------------------------------------------

async def test_phase4_duplicate_source_indices_deduplicated(
    async_client: AsyncClient,
    respx_mock,
    monkeypatch: _pytest.MonkeyPatch,
) -> None:
    """
    Requirement 32: LLM returns source_indices=[0, 0, 1]; duplicate index 0
    must appear only once in VerifyResponse.sources.
    """
    monkeypatch.setattr(
        config_module.settings, "TAVILY_API_KEY", SecretStr(_FAKE_TAVILY_KEY)
    )
    monkeypatch.setattr(
        config_module.settings, "GEMINI_API_KEY", SecretStr(_FAKE_GEMINI_KEY)
    )
    _setup_factual_no_match(monkeypatch, respx_mock)
    respx_mock.post(_TAVILY_SEARCH_URL).respond(json=_TAVILY_TWO_RESULTS)
    respx_mock.post(_GEMINI_VERIFY_URL).respond(
        json=_llm_body("false", 0.8, [0, 0, 1])
    )

    response = await async_client.post(VERIFY_URL, json={"text": _PHASE4_CLAIM})
    assert response.status_code == 200
    data = response.json()
    # 3 indices but only 2 unique sources (0 appears twice)
    assert len(data["sources"]) == 2
    urls = [s["url"] for s in data["sources"]]
    assert len(set(urls)) == len(urls), "Duplicate URLs found in sources"


# ---------------------------------------------------------------------------
# Requirement 33: OPINION early exit — Phase 4 not called
# ---------------------------------------------------------------------------

async def test_phase4_opinion_still_exits_early(
    async_client: AsyncClient,
    monkeypatch: _pytest.MonkeyPatch,
) -> None:
    """
    Requirement 33: OPINION early-exit is unaffected by Phase 4.
    Neither Tavily nor Gemini must be called.
    """
    monkeypatch.setattr(
        _verify_module_p4, "classify_claim",
        _AsyncMock(return_value=_ClaimType.OPINION),
    )
    search_mock = _AsyncMock(
        side_effect=AssertionError("search_evidence must not be called for OPINION")
    )
    llm_mock = _AsyncMock(
        side_effect=AssertionError("verify_with_llm must not be called for OPINION")
    )
    monkeypatch.setattr(_verify_module_p4, "search_evidence", search_mock)
    monkeypatch.setattr(_verify_module_p4, "verify_with_llm", llm_mock)

    response = await async_client.post(VERIFY_URL, json={"text": "I love coffee!"})
    assert response.status_code == 200
    assert response.json()["verdict"] == "unverifiable"


# ---------------------------------------------------------------------------
# Requirement 34: Phase 4 VerifyResponse schema validation
# ---------------------------------------------------------------------------

async def test_phase4_response_is_valid_verify_response_schema(
    async_client: AsyncClient,
    respx_mock,
    monkeypatch: _pytest.MonkeyPatch,
) -> None:
    """
    Requirement 34: The Phase 4 LLM fallback response conforms to the
    VerifyResponse schema (verdict, confidence_score, sources with correct types).
    """
    monkeypatch.setattr(
        config_module.settings, "TAVILY_API_KEY", SecretStr(_FAKE_TAVILY_KEY)
    )
    monkeypatch.setattr(
        config_module.settings, "GEMINI_API_KEY", SecretStr(_FAKE_GEMINI_KEY)
    )
    _setup_factual_no_match(monkeypatch, respx_mock)
    respx_mock.post(_TAVILY_SEARCH_URL).respond(json=_TAVILY_ONE_RESULT)
    respx_mock.post(_GEMINI_VERIFY_URL).respond(json=_llm_body("false", 0.87, [0]))

    response = await async_client.post(VERIFY_URL, json={"text": _PHASE4_CLAIM})
    assert response.status_code == 200
    data = response.json()
    assert "verdict" in data
    assert "confidence_score" in data
    assert "sources" in data
    assert data["verdict"] in ("true", "false", "misleading", "unverifiable")
    assert isinstance(data["confidence_score"], float)
    assert isinstance(data["sources"], list)
    assert "claim_type" not in data
    assert "type" not in data
