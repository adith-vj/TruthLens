"""
tests/test_factcheck.py — Full test suite for the Google Fact Check service.

Test structure
--------------
    Section A: normalize_rating() — sync unit tests for the rating mapping table.
               These test pure functions with zero I/O overhead.
    Section B: normalize_response() — sync unit tests for the raw-dict → FactCheckMatch
               conversion layer.
    Section C: query_factcheck_api() — async integration tests using respx_mock
               to intercept httpx requests without any real network calls.
    Section D: verify_claim_factcheck() — async end-to-end service tests.

Phase 1 compatibility
---------------------
This file fully replaces the Phase 1 stub (test_factcheck_service_interface_placeholder).
The service is now importable and callable; no placeholder is needed.

respx_mock usage
----------------
The `respx_mock` fixture is provided by the respx pytest plugin (installed via
requirements.txt).  It intercepts ALL httpx requests for the duration of each
test, preventing real network calls.  Any unregistered request raises an error,
ensuring tests cannot accidentally call the live Google API.

API key isolation
-----------------
The `isolate_settings` autouse fixture (conftest.py) does NOT set
GOOGLE_FACTCHECK_API_KEY, leaving it as SecretStr("").  Tests that need to
exercise the HTTP layer must set a fake key via monkeypatch before calling
query_factcheck_api / verify_claim_factcheck.
"""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from app.core import config as config_module
from app.services.factcheck import (
    FACTCHECK_API_URL,
    FactCheckAuthError,
    FactCheckConfigError,
    FactCheckMatch,
    FactCheckQuotaError,
    FactCheckServiceError,
    FactCheckTimeoutError,
    normalize_rating,
    normalize_response,
    query_factcheck_api,
    verify_claim_factcheck,
)

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

# A minimal, valid Google Fact Check API response representing one claim
# with a single claimReview rated "False".
_SAMPLE_FALSE_RESPONSE: dict = {
    "claims": [
        {
            "text": "The Earth is flat.",
            "claimReview": [
                {
                    "publisher": {
                        "name": "Example Fact Checker",
                        "site": "example-fact-checker.com",
                    },
                    "url": "https://example-fact-checker.com/earth-is-not-flat",
                    "title": "No, the Earth is not flat",
                    "textualRating": "False",
                }
            ],
        }
    ]
}

# Claim used in HTTP-layer tests — value is arbitrary; Google is mocked.
_CLAIM = "The Earth is flat."

# ---------------------------------------------------------------------------
# A. normalize_rating() — pure function tests (sync)
# ---------------------------------------------------------------------------
# These test the explicit _RATING_MAP table without any fixtures.
# Each test documents an explicit, approved mapping.


def test_normalize_rating_true() -> None:
    """PolitiFact 'True' maps to verdict='true' with high confidence."""
    verdict, score = normalize_rating("True")
    assert verdict == "true"
    assert score == pytest.approx(0.85)


def test_normalize_rating_mostly_true_maps_to_misleading() -> None:
    """
    AD-16: 'Mostly True' → 'misleading' (not 'true').

    This is the explicitly reviewed design decision: 'Mostly True' implies
    measurable inaccuracy and must not cause TruthLens to endorse the claim
    as accurate.  See AD-16 in services/factcheck.py for full rationale.
    """
    verdict, score = normalize_rating("Mostly True")
    assert verdict == "misleading", (
        "AD-16: 'Mostly True' must map to 'misleading', not 'true'. "
        "See services/factcheck.py _RATING_MAP docstring for rationale."
    )
    assert score == pytest.approx(0.70)


def test_normalize_rating_half_true() -> None:
    """PolitiFact 'Half True' maps to 'misleading'."""
    verdict, score = normalize_rating("Half True")
    assert verdict == "misleading"
    assert score == pytest.approx(0.65)


def test_normalize_rating_mostly_false() -> None:
    """PolitiFact 'Mostly False' maps to 'misleading' (more false than true, but mixed)."""
    verdict, score = normalize_rating("Mostly False")
    assert verdict == "misleading"
    assert score == pytest.approx(0.70)


def test_normalize_rating_false() -> None:
    """PolitiFact 'False' maps to verdict='false' with high confidence."""
    verdict, score = normalize_rating("False")
    assert verdict == "false"
    assert score == pytest.approx(0.85)


def test_normalize_rating_pants_on_fire() -> None:
    """PolitiFact 'Pants on Fire' maps to verdict='false' with highest confidence."""
    verdict, score = normalize_rating("Pants on Fire")
    assert verdict == "false"
    assert score == pytest.approx(0.90)


def test_normalize_rating_unknown_maps_to_fallback() -> None:
    """Any rating not in the explicit table falls back to ('unverifiable', 0.30)."""
    verdict, score = normalize_rating("Something Completely Unknown")
    assert verdict == "unverifiable"
    assert score == pytest.approx(0.30)


def test_normalize_rating_empty_string_fallback() -> None:
    """An empty rating string (e.g., textualRating missing in response) falls back."""
    verdict, score = normalize_rating("")
    assert verdict == "unverifiable"
    assert score == pytest.approx(0.30)


def test_normalize_rating_case_insensitive() -> None:
    """Normalization is case-insensitive: 'FALSE', 'False', 'false' all map the same."""
    for raw in ("FALSE", "False", "false", "fAlSe"):
        verdict, _ = normalize_rating(raw)
        assert verdict == "false", f"Failed for input: {raw!r}"


def test_normalize_rating_strips_punctuation() -> None:
    """Trailing/leading punctuation is stripped before lookup."""
    # "Pants on Fire!" should normalize to "pants on fire" → ("false", 0.90)
    verdict, score = normalize_rating("Pants on Fire!")
    assert verdict == "false"
    assert score == pytest.approx(0.90)


def test_normalize_rating_hyphenated_form() -> None:
    """Hyphenated forms like 'Half-True' are treated identically to 'Half True'."""
    verdict_hyphen, score_hyphen = normalize_rating("Half-True")
    verdict_space, score_space = normalize_rating("Half True")
    assert verdict_hyphen == verdict_space
    assert score_hyphen == pytest.approx(score_space)


def test_normalize_rating_four_pinocchios() -> None:
    """Washington Post 'Four Pinocchios' maps to 'false'."""
    verdict, score = normalize_rating("Four Pinocchios")
    assert verdict == "false"
    assert score == pytest.approx(0.90)


def test_normalize_rating_satire_is_unverifiable() -> None:
    """Snopes 'Satire' maps to 'unverifiable' — satire is not a factual claim."""
    verdict, _ = normalize_rating("Satire")
    assert verdict == "unverifiable"


# ---------------------------------------------------------------------------
# B. normalize_response() — pure function tests (sync)
# ---------------------------------------------------------------------------


def test_normalize_response_empty_object_returns_none() -> None:
    """Google returns {} (empty object) when no fact-check exists — must return None."""
    result = normalize_response({})
    assert result is None


def test_normalize_response_empty_claims_list_returns_none() -> None:
    """Google returns {'claims': []} — must return None."""
    result = normalize_response({"claims": []})
    assert result is None


def test_normalize_response_non_dict_returns_none() -> None:
    """Non-dict input (e.g., None, list) must return None safely without raising."""
    for bad_input in (None, [], "string", 42):
        result = normalize_response(bad_input)  # type: ignore[arg-type]
        assert result is None, f"Expected None for input: {bad_input!r}"


def test_normalize_response_valid_returns_match() -> None:
    """A well-formed response produces a FactCheckMatch with correct fields."""
    result = normalize_response(_SAMPLE_FALSE_RESPONSE)
    assert result is not None
    assert isinstance(result, FactCheckMatch)
    assert result.verdict == "false"
    assert result.raw_rating == "False"
    assert result.publisher == "Example Fact Checker"
    assert len(result.sources) == 1


def test_normalize_response_verdict_derives_from_rating() -> None:
    """The verdict in the match corresponds to the normalized textualRating."""
    response = {
        "claims": [
            {
                "claimReview": [
                    {
                        "publisher": {"name": "Checker"},
                        "url": "https://checker.org/review",
                        "title": "Review",
                        "textualRating": "Pants on Fire",
                    }
                ]
            }
        ]
    }
    result = normalize_response(response)
    assert result is not None
    assert result.verdict == "false"
    assert result.confidence_score == pytest.approx(0.90)
    assert result.raw_rating == "Pants on Fire"


def test_normalize_response_missing_url_skips_that_source() -> None:
    """A claimReview entry without a url is skipped; others are included."""
    response = {
        "claims": [
            {
                "claimReview": [
                    {
                        "publisher": {"name": "Checker A"},
                        # no "url" key — this entry must be skipped
                        "title": "No URL review",
                        "textualRating": "False",
                    },
                    {
                        "publisher": {"name": "Checker B"},
                        "url": "https://checkerb.org/article",
                        "title": "Has URL",
                        "textualRating": "False",
                    },
                ]
            }
        ]
    }
    result = normalize_response(response)
    assert result is not None
    assert len(result.sources) == 1
    assert "checkerb.org" in str(result.sources[0].url)


def test_normalize_response_all_reviews_missing_url_returns_none() -> None:
    """If every claimReview in every claim lacks a url, no match can be built."""
    response = {
        "claims": [
            {
                "claimReview": [
                    {"publisher": {"name": "A"}, "textualRating": "False"},
                    {"publisher": {"name": "B"}, "textualRating": "True"},
                ]
            }
        ]
    }
    result = normalize_response(response)
    assert result is None


def test_normalize_response_multiple_reviews_all_collected_as_sources() -> None:
    """All valid claimReview entries from the selected claim appear in sources."""
    response = {
        "claims": [
            {
                "claimReview": [
                    {
                        "publisher": {"name": "Org A"},
                        "url": "https://orga.org/review",
                        "title": "Review A",
                        "textualRating": "False",
                    },
                    {
                        "publisher": {"name": "Org B"},
                        "url": "https://orgb.org/review",
                        "title": "Review B",
                        "textualRating": "False",
                    },
                ]
            }
        ]
    }
    result = normalize_response(response)
    assert result is not None
    assert len(result.sources) == 2
    urls = [str(s.url) for s in result.sources]
    assert any("orga.org" in u for u in urls)
    assert any("orgb.org" in u for u in urls)


def test_normalize_response_multiple_claims_uses_first_valid() -> None:
    """
    When multiple claims are returned, the first with a valid claimReview is used.
    This relies on Google's relevance ordering (AD-14) — no custom ranking applied.
    """
    response = {
        "claims": [
            {
                # First claim: valid
                "claimReview": [
                    {
                        "publisher": {"name": "First"},
                        "url": "https://first.org/review",
                        "title": "First review",
                        "textualRating": "True",
                    }
                ]
            },
            {
                # Second claim: also valid, but should not be selected
                "claimReview": [
                    {
                        "publisher": {"name": "Second"},
                        "url": "https://second.org/review",
                        "title": "Second review",
                        "textualRating": "False",
                    }
                ]
            },
        ]
    }
    result = normalize_response(response)
    assert result is not None
    assert result.publisher == "First"
    assert result.verdict == "true"


def test_normalize_response_first_claim_skipped_if_no_valid_reviews() -> None:
    """If the first claim has no valid reviews, the second claim is tried."""
    response = {
        "claims": [
            {
                # First claim has a review but no url — will be skipped
                "claimReview": [
                    {"publisher": {"name": "No URL"}, "textualRating": "False"}
                ]
            },
            {
                # Second claim is valid
                "claimReview": [
                    {
                        "publisher": {"name": "Has URL"},
                        "url": "https://valid.org/review",
                        "title": "Review",
                        "textualRating": "True",
                    }
                ]
            },
        ]
    }
    result = normalize_response(response)
    assert result is not None
    assert result.publisher == "Has URL"


def test_normalize_response_missing_title_uses_default() -> None:
    """A claimReview without a title uses the default 'Fact Check' — no fabrication."""
    response = {
        "claims": [
            {
                "claimReview": [
                    {
                        "publisher": {"name": "Checker"},
                        "url": "https://checker.org/article",
                        # no "title" key
                        "textualRating": "False",
                    }
                ]
            }
        ]
    }
    result = normalize_response(response)
    assert result is not None
    assert result.sources[0].title == "Fact Check"


def test_normalize_response_missing_publisher_name_uses_site() -> None:
    """When publisher.name is absent, publisher.site is used instead."""
    response = {
        "claims": [
            {
                "claimReview": [
                    {
                        "publisher": {"site": "fallback-site.org"},
                        "url": "https://fallback-site.org/article",
                        "title": "Review",
                        "textualRating": "False",
                    }
                ]
            }
        ]
    }
    result = normalize_response(response)
    assert result is not None
    assert result.publisher == "fallback-site.org"


def test_normalize_response_missing_publisher_both_uses_unknown() -> None:
    """When both publisher.name and publisher.site are absent, 'Unknown Publisher' is used."""
    response = {
        "claims": [
            {
                "claimReview": [
                    {
                        "publisher": {},
                        "url": "https://checker.org/article",
                        "title": "Review",
                        "textualRating": "False",
                    }
                ]
            }
        ]
    }
    result = normalize_response(response)
    assert result is not None
    assert result.publisher == "Unknown Publisher"


def test_normalize_response_sources_contain_only_api_data() -> None:
    """
    Source integrity: every field in every SourceItem must come directly from
    the API response.  No field may be invented or inferred.
    """
    api_url = "https://api-fact-checker.org/specific-article"
    api_title = "The specific article title from the API"
    api_publisher = "API Fact Checker Org"

    response = {
        "claims": [
            {
                "claimReview": [
                    {
                        "publisher": {"name": api_publisher},
                        "url": api_url,
                        "title": api_title,
                        "textualRating": "False",
                    }
                ]
            }
        ]
    }
    result = normalize_response(response)
    assert result is not None
    source = result.sources[0]
    assert source.title == api_title
    assert str(source.url).rstrip("/") == api_url.rstrip("/")
    assert source.publisher == api_publisher


def test_normalize_response_unknown_rating_produces_unverifiable() -> None:
    """An unknown textualRating in an otherwise valid response returns 'unverifiable'."""
    response = {
        "claims": [
            {
                "claimReview": [
                    {
                        "publisher": {"name": "Checker"},
                        "url": "https://checker.org/article",
                        "title": "Review",
                        "textualRating": "Completely Made Up Rating XYZ",
                    }
                ]
            }
        ]
    }
    result = normalize_response(response)
    assert result is not None
    assert result.verdict == "unverifiable"
    assert result.confidence_score == pytest.approx(0.30)
    assert result.raw_rating == "Completely Made Up Rating XYZ"


# ---------------------------------------------------------------------------
# C. query_factcheck_api() — async HTTP layer tests (use respx_mock)
# ---------------------------------------------------------------------------
# All tests in this section use respx_mock to prevent real network calls.
# Each test that makes an HTTP call must set a non-empty API key via monkeypatch.

pytestmark = pytest.mark.anyio


async def test_query_factcheck_api_missing_key_raises_before_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When GOOGLE_FACTCHECK_API_KEY is empty, FactCheckConfigError is raised
    BEFORE any HTTP request is made.  No respx_mock needed.
    """
    monkeypatch.setattr(
        config_module.settings, "GOOGLE_FACTCHECK_API_KEY", SecretStr("")
    )
    with pytest.raises(FactCheckConfigError):
        await query_factcheck_api(_CLAIM)


async def test_query_factcheck_api_returns_dict_on_success(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 200 response with valid JSON is returned as a plain dict."""
    monkeypatch.setattr(
        config_module.settings, "GOOGLE_FACTCHECK_API_KEY", SecretStr("test-key")
    )
    respx_mock.get(FACTCHECK_API_URL).respond(json=_SAMPLE_FALSE_RESPONSE)

    result = await query_factcheck_api(_CLAIM)

    assert isinstance(result, dict)
    assert "claims" in result


async def test_query_factcheck_api_empty_response_returned_as_dict(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Google returns {} when no fact-check exists — passes through as an empty dict."""
    monkeypatch.setattr(
        config_module.settings, "GOOGLE_FACTCHECK_API_KEY", SecretStr("test-key")
    )
    respx_mock.get(FACTCHECK_API_URL).respond(json={})

    result = await query_factcheck_api(_CLAIM)

    assert result == {}


async def test_query_factcheck_api_timeout_raises_factcheck_timeout(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timeout from httpx is wrapped in FactCheckTimeoutError."""
    import httpx

    monkeypatch.setattr(
        config_module.settings, "GOOGLE_FACTCHECK_API_KEY", SecretStr("test-key")
    )
    respx_mock.get(FACTCHECK_API_URL).mock(
        side_effect=httpx.ConnectTimeout("Connection timed out")
    )

    with pytest.raises(FactCheckTimeoutError):
        await query_factcheck_api(_CLAIM)


async def test_query_factcheck_api_service_unavailable_503_raises_service_error(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP 503 from Google is wrapped in FactCheckServiceError."""
    monkeypatch.setattr(
        config_module.settings, "GOOGLE_FACTCHECK_API_KEY", SecretStr("test-key")
    )
    respx_mock.get(FACTCHECK_API_URL).respond(status_code=503)

    with pytest.raises(FactCheckServiceError):
        await query_factcheck_api(_CLAIM)


async def test_query_factcheck_api_server_error_500_raises_service_error(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP 500 from Google is wrapped in FactCheckServiceError."""
    monkeypatch.setattr(
        config_module.settings, "GOOGLE_FACTCHECK_API_KEY", SecretStr("test-key")
    )
    respx_mock.get(FACTCHECK_API_URL).respond(status_code=500)

    with pytest.raises(FactCheckServiceError):
        await query_factcheck_api(_CLAIM)


async def test_query_factcheck_api_auth_failure_401_raises_auth_error(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP 401 (invalid API key) is wrapped in FactCheckAuthError."""
    monkeypatch.setattr(
        config_module.settings, "GOOGLE_FACTCHECK_API_KEY", SecretStr("bad-key")
    )
    respx_mock.get(FACTCHECK_API_URL).respond(status_code=401)

    with pytest.raises(FactCheckAuthError):
        await query_factcheck_api(_CLAIM)


async def test_query_factcheck_api_auth_failure_403_raises_auth_error(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP 403 (unauthorized) is wrapped in FactCheckAuthError."""
    monkeypatch.setattr(
        config_module.settings, "GOOGLE_FACTCHECK_API_KEY", SecretStr("unauthorized-key")
    )
    respx_mock.get(FACTCHECK_API_URL).respond(status_code=403)

    with pytest.raises(FactCheckAuthError):
        await query_factcheck_api(_CLAIM)


async def test_query_factcheck_api_quota_exceeded_429_raises_quota_error(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP 429 (rate limit / quota) is wrapped in FactCheckQuotaError."""
    monkeypatch.setattr(
        config_module.settings, "GOOGLE_FACTCHECK_API_KEY", SecretStr("test-key")
    )
    respx_mock.get(FACTCHECK_API_URL).respond(status_code=429)

    with pytest.raises(FactCheckQuotaError):
        await query_factcheck_api(_CLAIM)


async def test_query_factcheck_api_connection_error_raises_service_error(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A connection failure (refused, DNS, etc.) is wrapped in FactCheckServiceError."""
    import httpx

    monkeypatch.setattr(
        config_module.settings, "GOOGLE_FACTCHECK_API_KEY", SecretStr("test-key")
    )
    respx_mock.get(FACTCHECK_API_URL).mock(
        side_effect=httpx.ConnectError("Connection refused")
    )

    with pytest.raises(FactCheckServiceError):
        await query_factcheck_api(_CLAIM)


# ---------------------------------------------------------------------------
# D. verify_claim_factcheck() — end-to-end service tests (async, respx_mock)
# ---------------------------------------------------------------------------


async def test_verify_claim_factcheck_returns_match_on_valid_response(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full pipeline: valid Google response → FactCheckMatch with correct fields."""
    monkeypatch.setattr(
        config_module.settings, "GOOGLE_FACTCHECK_API_KEY", SecretStr("test-key")
    )
    respx_mock.get(FACTCHECK_API_URL).respond(json=_SAMPLE_FALSE_RESPONSE)

    result = await verify_claim_factcheck(_CLAIM)

    assert result is not None
    assert isinstance(result, FactCheckMatch)
    assert result.verdict == "false"
    assert result.confidence_score == pytest.approx(0.85)
    assert len(result.sources) == 1
    assert result.raw_rating == "False"
    assert result.publisher == "Example Fact Checker"


async def test_verify_claim_factcheck_returns_none_on_empty_response(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full pipeline: empty {} response → None (no fact-check found)."""
    monkeypatch.setattr(
        config_module.settings, "GOOGLE_FACTCHECK_API_KEY", SecretStr("test-key")
    )
    respx_mock.get(FACTCHECK_API_URL).respond(json={})

    result = await verify_claim_factcheck(_CLAIM)

    assert result is None


async def test_verify_claim_factcheck_propagates_config_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FactCheckConfigError propagates to caller when API key is empty."""
    monkeypatch.setattr(
        config_module.settings, "GOOGLE_FACTCHECK_API_KEY", SecretStr("")
    )
    with pytest.raises(FactCheckConfigError):
        await verify_claim_factcheck(_CLAIM)
