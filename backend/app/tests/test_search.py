"""
tests/test_search.py — Comprehensive test suite for services/search.py.

Test structure
--------------
    Section A: _derive_domain_label() — sync unit tests (pure function).
    Section B: _normalize_result()    — sync unit tests (pure function).
    Section C: search_evidence()      — async integration tests using respx_mock.

Isolation strategy
------------------
The `isolate_settings` autouse fixture (conftest.py) resets TAVILY_API_KEY to
SecretStr("") for every test.  Tests in Section C that exercise the HTTP path
must:
    1. Set a fake key via monkeypatch.
    2. Provide respx_mock to intercept the HTTP call.

Tests that check the no-key path do NOT use respx_mock — any HTTP call would
cause an error, which is the desired safety property.
"""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from app.core import config as config_module
from app.services.search import (
    TAVILY_SEARCH_URL,
    SearchConfigError,
    SearchQuotaError,
    SearchResult,
    SearchServiceError,
    SearchTimeoutError,
    _derive_domain_label,
    _normalize_result,
    search_evidence,
)

pytestmark = pytest.mark.anyio

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_FAKE_KEY = "tvly-test-key-abc123"

_SAMPLE_RESULT = {
    "title": "Reuters: Scientists study coffee and cancer",
    "url": "https://www.reuters.com/health/coffee-cancer-2024-01-01",
    "content": "A new study suggests coffee may have protective effects against certain cancers.",
    "score": 0.92,
}

_SAMPLE_RESULT_2 = {
    "title": "NHS: Coffee and your health",
    "url": "https://www.nhs.uk/live-well/eat-well/food-types/coffee",
    "content": "The NHS advises moderate coffee consumption is not associated with health risks.",
    "score": 0.87,
}

_TAVILY_RESPONSE = {
    "query": "does coffee cause cancer",
    "results": [_SAMPLE_RESULT, _SAMPLE_RESULT_2],
    "response_time": 0.85,
}


# ---------------------------------------------------------------------------
# A. _derive_domain_label() — pure sync unit tests
# ---------------------------------------------------------------------------

def test_domain_label_strips_www() -> None:
    assert _derive_domain_label("https://www.reuters.com/article") == "reuters.com"


def test_domain_label_no_www() -> None:
    assert _derive_domain_label("https://reuters.com/article") == "reuters.com"


def test_domain_label_bbc_co_uk() -> None:
    assert _derive_domain_label("https://bbc.co.uk/news/world") == "bbc.co.uk"


def test_domain_label_www_bbc_co_uk() -> None:
    assert _derive_domain_label("https://www.bbc.co.uk/news/world") == "bbc.co.uk"


def test_domain_label_wikipedia() -> None:
    assert _derive_domain_label("https://en.wikipedia.org/wiki/Coffee") == "wikipedia.org"


def test_domain_label_nhs() -> None:
    assert _derive_domain_label("https://www.nhs.uk/live-well") == "nhs.uk"


def test_domain_label_http_scheme() -> None:
    """HTTP (not HTTPS) URLs should also be handled."""
    assert _derive_domain_label("http://example.com/page") == "example.com"


def test_domain_label_no_scheme_returns_none_or_string() -> None:
    """Bare hostnames without a scheme may return None; must not raise."""
    result = _derive_domain_label("not-a-url")
    # Either None or the bare string — must not crash
    assert result is None or isinstance(result, str)


def test_domain_label_empty_string_returns_none() -> None:
    result = _derive_domain_label("")
    assert result is None


def test_domain_label_only_www_strips_correctly() -> None:
    assert _derive_domain_label("https://www.snopes.com/fact-check/item") == "snopes.com"


# ---------------------------------------------------------------------------
# B. _normalize_result() — pure sync unit tests
# ---------------------------------------------------------------------------

def test_normalize_result_full_fields() -> None:
    result = _normalize_result(_SAMPLE_RESULT)
    assert result is not None
    assert result.title == "Reuters: Scientists study coffee and cancer"
    assert "reuters.com" in str(result.url)
    assert "coffee" in result.snippet
    assert result.publisher == "reuters.com"


def test_normalize_result_strips_www_from_publisher() -> None:
    raw = {**_SAMPLE_RESULT, "url": "https://www.reuters.com/health/item"}
    result = _normalize_result(raw)
    assert result is not None
    assert result.publisher == "reuters.com"


def test_normalize_result_missing_url_returns_none() -> None:
    raw = {k: v for k, v in _SAMPLE_RESULT.items() if k != "url"}
    result = _normalize_result(raw)
    assert result is None


def test_normalize_result_empty_url_returns_none() -> None:
    result = _normalize_result({**_SAMPLE_RESULT, "url": ""})
    assert result is None


def test_normalize_result_missing_content_returns_none() -> None:
    raw = {k: v for k, v in _SAMPLE_RESULT.items() if k != "content"}
    result = _normalize_result(raw)
    assert result is None


def test_normalize_result_empty_content_returns_none() -> None:
    result = _normalize_result({**_SAMPLE_RESULT, "content": ""})
    assert result is None


def test_normalize_result_missing_title_uses_empty_string() -> None:
    """Missing title defaults to empty string rather than causing a skip."""
    raw = {k: v for k, v in _SAMPLE_RESULT.items() if k != "title"}
    result = _normalize_result(raw)
    assert result is not None
    assert result.title == ""


def test_normalize_result_invalid_url_returns_none() -> None:
    result = _normalize_result({**_SAMPLE_RESULT, "url": "not-a-valid-url"})
    assert result is None


def test_normalize_result_returns_search_result_type() -> None:
    result = _normalize_result(_SAMPLE_RESULT)
    assert isinstance(result, SearchResult)


# ---------------------------------------------------------------------------
# C. search_evidence() — async tests using respx_mock
# ---------------------------------------------------------------------------

async def test_search_evidence_missing_key_raises_config_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing API key raises SearchConfigError before any HTTP call."""
    # TAVILY_API_KEY is already empty from isolate_settings.
    with pytest.raises(SearchConfigError):
        await search_evidence("The Earth is round.")


async def test_search_evidence_returns_normalized_results(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful Tavily response → list of normalized SearchResult objects."""
    monkeypatch.setattr(
        config_module.settings, "TAVILY_API_KEY", SecretStr(_FAKE_KEY)
    )
    respx_mock.post(TAVILY_SEARCH_URL).respond(json=_TAVILY_RESPONSE)

    results = await search_evidence("does coffee cause cancer")

    assert len(results) == 2
    assert all(isinstance(r, SearchResult) for r in results)
    assert results[0].title == "Reuters: Scientists study coffee and cancer"
    assert results[1].title == "NHS: Coffee and your health"


async def test_search_evidence_result_fields_normalized(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Result fields (url, snippet, publisher) are correctly mapped from Tavily."""
    monkeypatch.setattr(
        config_module.settings, "TAVILY_API_KEY", SecretStr(_FAKE_KEY)
    )
    respx_mock.post(TAVILY_SEARCH_URL).respond(json=_TAVILY_RESPONSE)

    results = await search_evidence("coffee and cancer")

    assert "reuters.com" in str(results[0].url)
    assert "coffee" in results[0].snippet
    assert results[0].publisher == "reuters.com"
    assert results[1].publisher == "nhs.uk"


async def test_search_evidence_empty_results_returns_empty_list(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tavily returns empty results list → search_evidence returns []."""
    monkeypatch.setattr(
        config_module.settings, "TAVILY_API_KEY", SecretStr(_FAKE_KEY)
    )
    respx_mock.post(TAVILY_SEARCH_URL).respond(
        json={"query": "test", "results": [], "response_time": 0.1}
    )

    results = await search_evidence("some claim")
    assert results == []


async def test_search_evidence_missing_results_key_returns_empty_list(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tavily response missing 'results' key → search_evidence returns []."""
    monkeypatch.setattr(
        config_module.settings, "TAVILY_API_KEY", SecretStr(_FAKE_KEY)
    )
    respx_mock.post(TAVILY_SEARCH_URL).respond(json={"query": "test"})

    results = await search_evidence("some claim")
    assert results == []


async def test_search_evidence_skips_result_missing_url(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Result with missing 'url' is skipped; other results are returned."""
    monkeypatch.setattr(
        config_module.settings, "TAVILY_API_KEY", SecretStr(_FAKE_KEY)
    )
    bad_result = {"title": "Bad", "content": "Some content"}  # no url
    respx_mock.post(TAVILY_SEARCH_URL).respond(
        json={"results": [bad_result, _SAMPLE_RESULT_2]}
    )

    results = await search_evidence("some claim")
    assert len(results) == 1
    assert "nhs.uk" in str(results[0].url)


async def test_search_evidence_skips_result_missing_content(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Result with missing 'content' (snippet) is skipped; others returned."""
    monkeypatch.setattr(
        config_module.settings, "TAVILY_API_KEY", SecretStr(_FAKE_KEY)
    )
    bad_result = {"title": "Bad", "url": "https://example.com/page"}  # no content
    respx_mock.post(TAVILY_SEARCH_URL).respond(
        json={"results": [bad_result, _SAMPLE_RESULT]}
    )

    results = await search_evidence("some claim")
    assert len(results) == 1
    assert "reuters.com" in str(results[0].url)


async def test_search_evidence_http_429_raises_quota_error(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP 429 from Tavily raises SearchQuotaError."""
    monkeypatch.setattr(
        config_module.settings, "TAVILY_API_KEY", SecretStr(_FAKE_KEY)
    )
    respx_mock.post(TAVILY_SEARCH_URL).respond(status_code=429)

    with pytest.raises(SearchQuotaError):
        await search_evidence("some claim")


async def test_search_evidence_http_500_raises_service_error(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP 500 from Tavily raises SearchServiceError."""
    monkeypatch.setattr(
        config_module.settings, "TAVILY_API_KEY", SecretStr(_FAKE_KEY)
    )
    respx_mock.post(TAVILY_SEARCH_URL).respond(status_code=500)

    with pytest.raises(SearchServiceError):
        await search_evidence("some claim")


async def test_search_evidence_http_403_raises_service_error(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP 403 (bad key) from Tavily raises SearchServiceError."""
    monkeypatch.setattr(
        config_module.settings, "TAVILY_API_KEY", SecretStr("bad-key")
    )
    respx_mock.post(TAVILY_SEARCH_URL).respond(status_code=403)

    with pytest.raises(SearchServiceError):
        await search_evidence("some claim")


async def test_search_evidence_timeout_raises_timeout_error(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Connection timeout raises SearchTimeoutError."""
    import httpx as _httpx

    monkeypatch.setattr(
        config_module.settings, "TAVILY_API_KEY", SecretStr(_FAKE_KEY)
    )
    respx_mock.post(TAVILY_SEARCH_URL).mock(
        side_effect=_httpx.ConnectTimeout("timed out")
    )

    with pytest.raises(SearchTimeoutError):
        await search_evidence("some claim")


async def test_search_evidence_connection_error_raises_service_error(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Connection refused raises SearchServiceError."""
    import httpx as _httpx

    monkeypatch.setattr(
        config_module.settings, "TAVILY_API_KEY", SecretStr(_FAKE_KEY)
    )
    respx_mock.post(TAVILY_SEARCH_URL).mock(
        side_effect=_httpx.ConnectError("connection refused")
    )

    with pytest.raises(SearchServiceError):
        await search_evidence("some claim")


async def test_search_evidence_max_results_respected(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SEARCH_MAX_RESULTS setting is sent in the Tavily request body."""
    monkeypatch.setattr(
        config_module.settings, "TAVILY_API_KEY", SecretStr(_FAKE_KEY)
    )
    monkeypatch.setattr(config_module.settings, "SEARCH_MAX_RESULTS", 3)

    captured_body = {}

    def capture(request):
        import json
        captured_body.update(json.loads(request.content))
        import httpx as _httpx
        return _httpx.Response(200, json={"results": [_SAMPLE_RESULT]})

    respx_mock.post(TAVILY_SEARCH_URL).mock(side_effect=capture)

    await search_evidence("some claim")
    assert captured_body.get("max_results") == 3


async def test_search_evidence_search_depth_always_basic(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """search_depth is always 'basic' (1 credit/request), never 'advanced'."""
    monkeypatch.setattr(
        config_module.settings, "TAVILY_API_KEY", SecretStr(_FAKE_KEY)
    )

    captured_body = {}

    def capture(request):
        import json
        captured_body.update(json.loads(request.content))
        import httpx as _httpx
        return _httpx.Response(200, json={"results": []})

    respx_mock.post(TAVILY_SEARCH_URL).mock(side_effect=capture)

    await search_evidence("some claim")
    assert captured_body.get("search_depth") == "basic"


async def test_search_evidence_include_answer_false(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """include_answer is always False to avoid consuming extra credits."""
    monkeypatch.setattr(
        config_module.settings, "TAVILY_API_KEY", SecretStr(_FAKE_KEY)
    )

    captured_body = {}

    def capture(request):
        import json
        captured_body.update(json.loads(request.content))
        import httpx as _httpx
        return _httpx.Response(200, json={"results": []})

    respx_mock.post(TAVILY_SEARCH_URL).mock(side_effect=capture)

    await search_evidence("some claim")
    assert captured_body.get("include_answer") is False


async def test_search_evidence_multiple_results_all_returned(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All valid results in the Tavily response are returned."""
    monkeypatch.setattr(
        config_module.settings, "TAVILY_API_KEY", SecretStr(_FAKE_KEY)
    )
    three_results = {
        "results": [
            _SAMPLE_RESULT,
            _SAMPLE_RESULT_2,
            {
                "title": "Snopes fact check",
                "url": "https://www.snopes.com/fact-check/coffee",
                "content": "Snopes rates the claim as mostly false.",
                "score": 0.75,
            },
        ]
    }
    respx_mock.post(TAVILY_SEARCH_URL).respond(json=three_results)

    results = await search_evidence("coffee causes cancer")
    assert len(results) == 3
    assert results[2].publisher == "snopes.com"
