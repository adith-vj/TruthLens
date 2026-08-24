"""
tests/test_llm.py — Comprehensive test suite for services/llm.py.

Test structure
--------------
    Section A: _build_prompt()        — sync unit tests (pure function).
    Section B: _parse_llm_response()  — sync unit tests (pure function).
    Section C: verify_with_llm()      — async integration tests via respx_mock.

Isolation strategy
------------------
The `isolate_settings` autouse fixture resets GEMINI_API_KEY to SecretStr("")
for every test.  Tests that need a key set a fake one via monkeypatch.

The `evidence` list used in these tests is deterministic — index 0 and index 1
correspond to known SearchResult objects so source provenance assertions are
unambiguous.
"""

from __future__ import annotations

import json

import pytest
from pydantic import SecretStr, AnyUrl

from app.core import config as config_module
from app.services.llm import (
    GEMINI_VERIFY_URL,
    LLMConfigError,
    LLMParseError,
    LLMQuotaError,
    LLMServiceError,
    LLMTimeoutError,
    LLMVerdict,
    _build_prompt,
    _parse_llm_response,
    verify_with_llm,
)
from app.services.search import SearchResult

pytestmark = pytest.mark.anyio

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

_FAKE_KEY = "AIzaSy-fake-test-key"

_EVIDENCE = [
    SearchResult(
        title="Reuters: Coffee study findings",
        url=AnyUrl("https://reuters.com/health/coffee"),
        snippet="A new study found coffee reduces certain cancer risks by 12%.",
        publisher="reuters.com",
    ),
    SearchResult(
        title="WHO: Cancer and diet",
        url=AnyUrl("https://who.int/cancer/diet"),
        snippet="The WHO states that coffee is not classified as a carcinogen.",
        publisher="who.int",
    ),
]


def _gemini_body(
    verdict: str = "false",
    confidence: float = 0.82,
    indices: list = None,
) -> dict:
    """Build a minimal Gemini generateContent response body."""
    payload = {
        "verdict": verdict,
        "confidence_score": confidence,
        "source_indices": indices if indices is not None else [0, 1],
    }
    return {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": json.dumps(payload)}]
                }
            }
        ]
    }


# ---------------------------------------------------------------------------
# A. _build_prompt() — sync unit tests
# ---------------------------------------------------------------------------

def test_build_prompt_contains_claim() -> None:
    prompt = _build_prompt("Coffee causes cancer.", _EVIDENCE)
    assert "Coffee causes cancer." in prompt


def test_build_prompt_contains_titles() -> None:
    prompt = _build_prompt("claim", _EVIDENCE)
    assert "Reuters: Coffee study findings" in prompt
    assert "WHO: Cancer and diet" in prompt


def test_build_prompt_contains_snippets() -> None:
    prompt = _build_prompt("claim", _EVIDENCE)
    assert "coffee reduces certain cancer risks" in prompt
    assert "WHO states that coffee is not classified" in prompt


def test_build_prompt_does_not_contain_urls() -> None:
    """URLs must NEVER appear in the prompt to prevent LLM from citing them."""
    prompt = _build_prompt("claim", _EVIDENCE)
    assert "reuters.com/health/coffee" not in prompt
    assert "who.int/cancer/diet" not in prompt
    assert "https://" not in prompt


def test_build_prompt_contains_indexed_evidence() -> None:
    """Evidence items are labelled with [0], [1], etc."""
    prompt = _build_prompt("claim", _EVIDENCE)
    assert "[0]" in prompt
    assert "[1]" in prompt


def test_build_prompt_contains_verdict_instructions() -> None:
    """Prompt includes the four valid verdict values."""
    prompt = _build_prompt("claim", _EVIDENCE)
    for verdict in ("true", "false", "misleading", "unverifiable"):
        assert verdict in prompt


def test_build_prompt_contains_unverifiable_rules() -> None:
    """Prompt explicitly instructs model to use 'unverifiable' for insufficient evidence."""
    prompt = _build_prompt("claim", _EVIDENCE)
    assert "unverifiable" in prompt
    assert "insufficient" in prompt.lower() or "Insufficient" in prompt


def test_build_prompt_single_evidence_item() -> None:
    """Prompt works correctly with a single evidence item."""
    prompt = _build_prompt("claim", [_EVIDENCE[0]])
    assert "[0]" in prompt
    assert "[1]" not in prompt


# ---------------------------------------------------------------------------
# B. _parse_llm_response() — sync unit tests
# ---------------------------------------------------------------------------

def test_parse_returns_llm_verdict() -> None:
    result = _parse_llm_response(_gemini_body("false", 0.82, [0, 1]))
    assert isinstance(result, LLMVerdict)


def test_parse_verdict_true() -> None:
    result = _parse_llm_response(_gemini_body("true", 0.9, [0]))
    assert result.verdict == "true"


def test_parse_verdict_false() -> None:
    result = _parse_llm_response(_gemini_body("false", 0.8, []))
    assert result.verdict == "false"


def test_parse_verdict_misleading() -> None:
    result = _parse_llm_response(_gemini_body("misleading", 0.7, [1]))
    assert result.verdict == "misleading"


def test_parse_verdict_unverifiable() -> None:
    result = _parse_llm_response(_gemini_body("unverifiable", 0.1, []))
    assert result.verdict == "unverifiable"
    assert result.confidence_score == pytest.approx(0.1)


def test_parse_source_indices_correct() -> None:
    result = _parse_llm_response(_gemini_body("false", 0.8, [0, 1]))
    assert result.source_indices == [0, 1]


def test_parse_empty_source_indices() -> None:
    result = _parse_llm_response(_gemini_body("unverifiable", 0.1, []))
    assert result.source_indices == []


def test_parse_confidence_zero() -> None:
    result = _parse_llm_response(_gemini_body("unverifiable", 0.0, []))
    assert result.confidence_score == pytest.approx(0.0)


def test_parse_confidence_one() -> None:
    result = _parse_llm_response(_gemini_body("true", 1.0, [0]))
    assert result.confidence_score == pytest.approx(1.0)


def test_parse_confidence_above_one_raises_parse_error() -> None:
    """Confidence > 1.0 must raise LLMParseError (not silently clamped)."""
    with pytest.raises(LLMParseError, match="confidence_score"):
        _parse_llm_response(_gemini_body("true", 1.5, [0]))


def test_parse_confidence_negative_raises_parse_error() -> None:
    """Confidence < 0.0 must raise LLMParseError."""
    with pytest.raises(LLMParseError, match="confidence_score"):
        _parse_llm_response(_gemini_body("true", -0.1, [0]))


def test_parse_confidence_string_raises_parse_error() -> None:
    """Non-numeric confidence must raise LLMParseError."""
    payload = {"verdict": "true", "confidence_score": "high", "source_indices": []}
    body = {
        "candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]
    }
    with pytest.raises(LLMParseError, match="confidence_score"):
        _parse_llm_response(body)


def test_parse_missing_confidence_raises_parse_error() -> None:
    payload = {"verdict": "true", "source_indices": []}
    body = {
        "candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]
    }
    with pytest.raises(LLMParseError, match="confidence_score"):
        _parse_llm_response(body)


def test_parse_invalid_verdict_raises_parse_error() -> None:
    with pytest.raises(LLMParseError, match="verdict"):
        _parse_llm_response(_gemini_body("maybe", 0.5, []))


def test_parse_missing_verdict_raises_parse_error() -> None:
    payload = {"confidence_score": 0.5, "source_indices": []}
    body = {
        "candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]
    }
    with pytest.raises(LLMParseError, match="verdict"):
        _parse_llm_response(body)


def test_parse_non_json_text_raises_parse_error() -> None:
    body = {
        "candidates": [{"content": {"parts": [{"text": "I cannot determine the answer."}]}}]
    }
    with pytest.raises(LLMParseError):
        _parse_llm_response(body)


def test_parse_empty_candidates_raises_parse_error() -> None:
    body = {"candidates": []}
    with pytest.raises(LLMParseError, match="candidates"):
        _parse_llm_response(body)


def test_parse_no_candidates_key_raises_parse_error() -> None:
    body = {}
    with pytest.raises(LLMParseError, match="candidates"):
        _parse_llm_response(body)


def test_parse_non_integer_source_index_discarded() -> None:
    """Non-integer items in source_indices are discarded, not an error."""
    payload = {
        "verdict": "true",
        "confidence_score": 0.8,
        "source_indices": [0, "bad", 1],
    }
    body = {
        "candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]
    }
    result = _parse_llm_response(body)
    assert result.source_indices == [0, 1]


def test_parse_whole_number_float_index_accepted() -> None:
    """JSON doesn't distinguish int from float; 1.0 should be accepted as 1."""
    payload = {
        "verdict": "true",
        "confidence_score": 0.8,
        "source_indices": [0.0, 1.0],
    }
    body = {
        "candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]
    }
    result = _parse_llm_response(body)
    assert result.source_indices == [0, 1]


def test_parse_source_indices_not_list_treated_as_empty() -> None:
    """source_indices that is not a list is treated as empty."""
    payload = {
        "verdict": "true",
        "confidence_score": 0.8,
        "source_indices": "0, 1",
    }
    body = {
        "candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]
    }
    result = _parse_llm_response(body)
    assert result.source_indices == []


# ---------------------------------------------------------------------------
# C. verify_with_llm() — async integration tests
# ---------------------------------------------------------------------------

async def test_verify_with_llm_missing_key_raises_config_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing API key raises LLMConfigError before any HTTP call."""
    # GEMINI_API_KEY is already empty from isolate_settings.
    with pytest.raises(LLMConfigError):
        await verify_with_llm("some claim", _EVIDENCE)


async def test_verify_with_llm_returns_false_verdict(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful Gemini response → LLMVerdict with correct fields."""
    monkeypatch.setattr(
        config_module.settings, "GEMINI_API_KEY", SecretStr(_FAKE_KEY)
    )
    respx_mock.post(GEMINI_VERIFY_URL).respond(
        json=_gemini_body("false", 0.82, [0, 1])
    )

    verdict = await verify_with_llm("Coffee causes cancer.", _EVIDENCE)

    assert isinstance(verdict, LLMVerdict)
    assert verdict.verdict == "false"
    assert verdict.confidence_score == pytest.approx(0.82)
    assert verdict.source_indices == [0, 1]


async def test_verify_with_llm_returns_true_verdict(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        config_module.settings, "GEMINI_API_KEY", SecretStr(_FAKE_KEY)
    )
    respx_mock.post(GEMINI_VERIFY_URL).respond(
        json=_gemini_body("true", 0.91, [0])
    )

    verdict = await verify_with_llm("Coffee is safe.", _EVIDENCE)
    assert verdict.verdict == "true"
    assert verdict.confidence_score == pytest.approx(0.91)


async def test_verify_with_llm_returns_unverifiable(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """'unverifiable' is a valid verdict, not an error."""
    monkeypatch.setattr(
        config_module.settings, "GEMINI_API_KEY", SecretStr(_FAKE_KEY)
    )
    respx_mock.post(GEMINI_VERIFY_URL).respond(
        json=_gemini_body("unverifiable", 0.1, [])
    )

    verdict = await verify_with_llm("claim", _EVIDENCE)
    assert verdict.verdict == "unverifiable"
    assert verdict.source_indices == []


async def test_verify_with_llm_returns_misleading(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        config_module.settings, "GEMINI_API_KEY", SecretStr(_FAKE_KEY)
    )
    respx_mock.post(GEMINI_VERIFY_URL).respond(
        json=_gemini_body("misleading", 0.65, [1])
    )

    verdict = await verify_with_llm("claim", _EVIDENCE)
    assert verdict.verdict == "misleading"
    assert verdict.source_indices == [1]


async def test_verify_with_llm_http_429_raises_quota_error(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        config_module.settings, "GEMINI_API_KEY", SecretStr(_FAKE_KEY)
    )
    respx_mock.post(GEMINI_VERIFY_URL).respond(status_code=429)

    with pytest.raises(LLMQuotaError):
        await verify_with_llm("claim", _EVIDENCE)


async def test_verify_with_llm_http_500_raises_service_error(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        config_module.settings, "GEMINI_API_KEY", SecretStr(_FAKE_KEY)
    )
    respx_mock.post(GEMINI_VERIFY_URL).respond(status_code=500)

    with pytest.raises(LLMServiceError):
        await verify_with_llm("claim", _EVIDENCE)


async def test_verify_with_llm_http_403_raises_service_error(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        config_module.settings, "GEMINI_API_KEY", SecretStr("bad-key")
    )
    respx_mock.post(GEMINI_VERIFY_URL).respond(status_code=403)

    with pytest.raises(LLMServiceError):
        await verify_with_llm("claim", _EVIDENCE)


async def test_verify_with_llm_timeout_raises_timeout_error(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx as _httpx

    monkeypatch.setattr(
        config_module.settings, "GEMINI_API_KEY", SecretStr(_FAKE_KEY)
    )
    respx_mock.post(GEMINI_VERIFY_URL).mock(
        side_effect=_httpx.ConnectTimeout("timed out")
    )

    with pytest.raises(LLMTimeoutError):
        await verify_with_llm("claim", _EVIDENCE)


async def test_verify_with_llm_connection_error_raises_service_error(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx as _httpx

    monkeypatch.setattr(
        config_module.settings, "GEMINI_API_KEY", SecretStr(_FAKE_KEY)
    )
    respx_mock.post(GEMINI_VERIFY_URL).mock(
        side_effect=_httpx.ConnectError("refused")
    )

    with pytest.raises(LLMServiceError):
        await verify_with_llm("claim", _EVIDENCE)


async def test_verify_with_llm_invalid_verdict_raises_parse_error(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid verdict string in Gemini JSON → LLMParseError."""
    monkeypatch.setattr(
        config_module.settings, "GEMINI_API_KEY", SecretStr(_FAKE_KEY)
    )
    respx_mock.post(GEMINI_VERIFY_URL).respond(
        json=_gemini_body("maybe", 0.5, [])
    )

    with pytest.raises(LLMParseError, match="verdict"):
        await verify_with_llm("claim", _EVIDENCE)


async def test_verify_with_llm_malformed_json_raises_parse_error(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-JSON text in Gemini response → LLMParseError."""
    monkeypatch.setattr(
        config_module.settings, "GEMINI_API_KEY", SecretStr(_FAKE_KEY)
    )
    bad_body = {
        "candidates": [
            {"content": {"parts": [{"text": "I cannot determine this."}]}}
        ]
    }
    respx_mock.post(GEMINI_VERIFY_URL).respond(json=bad_body)

    with pytest.raises(LLMParseError):
        await verify_with_llm("claim", _EVIDENCE)


async def test_verify_with_llm_invalid_confidence_raises_parse_error(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Confidence > 1.0 raises LLMParseError (not silently clamped)."""
    monkeypatch.setattr(
        config_module.settings, "GEMINI_API_KEY", SecretStr(_FAKE_KEY)
    )
    respx_mock.post(GEMINI_VERIFY_URL).respond(
        json=_gemini_body("true", 1.5, [0])
    )

    with pytest.raises(LLMParseError, match="confidence_score"):
        await verify_with_llm("claim", _EVIDENCE)


async def test_verify_with_llm_prompt_excludes_urls(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Gemini request body must not contain any URL text."""
    monkeypatch.setattr(
        config_module.settings, "GEMINI_API_KEY", SecretStr(_FAKE_KEY)
    )

    captured_prompt: list[str] = []

    def capture(request):
        body = json.loads(request.content)
        text = body["contents"][0]["parts"][0]["text"]
        captured_prompt.append(text)
        import httpx as _httpx
        return _httpx.Response(200, json=_gemini_body("false", 0.8, [0]))

    respx_mock.post(GEMINI_VERIFY_URL).mock(side_effect=capture)

    await verify_with_llm("Coffee causes cancer.", _EVIDENCE)

    assert captured_prompt, "No request was captured"
    prompt = captured_prompt[0]
    assert "https://" not in prompt
    assert "reuters.com/health" not in prompt
    assert "who.int/cancer" not in prompt


async def test_verify_with_llm_prompt_includes_titles(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Gemini request prompt must include evidence titles."""
    monkeypatch.setattr(
        config_module.settings, "GEMINI_API_KEY", SecretStr(_FAKE_KEY)
    )

    captured_prompt: list[str] = []

    def capture(request):
        body = json.loads(request.content)
        captured_prompt.append(body["contents"][0]["parts"][0]["text"])
        import httpx as _httpx
        return _httpx.Response(200, json=_gemini_body("false", 0.8, [0]))

    respx_mock.post(GEMINI_VERIFY_URL).mock(side_effect=capture)

    await verify_with_llm("Coffee causes cancer.", _EVIDENCE)

    prompt = captured_prompt[0]
    assert "Reuters: Coffee study findings" in prompt
    assert "WHO: Cancer and diet" in prompt


async def test_verify_with_llm_out_of_range_indices_preserved_in_verdict(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Out-of-range indices are returned in LLMVerdict.source_indices — it is
    the route handler's responsibility to discard them, not the service's.
    """
    monkeypatch.setattr(
        config_module.settings, "GEMINI_API_KEY", SecretStr(_FAKE_KEY)
    )
    respx_mock.post(GEMINI_VERIFY_URL).respond(
        json=_gemini_body("true", 0.75, [0, 99])   # 99 is out of range
    )

    verdict = await verify_with_llm("claim", _EVIDENCE)
    # Service preserves the indices; route handler will filter 99
    assert 99 in verdict.source_indices


async def test_verify_with_llm_uses_json_mode(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Gemini request must use responseMimeType: application/json."""
    monkeypatch.setattr(
        config_module.settings, "GEMINI_API_KEY", SecretStr(_FAKE_KEY)
    )

    captured_config: list[dict] = []

    def capture(request):
        body = json.loads(request.content)
        captured_config.append(body.get("generationConfig", {}))
        import httpx as _httpx
        return _httpx.Response(200, json=_gemini_body("false", 0.8, []))

    respx_mock.post(GEMINI_VERIFY_URL).mock(side_effect=capture)

    await verify_with_llm("claim", _EVIDENCE)

    assert captured_config[0].get("responseMimeType") == "application/json"
