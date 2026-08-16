"""
tests/test_classifier.py — Full test suite for the hybrid claim classifier.

Test structure
--------------
    Section A: _classify_by_rules() — sync unit tests (no I/O, no fixtures needed).
               Tests the pure rule-based layer in isolation.

    Section B: classify_claim() with rules firing — async tests where rules fire
               confidently and Gemini is never called.  No respx_mock needed.

    Section C: classify_claim() with Gemini — async tests using respx_mock.
               Rules do not fire (claim is a plain factual statement), so the
               Gemini API is called and the mocked response is parsed.

    Section D: classify_claim() failure modes — Gemini errors of all kinds,
               all of which must cause a fallback to FACTUAL_CLAIM (never raise).

Isolation strategy
------------------
The `isolate_settings` autouse fixture (conftest.py) resets GEMINI_API_KEY
to SecretStr("") for every test.  Tests in Section C and D that need Gemini
must set a fake key via monkeypatch before calling classify_claim().
Tests in Section B do not need a key — the rules fire before Gemini is tried.

respx_mock usage
----------------
respx_mock is the respx pytest plugin fixture.  It intercepts ALL httpx
requests for the duration of each test, preventing accidental real API calls.
Any unregistered request raises an error.
"""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from app.core import config as config_module
from app.services.classifier import (
    GEMINI_CLASSIFY_URL,
    ClaimType,
    ClassifierConfigError,
    ClassifierServiceError,
    ClassifierTimeoutError,
    _classify_by_rules,  # noqa: PLC2701 — tested directly as a pure function
    classify_claim,
)

pytestmark = pytest.mark.anyio

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

# A gemini success response helper
def _gemini_response(label: str) -> dict:
    """Build a minimal Gemini generateContent response body."""
    return {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": label}]
                }
            }
        ]
    }


# ---------------------------------------------------------------------------
# A. _classify_by_rules() — sync unit tests
# ---------------------------------------------------------------------------


def test_rules_opinion_i_think() -> None:
    """'I think ...' is a strong opinion signal → OPINION."""
    result = _classify_by_rules("I think the economy is doing well.")
    assert result == ClaimType.OPINION


def test_rules_opinion_i_believe() -> None:
    """'I believe ...' → OPINION."""
    result = _classify_by_rules("I believe this policy is wrong.")
    assert result == ClaimType.OPINION


def test_rules_opinion_i_feel() -> None:
    """'I feel ...' → OPINION."""
    result = _classify_by_rules("I feel this approach is misguided.")
    assert result == ClaimType.OPINION


def test_rules_opinion_in_my_opinion() -> None:
    """'in my opinion' anywhere in text → OPINION."""
    result = _classify_by_rules("This is, in my opinion, the best approach.")
    assert result == ClaimType.OPINION


def test_rules_opinion_i_would_say() -> None:
    """'I would say ...' → OPINION."""
    result = _classify_by_rules("I would say this law is unfair.")
    assert result == ClaimType.OPINION


def test_rules_opinion_personally_i() -> None:
    """'personally, i' → OPINION."""
    result = _classify_by_rules("Personally, I find this film disappointing.")
    assert result == ClaimType.OPINION


def test_rules_opinion_case_insensitive() -> None:
    """Opinion matching is case-insensitive."""
    for variant in ("I THINK", "I Think", "i think", "I tHiNk"):
        text = f"{variant} this is wrong."
        result = _classify_by_rules(text)
        assert result == ClaimType.OPINION, f"Failed for: {text!r}"


def test_rules_ad_strong_buy_now() -> None:
    """'buy now' is a strong advertisement signal → ADVERTISEMENT."""
    result = _classify_by_rules("Get the best deal — buy now!")
    assert result == ClaimType.ADVERTISEMENT


def test_rules_ad_strong_promo_code() -> None:
    """'promo code' is a strong advertisement signal → ADVERTISEMENT."""
    result = _classify_by_rules("Use promo code SAVE20 for 20% off.")
    assert result == ClaimType.ADVERTISEMENT


def test_rules_ad_strong_free_shipping() -> None:
    """'free shipping' is a strong advertisement signal → ADVERTISEMENT."""
    result = _classify_by_rules("Order today and get free shipping on all items.")
    assert result == ClaimType.ADVERTISEMENT


def test_rules_ad_strong_limited_time_offer() -> None:
    """'limited time offer' → ADVERTISEMENT."""
    result = _classify_by_rules("This is a limited time offer — don't miss out.")
    assert result == ClaimType.ADVERTISEMENT


def test_rules_ad_weak_single_word_returns_none() -> None:
    """
    A single weak advertisement word (e.g., 'sale') is insufficient for
    classification.  Returns None so Gemini can decide.
    """
    result = _classify_by_rules("The annual sale of government bonds was oversubscribed.")
    assert result is None


def test_rules_ad_two_weak_words_returns_advertisement() -> None:
    """Two weak advertisement words together → ADVERTISEMENT."""
    result = _classify_by_rules("Huge sale and massive discount available now.")
    assert result == ClaimType.ADVERTISEMENT


def test_rules_factual_returns_none() -> None:
    """A plain factual claim has no OPINION or ADVERTISEMENT signals → None."""
    result = _classify_by_rules("The Earth orbits the Sun every 365.25 days.")
    assert result is None


def test_rules_should_alone_returns_none() -> None:
    """
    'should' alone must not classify as OPINION (high false-positive risk).
    'The government should publish the report' is a factual policy statement.
    """
    result = _classify_by_rules("The government should publish the report.")
    assert result is None


def test_rules_whitespace_handling() -> None:
    """Leading/trailing whitespace does not affect classification."""
    result = _classify_by_rules("   I think this is wrong.   ")
    assert result == ClaimType.OPINION


def test_rules_empty_string_returns_none() -> None:
    """An empty string has no signals → None."""
    result = _classify_by_rules("")
    assert result is None


def test_rules_opinion_from_my_perspective() -> None:
    """'from my perspective' → OPINION."""
    result = _classify_by_rules("From my perspective, the evidence is unconvincing.")
    assert result == ClaimType.OPINION


def test_rules_ad_use_code_returns_advertisement() -> None:
    """'use code' → ADVERTISEMENT."""
    result = _classify_by_rules("Use code PROMO10 at checkout.")
    assert result == ClaimType.ADVERTISEMENT


# ---------------------------------------------------------------------------
# B. classify_claim() — rules fire, no Gemini call needed
# ---------------------------------------------------------------------------
# GEMINI_API_KEY is empty (isolate_settings default), but these tests don't
# reach the Gemini layer because rules fire first.


async def test_classify_claim_opinion_via_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OPINION classified by rules without touching Gemini."""
    result = await classify_claim("I think the economy is struggling.")
    assert result == ClaimType.OPINION


async def test_classify_claim_advertisement_via_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ADVERTISEMENT classified by rules without touching Gemini."""
    result = await classify_claim("Buy now and get free shipping on all orders!")
    assert result == ClaimType.ADVERTISEMENT


async def test_classify_claim_no_key_returns_factual_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When GEMINI_API_KEY is empty (default in tests) and the rules don't fire,
    the Gemini layer raises ClassifierConfigError → classify_claim returns
    FACTUAL_CLAIM as the safe fallback.

    This tests requirement 12 (No Gemini API key → FACTUAL_CLAIM).
    """
    # GEMINI_API_KEY is already empty from isolate_settings.
    result = await classify_claim("The Earth orbits the Sun.")
    assert result == ClaimType.FACTUAL_CLAIM


# ---------------------------------------------------------------------------
# C. classify_claim() — Gemini layer (respx_mock required)
# ---------------------------------------------------------------------------


async def test_classify_claim_gemini_returns_factual_claim(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gemini correctly classifies a factual claim → FACTUAL_CLAIM."""
    monkeypatch.setattr(
        config_module.settings, "GEMINI_API_KEY", SecretStr("test-key")
    )
    respx_mock.post(GEMINI_CLASSIFY_URL).respond(
        json=_gemini_response("FACTUAL_CLAIM")
    )

    result = await classify_claim("The Earth is the third planet from the Sun.")
    assert result == ClaimType.FACTUAL_CLAIM


async def test_classify_claim_gemini_returns_ambiguous(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gemini classifies a borderline claim as AMBIGUOUS."""
    monkeypatch.setattr(
        config_module.settings, "GEMINI_API_KEY", SecretStr("test-key")
    )
    respx_mock.post(GEMINI_CLASSIFY_URL).respond(
        json=_gemini_response("AMBIGUOUS")
    )

    result = await classify_claim("Scientists question whether exercise extends lifespan.")
    assert result == ClaimType.AMBIGUOUS


async def test_classify_claim_gemini_returns_opinion(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gemini classifies a subtle opinion → OPINION (rules didn't catch it)."""
    monkeypatch.setattr(
        config_module.settings, "GEMINI_API_KEY", SecretStr("test-key")
    )
    respx_mock.post(GEMINI_CLASSIFY_URL).respond(
        json=_gemini_response("OPINION")
    )

    result = await classify_claim("This policy seems unfair to most people.")
    assert result == ClaimType.OPINION


async def test_classify_claim_gemini_label_case_insensitive(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gemini response parsing is case-insensitive (e.g., 'factual_claim' works)."""
    monkeypatch.setattr(
        config_module.settings, "GEMINI_API_KEY", SecretStr("test-key")
    )
    respx_mock.post(GEMINI_CLASSIFY_URL).respond(
        json=_gemini_response("factual_claim")
    )

    result = await classify_claim("The Eiffel Tower is located in Paris.")
    assert result == ClaimType.FACTUAL_CLAIM


async def test_classify_claim_gemini_unrecognized_label_returns_ambiguous(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Gemini returns a label that does not map to any ClaimType value.
    The parser must return AMBIGUOUS rather than raising or guessing.
    This tests requirement 11 (Invalid Gemini response → AMBIGUOUS).
    """
    monkeypatch.setattr(
        config_module.settings, "GEMINI_API_KEY", SecretStr("test-key")
    )
    respx_mock.post(GEMINI_CLASSIFY_URL).respond(
        json=_gemini_response("I'm not sure about this one.")
    )

    result = await classify_claim("The Eiffel Tower is located in Paris.")
    assert result == ClaimType.AMBIGUOUS


async def test_classify_claim_gemini_empty_response_returns_ambiguous(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gemini returns unexpected empty candidates → AMBIGUOUS (not a crash)."""
    monkeypatch.setattr(
        config_module.settings, "GEMINI_API_KEY", SecretStr("test-key")
    )
    respx_mock.post(GEMINI_CLASSIFY_URL).respond(json={"candidates": []})

    result = await classify_claim("The Eiffel Tower is located in Paris.")
    assert result == ClaimType.AMBIGUOUS


# ---------------------------------------------------------------------------
# D. classify_claim() failure modes — all must fall through to FACTUAL_CLAIM
# ---------------------------------------------------------------------------


async def test_classify_claim_gemini_timeout_returns_factual_claim(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Gemini request times out → classify_claim returns FACTUAL_CLAIM.
    This tests requirement 10 (Gemini timeout/failure → FACTUAL_CLAIM).
    """
    import httpx

    monkeypatch.setattr(
        config_module.settings, "GEMINI_API_KEY", SecretStr("test-key")
    )
    respx_mock.post(GEMINI_CLASSIFY_URL).mock(
        side_effect=httpx.ConnectTimeout("Connection timed out")
    )

    result = await classify_claim("The Eiffel Tower is located in Paris.")
    assert result == ClaimType.FACTUAL_CLAIM


async def test_classify_claim_gemini_connection_error_returns_factual_claim(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Gemini connection refused → classify_claim returns FACTUAL_CLAIM.
    This tests requirement 9 (Gemini unavailable → FACTUAL_CLAIM).
    """
    import httpx

    monkeypatch.setattr(
        config_module.settings, "GEMINI_API_KEY", SecretStr("test-key")
    )
    respx_mock.post(GEMINI_CLASSIFY_URL).mock(
        side_effect=httpx.ConnectError("Connection refused")
    )

    result = await classify_claim("The Eiffel Tower is located in Paris.")
    assert result == ClaimType.FACTUAL_CLAIM


async def test_classify_claim_gemini_500_error_returns_factual_claim(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gemini HTTP 500 → classify_claim returns FACTUAL_CLAIM."""
    monkeypatch.setattr(
        config_module.settings, "GEMINI_API_KEY", SecretStr("test-key")
    )
    respx_mock.post(GEMINI_CLASSIFY_URL).respond(status_code=500)

    result = await classify_claim("The Eiffel Tower is located in Paris.")
    assert result == ClaimType.FACTUAL_CLAIM


async def test_classify_claim_gemini_403_error_returns_factual_claim(
    respx_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gemini HTTP 403 (bad key) → classify_claim returns FACTUAL_CLAIM."""
    monkeypatch.setattr(
        config_module.settings, "GEMINI_API_KEY", SecretStr("bad-key")
    )
    respx_mock.post(GEMINI_CLASSIFY_URL).respond(status_code=403)

    result = await classify_claim("The Eiffel Tower is located in Paris.")
    assert result == ClaimType.FACTUAL_CLAIM


async def test_classify_claim_no_gemini_key_returns_factual_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    GEMINI_API_KEY is empty → classify_claim raises ClassifierConfigError
    internally, catches it, and returns FACTUAL_CLAIM.  No HTTP call is made.
    This tests requirement 12 explicitly with a plain factual claim text.
    """
    monkeypatch.setattr(
        config_module.settings, "GEMINI_API_KEY", SecretStr("")
    )
    # No respx_mock: if any HTTP call were made this would error.
    result = await classify_claim("The Eiffel Tower is located in Paris.")
    assert result == ClaimType.FACTUAL_CLAIM


async def test_classify_claim_never_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    classify_claim() must never raise, even when all underlying services fail.
    """
    monkeypatch.setattr(
        config_module.settings, "GEMINI_API_KEY", SecretStr("")
    )
    # Should not raise:
    result = await classify_claim("Some completely normal factual text.")
    assert isinstance(result, ClaimType)


async def test_classify_claim_returns_factual_claim_for_whitespace_only_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Whitespace-only text has no rule signals and Gemini is unavailable →
    FACTUAL_CLAIM.  (Whitespace-only is rejected at the route level before
    reaching classify_claim, so this tests the service layer in isolation.)
    """
    monkeypatch.setattr(
        config_module.settings, "GEMINI_API_KEY", SecretStr("")
    )
    result = await classify_claim("   ")
    assert result == ClaimType.FACTUAL_CLAIM
