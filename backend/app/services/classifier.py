"""
services/classifier.py — Hybrid claim type classifier.

Architecture
------------
This module classifies a user-submitted text into one of four types before
the verification pipeline runs.  It uses a two-layer hybrid approach:

    Layer 1: Rule-based heuristics (_classify_by_rules)
        Fast, synchronous, zero I/O.  Fires confidently on textbook opinion
        phrases ("I think", "in my opinion") and strong commercial signals
        ("buy now", "promo code").  Returns None when uncertain so that
        Layer 2 can make a better decision.

    Layer 2: Gemini zero-shot classifier (_classify_with_gemini)
        Called only when Layer 1 returns None.  Uses gemini-1.5-flash with
        temperature=0 for a deterministic, low-latency classification call.
        A strict prompt forces exactly one label from the four allowed values.
        Defensive response parsing: any unrecognized label maps to AMBIGUOUS.

    Orchestrator: classify_claim (public async entry point)
        Runs Layer 1, then Layer 2 if needed.  ALL exceptions from Layer 2
        are caught and logged; the function always returns a ClaimType.
        Gemini failure → FACTUAL_CLAIM (resilient fallthrough).

Classifier failure handling
---------------------------
A classifier failure must never prevent the fact-checking pipeline from
running.  The classifier is an optimization layer, not the source of truth.

    Gemini unavailable / timeout / no API key → FACTUAL_CLAIM
    Gemini returns unrecognized label         → AMBIGUOUS
    Any other unexpected exception           → FACTUAL_CLAIM

AMBIGUOUS confidence reduction
-------------------------------
When a claim is AMBIGUOUS the route handler multiplies the factcheck
confidence score by 0.7.  This reduction is applied in api/verify.py, NOT
here, to keep this module's concerns limited to classification only.

Security rules
--------------
- GEMINI_API_KEY is never logged.
- Raw Gemini error messages are never forwarded to the client.
- All exception messages in logs use static strings or safe metadata
  (exception type names, HTTP status codes) only.
"""

from __future__ import annotations

import re
from enum import Enum

import httpx

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 0. ClaimType enum
# ---------------------------------------------------------------------------


class ClaimType(str, Enum):
    """
    The category of an input text as determined by the claim classifier.

    Values:
        FACTUAL_CLAIM:  Text makes a specific, verifiable assertion of fact.
                        Proceed to Google Fact Check and/or LLM verification.
        OPINION:        Text expresses a subjective view or preference.
                        Return verdict='unverifiable' without querying sources.
        ADVERTISEMENT:  Text is a call to action, promotional, or commercial.
                        Return verdict='unverifiable' without querying sources.
        AMBIGUOUS:      Text cannot be clearly classified. Proceed with lower
                        confidence weighting (final confidence × 0.7).
    """

    FACTUAL_CLAIM = "factual_claim"
    OPINION = "opinion"
    ADVERTISEMENT = "advertisement"
    AMBIGUOUS = "ambiguous"

# ---------------------------------------------------------------------------
# Re-export ClaimType so callers only need to import from this module
# ---------------------------------------------------------------------------
__all__ = [
    "ClaimType",
    "ClassifierError",
    "ClassifierConfigError",
    "ClassifierQuotaError",
    "ClassifierTimeoutError",
    "ClassifierServiceError",
    "GEMINI_CLASSIFY_URL",
    "classify_claim",
]

# ---------------------------------------------------------------------------
# 1. Exception hierarchy
# ---------------------------------------------------------------------------


class ClassifierError(Exception):
    """Base class for all classifier service errors."""


class ClassifierQuotaError(ClassifierError):
    """Raised when Gemini returns HTTP 429."""


class ClassifierConfigError(ClassifierError):
    """
    Raised before any network request when GEMINI_API_KEY is missing or empty.
    The orchestrator catches this and falls through to FACTUAL_CLAIM so the
    server remains functional without a configured Gemini key.
    """


class ClassifierTimeoutError(ClassifierError):
    """Raised when the Gemini API request exceeds CLASSIFIER_TIMEOUT_SECONDS."""


class ClassifierServiceError(ClassifierError):
    """Raised on HTTP errors (4xx/5xx) or connection failures from Gemini."""


# ---------------------------------------------------------------------------
# 2. Constants
# ---------------------------------------------------------------------------

# Exported so tests can import it for respx URL matching.
GEMINI_CLASSIFY_URL = (
    "https://generativelanguage.googleapis.com"
    "/v1beta/models/gemini-3.5-flash-lite:generateContent"
)

# Strict zero-shot classification prompt.
# The trailing "Category:" completion cue steers the model toward a
# single-token response and makes parsing reliable.
_CLASSIFIER_PROMPT = """\
Classify the following text into exactly one category.

Categories:
FACTUAL_CLAIM - A specific, verifiable statement of fact about the world.
OPINION       - A personal view, subjective belief, preference, or value judgment.
ADVERTISEMENT - Commercial promotion, sales pitch, or call to action.
AMBIGUOUS     - Cannot be clearly classified as any of the above.

Rules:
- Reply with ONLY the category name. No explanation, no punctuation, no extra text.
- Choose the single best category.
- When uncertain, use AMBIGUOUS.

Text:
{text}

Category:"""

# ---------------------------------------------------------------------------
# 3. Rule-based phrase lists
# ---------------------------------------------------------------------------

# Strong opinion phrases — any one is sufficient for OPINION classification.
# These are exact substring matches in the lowercased claim text.
# Intentionally conservative: "should", "best", "worst" are excluded because
# they generate too many false positives (e.g. "The government should publish
# the report" is a factual/policy claim, not an opinion).
_OPINION_PHRASES: tuple[str, ...] = (
    "i think ",
    "i think,",
    "i believe ",
    "i believe,",
    "i feel ",
    "i feel,",
    "in my opinion",
    "i would say",
    "i would argue",
    "from my perspective",
    "in my view",
    "from my point of view",
    "i personally think",
    "i personally believe",
    "i personally feel",
    "personally, i",
    "personally i think",
    "personally i believe",
)

# Strong advertisement phrases — any one alone is sufficient.
_AD_STRONG_PHRASES: tuple[str, ...] = (
    "buy now",
    "shop now",
    "order now",
    "subscribe now",
    "free shipping",
    "use code",
    "promo code",
    "discount code",
    "limited time offer",
    "limited offer",
    "add to cart",
    "flash sale",
    "sale ends",
    "click here to buy",
    "get yours now",
    "act now",
    "exclusive deal",
    "limited time only",
)

# Weak advertisement words — two or more must be present simultaneously.
_AD_WEAK_PHRASES: tuple[str, ...] = (
    "sale",
    "discount",
    "deal",
    "offer",
)

# Mapping from normalized Gemini response labels to ClaimType members.
# Built once at import time; includes every valid ClaimType value.
_LABEL_TO_CLAIM_TYPE: dict[str, ClaimType] = {ct.value: ct for ct in ClaimType}
# Add common shorthand mappings that the model might generate
_LABEL_TO_CLAIM_TYPE["fact"] = ClaimType.FACTUAL_CLAIM
_LABEL_TO_CLAIM_TYPE["factual"] = ClaimType.FACTUAL_CLAIM


# ---------------------------------------------------------------------------
# 4. Rule-based classifier (sync, pure — no I/O)
# ---------------------------------------------------------------------------


def _classify_by_rules(text: str) -> ClaimType | None:
    """
    Classify the text using conservative keyword heuristics.

    Returns a ClaimType when the evidence is strong and unambiguous.
    Returns None when the evidence is insufficient — the caller should
    then escalate to the Gemini layer.

    This function is synchronous and performs zero I/O.  It is designed
    to be independently unit-testable without any mocking overhead.

    Args:
        text: The raw claim text (should be stripped but not required).

    Returns:
        ClaimType if a confident classification was made, else None.
    """
    normalised = text.lower()

    # --- OPINION check ---
    for phrase in _OPINION_PHRASES:
        if phrase in normalised:
            logger.debug("OPINION detected by rule (phrase=%r)", phrase)
            return ClaimType.OPINION

    # --- ADVERTISEMENT check (strong single indicator) ---
    for phrase in _AD_STRONG_PHRASES:
        if phrase in normalised:
            logger.debug("ADVERTISEMENT detected by rule (strong phrase=%r)", phrase)
            return ClaimType.ADVERTISEMENT

    # --- ADVERTISEMENT check (weak: require 2+ indicators) ---
    weak_matches = sum(1 for phrase in _AD_WEAK_PHRASES if phrase in normalised)
    if weak_matches >= 2:
        logger.debug(
            "ADVERTISEMENT detected by rule (%d weak indicators present)", weak_matches
        )
        return ClaimType.ADVERTISEMENT

    return None


# ---------------------------------------------------------------------------
# 5. Gemini zero-shot classifier (async)
# ---------------------------------------------------------------------------


async def _classify_with_gemini(text: str) -> ClaimType:
    """
    Classify the text using the Gemini API (zero-shot).

    Sends a single generateContent request with a strict classification
    prompt and parses the model's response into a ClaimType.

    Args:
        text: The claim text to classify.

    Returns:
        ClaimType based on Gemini's response.
        Returns AMBIGUOUS if the response text cannot be mapped to a
        known ClaimType (e.g., the model hedges or replies unexpectedly).

    Raises:
        ClassifierConfigError:  GEMINI_API_KEY is empty.
        ClassifierTimeoutError: Request timed out.
        ClassifierServiceError: HTTP error or connection failure.
    """
    api_key = settings.GEMINI_API_KEY.get_secret_value()
    if not api_key:
        raise ClassifierConfigError("GEMINI_API_KEY is not configured")

    payload = {
        "contents": [{"parts": [{"text": _CLASSIFIER_PROMPT.format(text=text)}]}],
        "generationConfig": {
            "temperature": 0.0,
            # Increased from 20 to 100 because the model might spend tokens on implicit
            # chain-of-thought ("thoughts") before emitting the actual label.
            "maxOutputTokens": 100,
            "candidateCount": 1,
            "stopSequences": ["\n"],
        },
    }

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(settings.CLASSIFIER_TIMEOUT_SECONDS)
        ) as client:
            response = await client.post(
                GEMINI_CLASSIFY_URL,
                json=payload,
                # API key as a query parameter — standard for Gemini REST API.
                # Never include the key in log messages.
                params={"key": api_key},
            )
            if response.status_code == 429:
                raise ClassifierQuotaError("Gemini classification quota exceeded (HTTP 429)")
            response.raise_for_status()
            data = response.json()

    except httpx.TimeoutException as exc:
        raise ClassifierTimeoutError("Gemini API request timed out") from exc
    except httpx.HTTPStatusError as exc:
        raise ClassifierServiceError(
            f"Gemini API returned HTTP {exc.response.status_code}"
        ) from exc
    except httpx.RequestError as exc:
        raise ClassifierServiceError(
            f"Gemini API connection failed: {type(exc).__name__}"
        ) from exc

    # --- Defensive response parsing ---
    # Extract the generated text from the nested Gemini response structure.
    try:
        raw_label: str = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError, AttributeError):
        # Unexpected response structure — treat as uncertain.
        logger.warning(
            "Gemini classification: unexpected response structure — returning AMBIGUOUS"
        )
        return ClaimType.AMBIGUOUS

    # Normalize: strip whitespace, collapse to lowercase, spaces → underscores.
    # This maps both "FACTUAL_CLAIM" and "factual claim" to "factual_claim".
    normalized_label = re.sub(r"\s+", "_", raw_label.strip().lower())
    claim_type = _LABEL_TO_CLAIM_TYPE.get(normalized_label)

    if claim_type is None:
        logger.warning(
            "Gemini classification: unrecognized label %r — returning AMBIGUOUS",
            raw_label[:50],  # truncate to avoid leaking unexpected content
        )
        return ClaimType.AMBIGUOUS

    logger.debug("Gemini classification: %s", claim_type.value)
    return claim_type


# ---------------------------------------------------------------------------
# 6. Public orchestrator (async)
# ---------------------------------------------------------------------------


async def classify_claim(text: str) -> ClaimType:
    """
    Classify the input text as a factual claim, opinion, advertisement,
    or ambiguous using a hybrid rule-based + Gemini approach.

    Pipeline:
        1. Run conservative rule-based heuristics.
           If a confident classification is found, return immediately
           (zero network calls, deterministic).

        2. If rules return None, call the Gemini zero-shot classifier.
           If Gemini returns a valid label, return it.

        3. If Gemini fails for any reason (no key, timeout, HTTP error,
           connection refused), log a WARNING and return FACTUAL_CLAIM.
           The fact-checking pipeline will still run.

    This function NEVER raises an exception.  All errors are handled
    internally to ensure classifier failure does not propagate to callers.

    Args:
        text: The raw user-submitted claim text.

    Returns:
        ClaimType indicating how the verification pipeline should proceed.
    """
    stripped = text.strip()

    # Layer 1: rule-based (sync, fast, zero I/O)
    rule_result = _classify_by_rules(stripped)
    if rule_result is not None:
        logger.info("Claim classified by rules as %s", rule_result.value)
        return rule_result

    # Layer 2: Gemini (async, fallback)
    try:
        gemini_result = await _classify_with_gemini(stripped)
        logger.info("Claim classified by Gemini as %s", gemini_result.value)
        return gemini_result

    except ClassifierQuotaError:
        raise
    except ClassifierConfigError:
        logger.warning(
            "Gemini classifier: API key not configured — "
            "treating claim as FACTUAL_CLAIM"
        )
    except ClassifierTimeoutError:
        logger.warning(
            "Gemini classifier: request timed out — "
            "treating claim as FACTUAL_CLAIM"
        )
    except ClassifierServiceError as exc:
        logger.warning(
            "Gemini classifier: service error (%s) — "
            "treating claim as FACTUAL_CLAIM",
            type(exc).__name__,
        )
    except Exception as exc:  # noqa: BLE001 — safety net, must not propagate
        logger.error(
            "Gemini classifier: unexpected error (%s) — "
            "treating claim as FACTUAL_CLAIM",
            type(exc).__name__,
        )

    return ClaimType.FACTUAL_CLAIM
