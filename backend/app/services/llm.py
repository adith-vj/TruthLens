"""
services/llm.py — Gemini LLM verification service (Phase 4 fallback).

Architecture
------------
This module is organised in five layers:

    1. LLMVerdict   — Internal DTO returned to the route handler.
    2. Exceptions   — Custom hierarchy covering every Gemini failure mode.
    3. Constants    — Gemini endpoint URL (exported for test mocking).
    4. Helpers      — Pure sync functions for prompt building and response
                      parsing.  Zero knowledge of HTTP or settings.
    5. Orchestrator — Public async entry point called by the route handler.

Non-fabrication guarantee (hard requirement)
---------------------------------------------
The LLM never supplies URL, title, publisher, or other source metadata.
It returns only integer indices into the evidence list supplied by the caller.
The route handler builds VerifyResponse.sources by indexing into the original
SearchResult objects using those indices.

Even a hallucinating model cannot inject a fake URL into the response through
this design — invalid indices are silently discarded by the route.

Source of truth: the evidence list supplied by search_evidence() is the ONLY
permissible source of URL/title/publisher data in the final response.

LLM context: title + snippet per evidence item; URLs are NEVER in the prompt.
Invalid or out-of-range indices are not an LLM error — they are discarded by
the route.

Fail-safe rule (enforced by both the prompt and the parse layer)
-----------------------------------------------------------------
Any of the following conditions must produce verdict = "unverifiable":
    - Evidence is insufficient or absent.
    - Evidence directly contradicts itself.
    - Evidence discusses a related but different claim.
    - The model cannot reach a supportable conclusion.

Confidence validation
---------------------
confidence_score must be a float in [0.0, 1.0].  A value outside this range
or of the wrong type is treated as a parse error (LLMParseError) — it is NOT
silently clamped.  Rationale: a model returning confidence_score=5.0 or
confidence_score="high" is not reasoning correctly; returning unverifiable is
safer than masking the model's confusion.

Model
-----
gemini-2.5-flash (GA, free tier confirmed August 2026).
Uses JSON-mode output (responseMimeType: "application/json") to enforce
structured responses.

Separation of concerns
-----------------------
This service does NOT call Tavily or any other search provider.
The evidence list must be supplied by the caller (api/verify.py, via
services/search.py).

API key
-------
Reuses GEMINI_API_KEY from settings.  Check for empty key before any request.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal

import httpx

from app.core.config import settings
from app.core.logging import get_logger
from app.models.verification import VerdictType
from app.services.search import SearchResult

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 1. Internal DTO returned by verify_with_llm()
# ---------------------------------------------------------------------------


@dataclass
class LLMVerdict:
    """
    Internal data transfer object produced by verify_with_llm().

    This is NOT the public API response.  The route handler converts it to
    a VerifyResponse by resolving source_indices against the evidence list.

    Fields:
        verdict:         One of the four VerdictType literals.
        confidence_score: Float in [0.0, 1.0].  Validated by _parse_llm_response();
                          a value outside this range raises LLMParseError.
        source_indices:  List of integer indices into the evidence list passed
                         to verify_with_llm().  May be empty.  Out-of-range
                         or duplicate indices are silently discarded by the
                         route handler — not by this service.
    """

    verdict: VerdictType
    confidence_score: float
    source_indices: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 2. Exception hierarchy
# ---------------------------------------------------------------------------


class LLMError(Exception):
    """Base class for all LLM verifier service errors."""


class LLMConfigError(LLMError):
    """
    Raised before any network request when GEMINI_API_KEY is missing or empty.
    The route catches this and returns 200 unverifiable.
    """


class LLMTimeoutError(LLMError):
    """Raised when the Gemini API request exceeds LLM_TIMEOUT_SECONDS."""


class LLMQuotaError(LLMError):
    """Raised when Gemini returns HTTP 429 (quota or rate limit exceeded)."""


class LLMServiceError(LLMError):
    """
    Raised on HTTP errors (4xx other than 429, 5xx) or connection failures.
    The route catches this and returns 200 unverifiable.
    """


class LLMParseError(LLMError):
    """
    Raised when the Gemini response cannot be parsed into a valid LLMVerdict.

    Conditions that raise LLMParseError:
      - Response JSON cannot be decoded.
      - Gemini returned no candidates or empty content.
      - Inner JSON (the structured output) cannot be decoded.
      - 'verdict' field is missing.
      - 'verdict' value is not one of the four VerdictType literals.
      - 'confidence_score' field is missing.
      - 'confidence_score' is not a numeric type.
      - 'confidence_score' is outside [0.0, 1.0].
    """


# ---------------------------------------------------------------------------
# 3. Constants
# ---------------------------------------------------------------------------

# Exported so tests can import for respx URL matching.
# Model: gemini-2.5-flash (GA, free tier confirmed August 2026).
# Separate from GEMINI_CLASSIFY_URL in classifier.py (uses gemini-1.5-flash-latest).
GEMINI_VERIFY_URL = (
    "https://generativelanguage.googleapis.com"
    "/v1beta/models/gemini-2.5-flash:generateContent"
)

# The four valid verdict values (matches VerdictType in models/verification.py).
_VALID_VERDICTS: frozenset[str] = frozenset(
    {"true", "false", "misleading", "unverifiable"}
)

# ---------------------------------------------------------------------------
# 4. Helpers — pure sync functions
# ---------------------------------------------------------------------------


def _build_prompt(claim: str, evidence: list[SearchResult]) -> str:
    """
    Build the structured fact-checking prompt for the Gemini verifier.

    Includes title and snippet for each evidence item.
    URLs are NEVER included in the prompt — the LLM must not see them, as
    they create a surface where the model could repeat or fabricate URLs
    in its output, bypassing the index-only response format.

    Args:
        claim:    The verified, non-empty claim text.
        evidence: A non-empty list of SearchResult objects.

    Returns:
        A formatted prompt string ready for the Gemini generateContent API.
    """
    evidence_block = ""
    for i, result in enumerate(evidence):
        evidence_block += (
            f"[{i}]\n"
            f"Title: {result.title}\n"
            f"Snippet: {result.snippet}\n\n"
        )

    return f"""\
You are a fact-checking assistant. Your task is to assess the following claim \
using ONLY the provided evidence. Do not use your background knowledge.

Claim: "{claim}"

Evidence:
{evidence_block.rstrip()}

Based ONLY on the evidence above, respond with a JSON object:
{{
  "verdict": "<one of: true, false, misleading, unverifiable>",
  "confidence_score": <float between 0.0 and 1.0>,
  "source_indices": [<integer indices from the evidence list above>]
}}

Rules:
1. verdict must be exactly one of: "true", "false", "misleading", "unverifiable".
2. confidence_score must be a number between 0.0 and 1.0. Use lower values when \
evidence is partial, mixed, or tangentially related.
3. source_indices must be a list of integer indices (e.g. [0, 2]) referencing \
evidence items that directly support your verdict. Use an empty list [] if none apply.
4. Use "unverifiable" if:
   - Evidence is insufficient, absent, or does not directly address the claim.
   - Evidence contradicts itself without a clear consensus.
   - Evidence discusses a related but different claim.
   - You cannot reach a supportable conclusion from the provided snippets.
5. Never invent facts, quotes, statistics, or references not present in the \
evidence snippets above.
6. Return ONLY valid JSON. No explanation outside the JSON object.\
"""


def _parse_llm_response(raw_json: dict) -> LLMVerdict:
    """
    Parse and validate a Gemini generateContent response into an LLMVerdict.

    Args:
        raw_json: The parsed JSON response body from Gemini.

    Returns:
        A validated LLMVerdict.

    Raises:
        LLMParseError: If the response is structurally invalid, the inner JSON
                       cannot be decoded, or any field fails validation.
    """
    # --- Extract text from Gemini response envelope ---
    try:
        candidates = raw_json.get("candidates", [])
        if not candidates:
            raise LLMParseError(
                "Gemini response contained no candidates"
            )
        text = candidates[0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMParseError(
            f"Unexpected Gemini response structure: {exc}"
        ) from exc

    # --- Parse the inner JSON object ---
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMParseError(
            f"Gemini returned non-JSON text: {text[:200]!r}"
        ) from exc

    if not isinstance(parsed, dict):
        raise LLMParseError(
            f"Gemini returned JSON but not an object: {type(parsed).__name__}"
        )

    # --- Validate 'verdict' ---
    verdict_raw = parsed.get("verdict")
    if verdict_raw is None:
        raise LLMParseError("Gemini response missing 'verdict' field")
    if not isinstance(verdict_raw, str) or verdict_raw not in _VALID_VERDICTS:
        raise LLMParseError(
            f"Gemini returned invalid verdict {verdict_raw!r}; "
            f"must be one of {sorted(_VALID_VERDICTS)}"
        )
    verdict: VerdictType = verdict_raw  # type: ignore[assignment]

    # --- Validate 'confidence_score' ---
    confidence_raw = parsed.get("confidence_score")
    if confidence_raw is None:
        raise LLMParseError("Gemini response missing 'confidence_score' field")
    if not isinstance(confidence_raw, (int, float)):
        raise LLMParseError(
            f"'confidence_score' must be a number; got {type(confidence_raw).__name__}: "
            f"{confidence_raw!r}"
        )
    confidence = float(confidence_raw)
    if not (0.0 <= confidence <= 1.0):
        raise LLMParseError(
            f"'confidence_score' must be in [0.0, 1.0]; got {confidence}"
        )

    # --- Validate 'source_indices' ---
    # Invalid, non-integer, or out-of-range indices are discarded here at the
    # service level only for type safety.  Bounds checking (against the actual
    # evidence list length) is the route handler's responsibility.
    raw_indices = parsed.get("source_indices", [])
    if not isinstance(raw_indices, list):
        logger.warning(
            "Gemini 'source_indices' is not a list (%s) — treating as empty",
            type(raw_indices).__name__,
        )
        raw_indices = []

    source_indices: list[int] = []
    for item in raw_indices:
        if isinstance(item, int):
            source_indices.append(item)
        elif isinstance(item, float) and item == int(item):
            # JSON doesn't distinguish int from float; accept whole-number floats.
            source_indices.append(int(item))
        else:
            logger.warning(
                "Gemini returned non-integer source index %r — discarding", item
            )

    return LLMVerdict(
        verdict=verdict,
        confidence_score=confidence,
        source_indices=source_indices,
    )


# ---------------------------------------------------------------------------
# 5. Orchestrator — public async entry point
# ---------------------------------------------------------------------------


async def verify_with_llm(
    claim: str,
    evidence: list[SearchResult],
) -> LLMVerdict:
    """
    Verify a claim against the supplied evidence using the Gemini LLM.

    The LLM receives the claim and a numbered list of evidence items (title +
    snippet only — URLs are excluded from the prompt).  It returns a verdict,
    a confidence score, and integer indices identifying which evidence items
    support its conclusion.

    The route handler uses those indices to build VerifyResponse.sources from
    the original SearchResult objects.  The LLM never supplies URL, title, or
    publisher text directly.

    Pre-condition: evidence must be non-empty.  The route handler is responsible
    for checking this before calling this function.

    Args:
        claim:    The verified, non-empty claim text.
        evidence: A non-empty list of SearchResult objects from search_evidence().

    Returns:
        An LLMVerdict with a validated verdict, confidence_score, and
        source_indices.

    Raises:
        LLMConfigError:  GEMINI_API_KEY is not configured.
        LLMTimeoutError: The HTTP request exceeded LLM_TIMEOUT_SECONDS.
        LLMQuotaError:   Gemini returned HTTP 429.
        LLMServiceError: Any other HTTP or connection error.
        LLMParseError:   The Gemini response could not be parsed into a valid
                         LLMVerdict (invalid JSON, wrong verdict, bad confidence).
    """
    api_key = settings.GEMINI_API_KEY.get_secret_value()
    if not api_key:
        raise LLMConfigError(
            "GEMINI_API_KEY is not configured — LLM verifier unavailable"
        )

    prompt = _build_prompt(claim, evidence)

    request_body = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            # Force structured JSON output from the model.
            "responseMimeType": "application/json",
        },
    }

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(settings.LLM_TIMEOUT_SECONDS)
        ) as client:
            response = await client.post(
                GEMINI_VERIFY_URL,
                json=request_body,
                params={"key": api_key},
            )

    except httpx.TimeoutException as exc:
        logger.warning(
            "Gemini verifier timed out after %.1fs: %s",
            settings.LLM_TIMEOUT_SECONDS,
            exc,
        )
        raise LLMTimeoutError(str(exc)) from exc

    except httpx.ConnectError as exc:
        logger.error("Gemini verifier connection error: %s", exc)
        raise LLMServiceError(str(exc)) from exc

    except httpx.RequestError as exc:
        logger.error("Gemini verifier request error: %s", exc)
        raise LLMServiceError(str(exc)) from exc

    # --- HTTP status handling ---
    if response.status_code == 429:
        logger.warning("Gemini verifier quota exceeded (HTTP 429)")
        raise LLMQuotaError("Gemini API quota exceeded (HTTP 429)")

    if response.status_code >= 400:
        logger.error(
            "Gemini verifier returned HTTP %d: %s",
            response.status_code,
            response.text[:200],
        )
        raise LLMServiceError(
            f"Gemini API error (HTTP {response.status_code})"
        )

    # --- Parse response ---
    try:
        raw_json = response.json()
    except Exception as exc:
        raise LLMParseError(
            "Gemini verifier response was not valid JSON"
        ) from exc

    verdict = _parse_llm_response(raw_json)

    logger.info(
        "LLM verdict: %s (confidence=%.2f, source_indices=%s)",
        verdict.verdict,
        verdict.confidence_score,
        verdict.source_indices,
    )
    return verdict
