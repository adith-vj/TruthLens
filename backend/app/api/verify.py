"""
api/verify.py — Route handler for POST /api/verify.

Phase 4 behavior
-----------------
When verify_claim_factcheck() returns None (no existing fact-check found),
the pipeline falls through to the web-evidence fallback:

    1. search_evidence(claim)      — Tavily AI web search (search.py)
    2. verify_with_llm(claim, ev)  — Gemini verifier (llm.py)
    3. Build VerifyResponse.sources from LLMVerdict.source_indices

Empty evidence early exit:
    If search_evidence() returns [] (no usable Tavily results), the route
    returns 200 unverifiable immediately WITHOUT calling verify_with_llm().
    The LLM is never called with an empty evidence list.

Phase 4 failures return 200 unverifiable (not 502/503):
    Search and LLM failures are best-effort — they must not degrade the
    user experience beyond the existing 'no fact-check found' state.

Source provenance (hard invariant):
    VerifyResponse.sources is built exclusively from the original SearchResult
    objects returned by search_evidence(), using the integer indices from
    LLMVerdict.source_indices as pointers.  The LLM never supplies URL, title,
    or publisher text.  Out-of-range or duplicate indices are silently discarded.

Phase 3 behavior (preserved)
------------------------------
A hybrid claim classifier runs BEFORE the Google Fact Check lookup.
OPINION and ADVERTISEMENT claims are short-circuited immediately and
return the unverifiable placeholder without calling any external API.
FACTUAL_CLAIM and AMBIGUOUS both proceed to the Google Fact Check layer.
AMBIGUOUS claims have their final confidence score reduced by × 0.7 when a
fact-check match is found.

Phase 2 behavior (preserved)
------------------------------
Input is validated, then the Google Fact Check API is queried via the
factcheck service.  The normalized result is returned as a VerifyResponse
if a relevant fact-check is found.

Full request flow (Phase 4)
-----------------------------
    POST /api/verify
      │
      ├─ Pydantic validates VerifyRequest              → 422 on failure
      ├─ Route-level length check                      → 422 on failure
      │
      ├─ classify_claim(claim_text)                    [Phase 3]
      │     OPINION / ADVERTISEMENT → 200 unverifiable (exit, no factcheck)
      │     FACTUAL_CLAIM / AMBIGUOUS → continue
      │     (classifier never raises; failure → FACTUAL_CLAIM)
      │
      ├─ verify_claim_factcheck(claim_text)            [Phase 2]
      │     FactCheckConfigError  → log + fall through (no key)
      │     FactCheckAuthError    → 502
      │     FactCheckQuotaError   → 503
      │     Timeout / Service err → 503
      │     match found           → 200 result (AMBIGUOUS: conf × 0.7)
      │     None                  → fall through to Phase 4
      │
      ├─ search_evidence(claim_text)                   [Phase 4]
      │     SearchConfigError     → log + 200 unverifiable
      │     SearchQuotaError      → log + 200 unverifiable
      │     SearchTimeout/Service → log + 200 unverifiable
      │     [] (empty)            → 200 unverifiable (LLM NOT called)
      │
      ├─ verify_with_llm(claim_text, evidence)         [Phase 4]
      │     LLMConfigError        → log + 200 unverifiable
      │     LLMQuotaError         → log + 200 unverifiable
      │     LLMTimeout/Service    → log + 200 unverifiable
      │     LLMParseError         → log + 200 unverifiable
      │     LLMVerdict            → build VerifyResponse from source indices
      │
      └─ 200 unverifiable (final fallback, should be unreachable)

Error mapping summary
----------------------
Phase 2 errors:
    FactCheckConfigError      → log error, fall through to Phase 4
    FactCheckAuthError        → 502 Bad Gateway
    FactCheckQuotaError       → 503 Service Unavailable
    FactCheckTimeoutError     → 503 Service Unavailable
    FactCheckServiceError     → 503 Service Unavailable

Phase 4 errors (all → 200 unverifiable, never 5xx):
    SearchConfigError         → log warning, return 200 unverifiable
    SearchQuotaError          → log warning, return 200 unverifiable
    SearchTimeoutError        → log warning, return 200 unverifiable
    SearchServiceError        → log error, return 200 unverifiable
    LLMConfigError            → log warning, return 200 unverifiable
    LLMQuotaError             → log warning, return 200 unverifiable
    LLMTimeoutError           → log warning, return 200 unverifiable
    LLMServiceError           → log error, return 200 unverifiable
    LLMParseError             → log error, return 200 unverifiable
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi import status as http_status

from app.core.config import settings
from app.core.logging import get_logger
from app.models.verification import SourceItem, VerifyRequest, VerifyResponse
from app.services.classifier import ClaimType, classify_claim, ClassifierQuotaError
from app.services.factcheck import (
    FactCheckAuthError,
    FactCheckConfigError,
    FactCheckQuotaError,
    FactCheckServiceError,
    FactCheckTimeoutError,
    verify_claim_factcheck,
)
from app.services.llm import (
    LLMConfigError,
    LLMError,
    LLMParseError,
    LLMQuotaError,
    LLMServiceError,
    LLMTimeoutError,
    verify_with_llm,
)
from app.services.search import (
    SearchConfigError,
    SearchError,
    SearchQuotaError,
    SearchServiceError,
    SearchTimeoutError,
    search_evidence,
)

logger = get_logger(__name__)

router = APIRouter()

# Confidence multiplier applied to AMBIGUOUS claims when a fact-check result
# is found.  Applied here (in the route layer) rather than in the factcheck
# service to keep classification concerns out of the normalization layer.
_AMBIGUOUS_CONFIDENCE_FACTOR = 0.7


@router.post(
    "/verify",
    response_model=VerifyResponse,
    summary="Verify a factual claim",
    description=(
        "Accepts a text claim selected by the user and returns a structured "
        "verdict with a confidence score and supporting sources. "
        "A hybrid classifier pre-filters opinions and advertisements. "
        "When no existing fact-check is found, a Tavily web search and "
        "Gemini LLM provide web-evidence grounded verification. "
        "Returns a placeholder response when no evidence is available."
    ),
    responses={
        200: {"description": "Verification result"},
        422: {"description": "Invalid or malformed request body"},
        502: {"description": "Upstream authentication failure (fact-check layer only)"},
        503: {"description": "Upstream service temporarily unavailable (fact-check layer only)"},
        500: {"description": "Internal server error"},
    },
)
async def verify_claim(request: VerifyRequest) -> VerifyResponse:
    """
    Verify a factual claim submitted by the Chrome Extension.

    Validates input, classifies claim type, queries fact-check evidence, then
    falls back to web-search + LLM verification if no fact-check is found.

    Args:
        request: Parsed and validated VerifyRequest body.

    Returns:
        VerifyResponse with verdict, confidence_score, and sources.

    Raises:
        HTTPException(422): claim text exceeds MAX_CLAIM_LENGTH.
        HTTPException(502): fact-check upstream authentication failure.
        HTTPException(503): fact-check upstream service unavailable or quota exceeded.
    """
    claim_text = request.text.strip()

    # --- Input length guard ---
    if len(claim_text) > settings.MAX_CLAIM_LENGTH:
        logger.warning(
            "Claim rejected: text too long (length=%d, max=%d)",
            len(claim_text),
            settings.MAX_CLAIM_LENGTH,
        )
        raise HTTPException(
            status_code=http_status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"text exceeds maximum allowed length of "
                f"{settings.MAX_CLAIM_LENGTH} characters"
            ),
        )

    # --- Phase 3: Classify claim type ---
    # classify_claim() never raises; all failures fall back to FACTUAL_CLAIM.
    try:
        claim_type = await classify_claim(claim_text)
    except ClassifierQuotaError:
        logger.warning("Gemini classifier quota exceeded - returning unverifiable")
        return _unverifiable()
    logger.info("Claim type: %s (length=%d)", claim_type.value, len(claim_text))

    # OPINION and ADVERTISEMENT: early exit — no external calls.
    if claim_type in (ClaimType.OPINION, ClaimType.ADVERTISEMENT):
        logger.info(
            "Early exit: claim classified as %s — skipping fact-check",
            claim_type.value,
        )
        return VerifyResponse(
            verdict="unverifiable",
            confidence_score=0.0,
            sources=[],
        )

    # FACTUAL_CLAIM and AMBIGUOUS proceed to the fact-check layer.

    # --- Phase 2: Google Fact Check lookup ---
    match = None
    try:
        match = await verify_claim_factcheck(claim_text)

    except FactCheckConfigError:
        # API key not configured: fall through to Phase 4.
        logger.error(
            "Google Fact Check API key not configured — "
            "falling through to Phase 4 (web search + LLM)"
        )

    except FactCheckAuthError:
        logger.error("Google Fact Check API authentication failure")
        raise HTTPException(
            status_code=http_status.HTTP_502_BAD_GATEWAY,
            detail="upstream service error",
        )

    except FactCheckQuotaError:
        logger.warning("Google Fact Check API quota exceeded")
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="upstream service temporarily unavailable",
        )

    except (FactCheckTimeoutError, FactCheckServiceError) as exc:
        logger.error(
            "Google Fact Check API unavailable: %s", type(exc).__name__
        )
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="upstream service temporarily unavailable",
        )

    # --- Fact-check match found: build VerifyResponse ---
    if match is not None:
        confidence = match.confidence_score
        if claim_type == ClaimType.AMBIGUOUS:
            adjusted = max(0.0, min(1.0, confidence * _AMBIGUOUS_CONFIDENCE_FACTOR))
            logger.info(
                "AMBIGUOUS claim: confidence adjusted from %.2f to %.2f (× %.1f)",
                confidence,
                adjusted,
                _AMBIGUOUS_CONFIDENCE_FACTOR,
            )
            confidence = adjusted

        logger.info(
            "Fact-check result: verdict=%s confidence=%.2f sources=%d",
            match.verdict,
            confidence,
            len(match.sources),
        )
        return VerifyResponse(
            verdict=match.verdict,
            confidence_score=confidence,
            sources=match.sources,
        )

    # --- Phase 4: Web search + LLM fallback ---
    # Only reached when the fact-check layer returned None or had no key.
    logger.info("No fact-check found — invoking Phase 4 (web search + LLM fallback)")

    # --- Phase 4a: Tavily web search ---
    evidence = []
    try:
        evidence = await search_evidence(claim_text)
        logger.info("Tavily returned %d evidence items", len(evidence))

    except SearchConfigError:
        logger.warning(
            "TAVILY_API_KEY not configured — Phase 4 unavailable, returning unverifiable"
        )
        return _unverifiable()

    except SearchQuotaError:
        logger.warning(
            "Tavily API quota exceeded — returning unverifiable"
        )
        return _unverifiable()

    except SearchTimeoutError:
        logger.warning("Tavily search timed out — returning unverifiable")
        return _unverifiable()

    except SearchServiceError:
        logger.error("Tavily search failed — returning unverifiable")
        return _unverifiable()

    # Empty evidence: return unverifiable without calling the LLM.
    if not evidence:
        logger.info("Tavily returned no usable evidence — returning unverifiable")
        return _unverifiable()

    # --- Phase 4b: Gemini LLM verification ---
    try:
        llm_verdict = await verify_with_llm(claim_text, evidence)

    except LLMConfigError:
        logger.warning(
            "GEMINI_API_KEY not configured — LLM verifier unavailable, returning unverifiable"
        )
        return _unverifiable()

    except LLMQuotaError:
        logger.warning("Gemini LLM quota exceeded — returning unverifiable")
        return _unverifiable()

    except LLMTimeoutError:
        logger.warning("Gemini LLM verifier timed out — returning unverifiable")
        return _unverifiable()

    except (LLMServiceError, LLMParseError) as exc:
        logger.error(
            "Gemini LLM verifier failed (%s) — returning unverifiable",
            type(exc).__name__,
        )
        return _unverifiable()

    # --- Build VerifyResponse from LLMVerdict ---
    # Source provenance: the backend owns all URL/title/publisher metadata.
    # The LLM's source_indices are used only as pointers into the evidence list.
    seen_urls: set[str] = set()
    sources: list[SourceItem] = []
    for idx in llm_verdict.source_indices:
        if not (0 <= idx < len(evidence)):
            logger.warning(
                "LLM returned out-of-range source index %d (evidence length=%d) — skipping",
                idx,
                len(evidence),
            )
            continue
        result = evidence[idx]
        url_str = str(result.url)
        if url_str in seen_urls:
            continue
        seen_urls.add(url_str)
        sources.append(
            SourceItem(
                title=result.title,
                url=result.url,
                publisher=result.publisher or "",
            )
        )

    logger.info(
        "LLM fallback result: verdict=%s confidence=%.2f sources=%d",
        llm_verdict.verdict,
        llm_verdict.confidence_score,
        len(sources),
    )
    return VerifyResponse(
        verdict=llm_verdict.verdict,
        confidence_score=llm_verdict.confidence_score,
        sources=sources,
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _unverifiable() -> VerifyResponse:
    """Return the standard unverifiable placeholder response."""
    return VerifyResponse(
        verdict="unverifiable",
        confidence_score=0.0,
        sources=[],
    )
