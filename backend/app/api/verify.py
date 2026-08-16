"""
api/verify.py — Route handler for POST /api/verify.

Phase 3 behavior
-----------------
A hybrid claim classifier runs BEFORE the Google Fact Check lookup.
OPINION and ADVERTISEMENT claims are short-circuited immediately and
return the unverifiable placeholder without calling any external API.
FACTUAL_CLAIM and AMBIGUOUS both proceed to the Google Fact Check layer.
AMBIGUOUS claims have their final confidence score reduced by × 0.7.

Phase 2 behavior (preserved)
------------------------------
Input is validated, then the Google Fact Check API is queried via the
factcheck service.  The normalized result is returned as a VerifyResponse
if a relevant fact-check is found.  If not (or if the API key is absent),
the route falls through to the placeholder response.

Error mapping (factcheck layer — unchanged from Phase 2)
---------------------------------------------------------
    FactCheckConfigError      → log error, fall through to placeholder (200)
    FactCheckAuthError        → 502 Bad Gateway
    FactCheckQuotaError       → 503 Service Unavailable
    FactCheckTimeoutError     → 503 Service Unavailable
    FactCheckServiceError     → 503 Service Unavailable

Classifier error handling (Phase 3)
-------------------------------------
    classify_claim() NEVER raises — all exceptions are caught internally.
    Classifier failure → treated as FACTUAL_CLAIM → pipeline continues.

ClaimType is NOT exposed in VerifyResponse.  The public API schema is unchanged.

Full request flow (Phase 3)
----------------------------
    POST /api/verify
      │
      ├─ Pydantic validates VerifyRequest              → 422 on failure
      ├─ Route-level length check                      → 422 on failure
      │
      ├─ classify_claim(claim_text)                    [Phase 3]
      │     ├─ OPINION / ADVERTISEMENT → return placeholder (200, no factcheck)
      │     ├─ FACTUAL_CLAIM           → proceed
      │     └─ AMBIGUOUS               → proceed (confidence × 0.7 applied later)
      │     (classifier never raises; failure → FACTUAL_CLAIM)
      │
      ├─ verify_claim_factcheck(claim_text)            [Phase 2]
      │     ├─ FactCheckConfigError  → log + fall through to placeholder
      │     ├─ FactCheckAuthError    → 502
      │     ├─ FactCheckQuotaError   → 503
      │     ├─ Timeout / Service err → 503
      │     ├─ match found           → build VerifyResponse
      │     │     └─ AMBIGUOUS: confidence × 0.7 applied here
      │     └─ None returned         → fall through to placeholder
      │
      └─ Placeholder: VerifyResponse(verdict="unverifiable", confidence_score=0.0, sources=[])
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi import status as http_status

from app.core.config import settings
from app.core.logging import get_logger
from app.models.verification import VerifyRequest, VerifyResponse
from app.services.classifier import ClaimType, classify_claim
from app.services.factcheck import (
    FactCheckAuthError,
    FactCheckConfigError,
    FactCheckQuotaError,
    FactCheckServiceError,
    FactCheckTimeoutError,
    verify_claim_factcheck,
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
        "A hybrid classifier pre-filters opinions and advertisements before "
        "querying the Google Fact Check Tools API. "
        "Returns a placeholder response when no fact-check is available."
    ),
    responses={
        200: {"description": "Verification result"},
        422: {"description": "Invalid or malformed request body"},
        502: {"description": "Upstream authentication failure"},
        503: {"description": "Upstream service temporarily unavailable"},
        500: {"description": "Internal server error"},
    },
)
async def verify_claim(request: VerifyRequest) -> VerifyResponse:
    """
    Verify a factual claim submitted by the Chrome Extension.

    Validates the input, classifies the claim type, then queries the Google
    Fact Check API for matching fact-checks.  All results are passed through
    the VerifyResponse Pydantic model before being returned to guarantee
    schema correctness.

    Args:
        request: Parsed and validated VerifyRequest body.

    Returns:
        VerifyResponse with verdict, confidence_score, and sources.

    Raises:
        HTTPException(422): claim text exceeds MAX_CLAIM_LENGTH.
        HTTPException(502): upstream authentication failure.
        HTTPException(503): upstream service unavailable or quota exceeded.
    """
    claim_text = request.text.strip()

    # --- Input length guard ---
    # Lives here (not in the Pydantic validator) so MAX_CLAIM_LENGTH is
    # configurable via settings without touching schema code.
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
    claim_type = await classify_claim(claim_text)
    logger.info("Claim type: %s (length=%d)", claim_type.value, len(claim_text))

    # OPINION and ADVERTISEMENT: early exit — no fact-check call.
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

    # FACTUAL_CLAIM and AMBIGUOUS both proceed to the fact-check layer.
    # For AMBIGUOUS, the confidence score will be reduced after the lookup.

    # --- Phase 2: Google Fact Check lookup ---
    match = None
    try:
        match = await verify_claim_factcheck(claim_text)

    except FactCheckConfigError:
        # API key not configured: fall through to placeholder.
        # The server stays functional during development without a key.
        logger.error(
            "Google Fact Check API key not configured — "
            "falling through to placeholder response"
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

    # --- Fact-check found: build VerifyResponse through the Pydantic model ---
    if match is not None:
        # Apply AMBIGUOUS confidence reduction in this layer, not in the
        # factcheck service, to keep normalization logic uncontaminated.
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

    # --- No match / not configured: placeholder ---
    # TODO (Phase 4): add LLM/search fallback before this placeholder.
    logger.info(
        "No fact-check found for claim (length=%d) — returning placeholder",
        len(claim_text),
    )
    return VerifyResponse(
        verdict="unverifiable",
        confidence_score=0.0,
        sources=[],
    )
