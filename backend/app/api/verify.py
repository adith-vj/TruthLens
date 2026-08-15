"""
api/verify.py — Route handler for POST /api/verify.

Phase 2 behavior
-----------------
Input is validated, then the Google Fact Check API is queried via the
factcheck service.  The normalized result is returned as a VerifyResponse
if a relevant fact-check is found.  If not (or if the API key is absent),
the route falls through to the Phase 1 placeholder response.

All normalized results are explicitly constructed through VerifyResponse
(the public Pydantic model) before being returned, ensuring final schema
validation on the way out regardless of what the service layer produced.

Error mapping
-------------
    FactCheckConfigError      → log error, fall through to placeholder (200)
    FactCheckAuthError        → 502 Bad Gateway
    FactCheckQuotaError       → 503 Service Unavailable
    FactCheckTimeoutError     → 503 Service Unavailable
    FactCheckServiceError     → 503 Service Unavailable

Client-facing error detail strings are static and safe — no raw upstream
error text, no API keys, and no internal stack detail is ever forwarded.

Request flow (Phase 2)
-----------------------
    POST /api/verify
      │
      ├─ Pydantic validates VerifyRequest    → 422 on failure
      ├─ Route-level length check            → 422 on failure
      │
      ├─ verify_claim_factcheck(claim_text)
      │     ├─ FactCheckConfigError  → log + fall through to placeholder
      │     ├─ FactCheckAuthError    → 502
      │     ├─ FactCheckQuotaError   → 503
      │     ├─ Timeout / Service err → 503
      │     ├─ match found           → return VerifyResponse (through Pydantic)
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


@router.post(
    "/verify",
    response_model=VerifyResponse,
    summary="Verify a factual claim",
    description=(
        "Accepts a text claim selected by the user and returns a structured "
        "verdict with a confidence score and supporting sources. "
        "Queries the Google Fact Check Tools API for existing fact-checks "
        "and normalizes the result into the standard VerifyResponse schema. "
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

    Validates the input, queries the Google Fact Check API, and returns a
    VerifyResponse.  All fact-check results are passed through the VerifyResponse
    Pydantic model before being returned to guarantee schema correctness.

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
    # Explicitly constructing VerifyResponse (not passing FactCheckMatch directly)
    # ensures the public schema is validated on every response path.
    if match is not None:
        logger.info(
            "Fact-check result: verdict=%s confidence=%.2f sources=%d",
            match.verdict,
            match.confidence_score,
            len(match.sources),
        )
        return VerifyResponse(
            verdict=match.verdict,
            confidence_score=match.confidence_score,
            sources=match.sources,  # list[SourceItem] — already validated by factcheck service
        )

    # --- No match / not configured: placeholder (Phase 1 behavior) ---
    # TODO (Phase 3): add claim classification layer before this fallback.
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
