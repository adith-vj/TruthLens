"""
api/verify.py — Route handler for POST /api/verify.

SCAFFOLDING PHASE:
    This route validates the incoming request and returns a static placeholder
    VerifyResponse. No external services are called.

    Placeholder response:
        { "verdict": "unverifiable", "confidence_score": 0.0, "sources": [] }

    The verification orchestration pipeline (classifier → factcheck → LLM)
    will be wired in as each service is implemented in Phases 2–4.

Request flow (scaffold):
    POST /api/verify
      │
      ├─ Pydantic parses VerifyRequest
      │    └─ Missing or non-string 'text'     → 422
      │    └─ Empty / whitespace-only 'text'   → 422 (field_validator)
      │
      ├─ Route-level length check
      │    └─ len(text) > MAX_CLAIM_LENGTH      → 422
      │
      └─ Return placeholder VerifyResponse
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from fastapi import status as http_status

from app.core.config import settings
from app.core.logging import get_logger
from app.models.verification import VerifyRequest, VerifyResponse

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/verify",
    response_model=VerifyResponse,
    summary="Verify a factual claim",
    description=(
        "Accepts a text claim selected by the user and returns a structured "
        "verdict with a confidence score and supporting sources. "
        "During the scaffolding phase, always returns a placeholder response."
    ),
    responses={
        200: {"description": "Verification result"},
        422: {"description": "Invalid or malformed request body"},
        500: {"description": "Internal server error"},
    },
)
async def verify_claim(request: VerifyRequest) -> VerifyResponse:
    """
    Verify a factual claim submitted by the Chrome Extension.

    Validates the input and returns a standardized VerifyResponse.

    Args:
        request: Parsed and validated VerifyRequest body.

    Returns:
        VerifyResponse with verdict, confidence_score, and sources.

    Raises:
        HTTPException(422): If the claim text exceeds MAX_CLAIM_LENGTH.
    """
    claim_text = request.text.strip()

    # --- Route-level length check ---
    # This check lives here (not in the Pydantic validator) so that
    # MAX_CLAIM_LENGTH is configurable via settings without modifying
    # schema code.
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

    logger.info("Claim received (length=%d) — returning scaffold response", len(claim_text))

    # --- SCAFFOLD: Return placeholder response ---
    # TODO (Phase 2): Replace with verification pipeline:
    #   1. classify_claim(claim_text)       → ClaimType
    #   2. query_factcheck_api(claim_text)  → FactCheckMatch | None
    #   3. verify_with_llm(claim_text)      → VerifyResponse (fallback)
    return VerifyResponse(
        verdict="unverifiable",
        confidence_score=0.0,
        sources=[],
    )
