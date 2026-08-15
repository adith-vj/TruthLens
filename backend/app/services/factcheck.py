"""
services/factcheck.py — Google Fact Check Tools API interface.

SCAFFOLDING PHASE: This file defines the data contract for the future
Google Fact Check API integration. No network calls or implementation logic
are present. This module is NOT imported or invoked by any route handler
in the current scaffold.

--- Future implementation task (Phase 2) ---

The Google Fact Check Tools API (https://developers.google.com/fact-check/tools/api)
provides access to existing fact-checks published by recognized fact-checking
organizations around the world.

IMPORTANT DESIGN CONSTRAINT:
    The API must be treated as a source of EXISTING fact-checks, not as a
    universal truth engine. It returns results only when a credible organization
    has already fact-checked that specific claim (or a very similar one).
    If no relevant fact-check is found, the pipeline must fall through to the
    LLM/search fallback layer (Phase 4) rather than fabricating a result.

Normalization responsibility:
    The Google Fact Check API returns ratings in free-text form from individual
    publishers (e.g., "Mostly True", "False", "Pants on Fire"). The implementation
    must map these heterogeneous ratings to our four standard verdict values:
        "true", "false", "misleading", "unverifiable"

API key:
    Configured via GOOGLE_FACTCHECK_API_KEY in settings. The implementation
    must check that this key is non-empty before making any request and raise
    a ConfigurationError if it is missing.

httpx:
    All HTTP calls must use httpx.AsyncClient so they are non-blocking inside
    FastAPI async route handlers. Never use requests.get() in this service.
"""

from __future__ import annotations

from typing import Literal

from pydantic import AnyUrl, BaseModel, Field


class FactCheckMatch(BaseModel):
    """
    A normalized fact-check result from the Google Fact Check Tools API.

    This is the internal data transfer object used between the factcheck
    service and the route handler. It is NOT the public VerifyResponse —
    the route handler is responsible for converting FactCheckMatch →
    VerifyResponse after applying any additional business logic.

    Fields:
        verdict:          Normalized verdict mapped from the raw publisher rating.
        confidence_score: A value in [0.0, 1.0] representing confidence in
                          the normalization. Lower when the raw rating is
                          ambiguous or does not map cleanly to our taxonomy.
        sources:          The original fact-check articles returned by the API.
        raw_rating:       The original, unnormalized rating string from the
                          fact-checking publisher (e.g., "Mostly False").
                          Preserved for auditability.
        publisher:        The fact-checking organization that produced the review.
    """

    verdict: Literal["true", "false", "misleading", "unverifiable"]
    confidence_score: float = Field(ge=0.0, le=1.0)
    sources: list[dict[str, str]] = Field(default_factory=list)
    raw_rating: str
    publisher: str


# ---------------------------------------------------------------------------
# Future function signature — DO NOT implement until Phase 2
# ---------------------------------------------------------------------------
#
# async def query_factcheck_api(claim: str) -> FactCheckMatch | None:
#     """
#     Query the Google Fact Check Tools API for an existing fact-check
#     matching the given claim.
#
#     Args:
#         claim: The verified, non-empty claim text from the user.
#
#     Returns:
#         A FactCheckMatch if a sufficiently relevant fact-check is found,
#         or None if no match exists. The caller must handle the None case
#         by falling through to the LLM/search fallback layer.
#
#     Raises:
#         httpx.TimeoutException:     If the API request times out.
#         httpx.HTTPStatusError:      If the API returns a 4xx or 5xx response.
#         ConfigurationError:         If GOOGLE_FACTCHECK_API_KEY is not set.
#     """
#     ...
