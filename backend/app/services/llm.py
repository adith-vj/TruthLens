"""
services/llm.py — LLM/Gemini verification fallback interface.

SCAFFOLDING PHASE: This file defines the contract for the future LLM-based
verification fallback. No implementation logic is present. This module is NOT
imported or invoked by any route handler in the current scaffold.

--- Future implementation task (Phase 4) ---

This service is the fallback layer invoked when the Google Fact Check Tools API
returns no relevant existing fact-check for a given claim.

The LLM (Google Gemini with search grounding) must:
    1. Formulate a search query from the claim
    2. Retrieve real search results via the search service (services/search.py)
    3. Evaluate the evidence from those results
    4. Return a verdict with a confidence score and the actual source URLs

NON-FABRICATION CONSTRAINTS — These are hard requirements, not suggestions:
    - The 'sources' array in VerifyResponse must ONLY contain URLs actually
      returned by the search tool. Never generate or infer URLs.
    - The LLM must NEVER fabricate:
        * URLs or domain names
        * Publisher or organization names
        * Publication dates
        * Quotes attributed to people or organizations
        * Statistical figures or numerical claims not present in search results
    - If no relevant evidence is found via search, the verdict must be
      'unverifiable' with a low confidence score, NOT a guess.
    - The LLM prompt must explicitly instruct the model against hallucination
      and must require it to cite only grounded sources.

Gemini API key:
    Configured via GEMINI_API_KEY in settings. The implementation must check
    for a non-empty key before making any request.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Future function signature — DO NOT implement until Phase 4
# ---------------------------------------------------------------------------
#
# async def verify_with_llm(claim: str) -> VerifyResponse:
#     """
#     Verify a claim using Gemini LLM with search grounding.
#
#     This is the fallback layer called when query_factcheck_api() returns None.
#
#     Args:
#         claim: The verified, non-empty claim text from the user.
#
#     Returns:
#         A VerifyResponse with a verdict, confidence score, and sources drawn
#         exclusively from actual search results — never fabricated.
#
#     Raises:
#         httpx.TimeoutException:  If the Gemini API request times out.
#         httpx.HTTPStatusError:   If the API returns a non-2xx response.
#         ConfigurationError:      If GEMINI_API_KEY is not set.
#
#     Non-fabrication guarantee:
#         The implementation MUST NOT return any URL, citation, publisher,
#         date, quote, or statistic that was not present in the search results
#         returned by services.search.search_evidence(). Violation of this
#         constraint produces misinformation and defeats the purpose of
#         TruthLens entirely.
#     """
#     ...
