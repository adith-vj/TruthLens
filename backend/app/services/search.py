"""
services/search.py — Evidence search service interface.

SCAFFOLDING PHASE: This file defines the data contract for the future search
integration that will ground LLM-based verification. No implementation logic
is present. This module is NOT imported or invoked by any route handler
in the current scaffold.

--- Future implementation task (Phase 4) ---

The search service retrieves real web search results to provide evidence for
the LLM verification layer (services/llm.py). The LLM must evaluate this
evidence rather than relying on its parametric knowledge.

Candidate implementations to evaluate in Phase 4:
    1. Google Custom Search JSON API
       https://developers.google.com/custom-search/v1/overview
    2. Gemini search grounding (built-in tool calling)
       https://ai.google.dev/gemini-api/docs/grounding
    3. Brave Search API (if Google quota is a concern)

Source integrity rule:
    SearchResult objects returned by this service are the ONLY permissible
    source of URLs that may appear in the final VerifyResponse.sources array.
    The LLM layer must never supplement these with invented citations.
"""

from __future__ import annotations

from pydantic import AnyUrl, BaseModel


class SearchResult(BaseModel):
    """
    A single search result returned by the search service.

    These results are passed to the LLM as grounding evidence.
    Only SearchResult objects from this service may appear as sources
    in the final VerifyResponse — never fabricated URLs or citations.

    Fields:
        title:     Title of the search result page.
        url:       URL of the result. Validated as a real URL.
        snippet:   Text excerpt from the search result, used as evidence
                   context for the LLM.
        publisher: Optional name of the publishing organization, if
                   determinable from the URL or result metadata.
    """

    title: str
    url: AnyUrl
    snippet: str
    publisher: str | None = None


# ---------------------------------------------------------------------------
# Future function signature — DO NOT implement until Phase 4
# ---------------------------------------------------------------------------
#
# async def search_evidence(claim: str) -> list[SearchResult]:
#     """
#     Retrieve web search results relevant to the given claim.
#
#     Args:
#         claim: The verified, non-empty claim text from the user.
#
#     Returns:
#         A list of SearchResult objects. May be empty if no relevant
#         results are found, in which case the LLM must return
#         verdict='unverifiable'.
#
#     Raises:
#         httpx.TimeoutException:  If the search API request times out.
#         httpx.HTTPStatusError:   If the API returns a non-2xx response.
#         ConfigurationError:      If the required search API key is not set.
#     """
#     ...
