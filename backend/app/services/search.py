"""
services/search.py — Tavily AI web search client.

Architecture
------------
This module is organised in four layers:

    1. Exceptions   — Custom hierarchy covering every Tavily failure mode.
    2. Constants    — Tavily endpoint URL (exported for test mocking).
    3. Helpers      — Pure sync functions for domain extraction and result
                      normalization.  Zero knowledge of HTTP or settings.
    4. Orchestrator — Public async entry point called by the route handler.

Source-integrity guarantee
---------------------------
SearchResult objects returned by this service are the ONLY permissible source
of URL, title, publisher, and snippet metadata that may appear in the final
VerifyResponse.sources array.  The LLM layer must never supplement these with
invented citations.

Tavily credit usage
--------------------
search_depth is always "basic" (1 credit/request).  "advanced" costs 2
credits/request and would halve the 1,000-credit monthly free quota.
auto_parameters is never enabled to prevent silent upgrades to "advanced".

Publisher metadata
-------------------
Tavily's API response does not include a publisher field.  The publisher
display label is derived from the URL's registered domain (eTLD+1) for
convenience only.  It is NOT a verified publisher identity.
"""

from __future__ import annotations

from urllib.parse import urlparse

import httpx
from pydantic import AnyUrl, BaseModel, ValidationError

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 0. Re-exported SearchResult model
#    (originally declared here in the Phase 1 stub — kept in this module)
# ---------------------------------------------------------------------------


class SearchResult(BaseModel):
    """
    A single normalised search result returned by the Tavily search service.

    These results are passed to the LLM as grounding evidence.
    Only SearchResult objects from this service may appear as sources in
    the final VerifyResponse — never fabricated URLs or citations.

    Fields:
        title:     Title of the search result page.
        url:       URL of the result.  Validated as a structurally correct URL.
        snippet:   Text excerpt (Tavily's `content` field) used as evidence
                   context for the LLM.
        publisher: Domain-derived display label.  NOT a verified publisher
                   identity.  None if the domain label cannot be extracted.
    """

    title: str
    url: AnyUrl
    snippet: str
    publisher: str | None = None


# ---------------------------------------------------------------------------
# 1. Exception hierarchy
# ---------------------------------------------------------------------------


class SearchError(Exception):
    """Base class for all search service errors."""


class SearchConfigError(SearchError):
    """
    Raised before any network request when TAVILY_API_KEY is missing or empty.
    The route catches this and returns 200 unverifiable without surfacing it
    as a 5xx error.
    """


class SearchTimeoutError(SearchError):
    """Raised when the Tavily API request exceeds SEARCH_TIMEOUT_SECONDS."""


class SearchQuotaError(SearchError):
    """Raised when the Tavily API returns HTTP 429 (rate limit / quota exceeded)."""


class SearchServiceError(SearchError):
    """
    Raised on HTTP errors (4xx other than 429, 5xx) or connection failures.
    The route catches this and returns 200 unverifiable.
    """


# ---------------------------------------------------------------------------
# 2. Constants
# ---------------------------------------------------------------------------

# Exported so tests can import it for respx URL matching.
TAVILY_SEARCH_URL = "https://api.tavily.com/search"

# ---------------------------------------------------------------------------
# 3. Helpers — pure sync functions
# ---------------------------------------------------------------------------


def _derive_domain_label(url: str) -> str | None:
    """
    Derive a display label from the URL's registered domain.

    This is NOT a verified publisher identity. It is a convenience label
    for display purposes only, derived mechanically from the URL hostname.

    Returns the registrable-domain-style label for common domains:
        https://www.reuters.com/article -> reuters.com
        https://en.wikipedia.org/wiki/   -> wikipedia.org
        https://bbc.co.uk/news/          -> bbc.co.uk

    Returns None if the URL cannot be parsed or has no hostname.
    """
    try:
        parsed = urlparse(url)
        host = parsed.hostname

        if not host:
            return None

        host = host.lower().rstrip(".")

        # Remove the conventional www. prefix.
        if host.startswith("www."):
            host = host[4:]

        parts = host.split(".")

        if len(parts) < 2:
            return host or None

        # Common multi-label public suffixes.
        # Keep the registrable domain + the public suffix.
        common_two_part_suffixes = {
            "co.uk",
            "org.uk",
            "ac.uk",
            "gov.uk",
            "com.au",
            "net.au",
            "org.au",
            "co.in",
            "firm.in",
            "net.in",
            "org.in",
            "gen.in",
            "ind.in",
            "co.nz",
            "net.nz",
            "org.nz",
            "co.jp",
            "ne.jp",
            "or.jp",
            "com.br",
            "com.cn",
            "com.sg",
            "com.my",
            "co.za",
        }

        suffix = ".".join(parts[-2:])

        if suffix in common_two_part_suffixes and len(parts) >= 3:
            return ".".join(parts[-3:])

        # Normal domains such as wikipedia.org, reuters.com, nhs.uk.
        return ".".join(parts[-2:])

    except Exception:  # pragma: no cover
        return None


def _normalize_result(raw: dict) -> SearchResult | None:
    """
    Map one Tavily result dict → SearchResult.

    Returns None (and logs a warning) if:
      - The 'url' field is missing or empty.
      - The 'content' field (snippet) is missing or empty.
      - The URL fails Pydantic's AnyUrl validation.
      - The 'title' field is missing (title defaults to empty string instead
        of causing a skip, to avoid discarding valid evidence).

    The 'title' field defaults to "" rather than causing a skip so that a
    missing title does not discard otherwise valid evidence.
    """
    url_raw = raw.get("url", "")
    snippet = raw.get("content", "")
    title = raw.get("title", "")

    if not url_raw:
        logger.warning("Tavily result missing 'url' field — skipping")
        return None
    if not snippet:
        logger.warning(
            "Tavily result missing 'content' field (url=%s) — skipping", url_raw
        )
        return None

    try:
        validated_url = AnyUrl(url_raw)
    except (ValidationError, ValueError) as exc:
        logger.warning(
            "Tavily result has unparseable URL %r: %s — skipping", url_raw, exc
        )
        return None

    publisher = _derive_domain_label(url_raw)

    return SearchResult(
        title=title,
        url=validated_url,
        snippet=snippet,
        publisher=publisher,
    )


# ---------------------------------------------------------------------------
# 4. Orchestrator — public async entry point
# ---------------------------------------------------------------------------


async def search_evidence(claim: str) -> list[SearchResult]:
    """
    Retrieve web search results relevant to the given claim via Tavily AI.

    The response is normalised into a list of SearchResult objects.
    Results with missing or unparseable URL/content fields are silently
    skipped; other results in the same response are still returned.

    Args:
        claim: The verified, non-empty claim text from the user.

    Returns:
        A (possibly empty) list of SearchResult objects.  An empty list means
        no usable results were found — the caller must NOT invoke the LLM
        in this case.

    Raises:
        SearchConfigError:  TAVILY_API_KEY is not configured.
        SearchTimeoutError: The HTTP request exceeded SEARCH_TIMEOUT_SECONDS.
        SearchQuotaError:   Tavily returned HTTP 429.
        SearchServiceError: Any other HTTP error or connection failure.
    """
    api_key = settings.TAVILY_API_KEY.get_secret_value()
    if not api_key:
        raise SearchConfigError(
            "TAVILY_API_KEY is not configured — search layer unavailable"
        )

    payload = {
        "query": claim,
        "search_depth": "basic",      # 1 credit/request; never use "auto"
        "max_results": settings.SEARCH_MAX_RESULTS,
        "include_answer": False,
        "include_raw_content": False,
        "topic": "general",
    }

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(settings.SEARCH_TIMEOUT_SECONDS)
        ) as client:
            response = await client.post(
                TAVILY_SEARCH_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )

    except httpx.TimeoutException as exc:
        logger.warning(
            "Tavily search timed out after %.1fs: %s",
            settings.SEARCH_TIMEOUT_SECONDS,
            exc,
        )
        raise SearchTimeoutError(str(exc)) from exc

    except httpx.ConnectError as exc:
        logger.error("Tavily connection error: %s", exc)
        raise SearchServiceError(str(exc)) from exc

    except httpx.RequestError as exc:
        logger.error("Tavily request error: %s", exc)
        raise SearchServiceError(str(exc)) from exc

    # --- HTTP status handling ---
    if response.status_code == 429:
        logger.warning(
            "Tavily API quota exceeded (HTTP 429). "
            "Monthly free-tier credit limit may have been reached."
        )
        raise SearchQuotaError("Tavily API quota exceeded (HTTP 429)")

    if response.status_code >= 400:
        logger.error(
            "Tavily API returned HTTP %d: %s",
            response.status_code,
            response.text[:200],
        )
        raise SearchServiceError(
            f"Tavily API error (HTTP {response.status_code})"
        )

    # --- Parse and normalize results ---
    try:
        data = response.json()
    except Exception as exc:
        logger.error("Tavily response JSON parse failure: %s", exc)
        raise SearchServiceError("Tavily response was not valid JSON") from exc

    raw_results = data.get("results", [])
    if not isinstance(raw_results, list):
        logger.warning("Tavily 'results' field is not a list — returning empty")
        return []

    results: list[SearchResult] = []
    for raw in raw_results:
        if not isinstance(raw, dict):
            continue
        normalized = _normalize_result(raw)
        if normalized is not None:
            results.append(normalized)

    logger.info(
        "Tavily search returned %d usable results (raw=%d)",
        len(results),
        len(raw_results),
    )
    return results
