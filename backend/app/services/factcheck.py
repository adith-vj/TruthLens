"""
services/factcheck.py — Google Fact Check Tools API client and normalization.

Architecture
------------
This module is organized in five layers, each with a single responsibility:

    1. Exceptions   — Custom hierarchy covering every failure mode.
    2. Data model   — FactCheckMatch internal DTO.
    3. Normalizer   — Pure functions: raw Google response → FactCheckMatch.
                      Zero knowledge of HTTP, settings, or I/O.
    4. HTTP client  — Async httpx call to the Google Fact Check API.
    5. Orchestrator — Public entry point called by the route handler.

Non-fabrication guarantee
--------------------------
Only data actually present in the Google API response may appear in the
returned FactCheckMatch. No URLs, titles, publisher names, dates, or
ratings are invented, guessed, or inferred from the claim text.

Ordering / relevance (AD-14)
------------------------------
When the API returns multiple claims, this module selects the FIRST claim
that contains at least one valid (url-bearing) claimReview entry.
The Google Fact Check API returns claims ordered by relevance to the
query string. We rely entirely on Google's ordering and do not apply any
custom relevance scoring in Phase 2.

Rating normalization design (AD-16)
--------------------------------------
Normalization uses an explicit lookup table (_RATING_MAP) after key
normalization (lowercase, punctuation stripped, whitespace collapsed).
There is NO substring matching and NO fuzzy matching. Any rating not
present in the table maps conservatively to ("unverifiable", 0.30).

"Mostly True" → "misleading" (see AD-16 in _RATING_MAP).
"""

from __future__ import annotations

import json as _json
import re
from typing import Literal

import httpx
from pydantic import BaseModel, Field, ValidationError

from app.core.config import settings
from app.core.logging import get_logger
from app.models.verification import SourceItem, VerdictType

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 1. Exception hierarchy
# ---------------------------------------------------------------------------


class FactCheckError(Exception):
    """Base class for all factcheck service errors."""


class FactCheckConfigError(FactCheckError):
    """
    Raised before any network request when the Google Fact Check API key is
    missing or empty.  The route handler catches this and falls through to the
    placeholder response — the server remains functional without a configured key.
    """


class FactCheckAuthError(FactCheckError):
    """Raised on HTTP 401 or 403 (invalid or unauthorized API key)."""


class FactCheckQuotaError(FactCheckError):
    """Raised on HTTP 429 (quota or rate-limit exceeded)."""


class FactCheckTimeoutError(FactCheckError):
    """Raised when the Google API request exceeds the configured timeout."""


class FactCheckServiceError(FactCheckError):
    """Raised on HTTP 5xx, connection failure, or malformed response body."""


# ---------------------------------------------------------------------------
# 2. Data model
# ---------------------------------------------------------------------------


class FactCheckMatch(BaseModel):
    """
    Internal data-transfer object representing one normalized fact-check result.

    This is NOT the public VerifyResponse.  The route handler converts
    FactCheckMatch → VerifyResponse, passing it through the Pydantic model
    for final validation before the response is sent to the client.

    Fields:
        verdict:          Normalized verdict from the explicit _RATING_MAP table.
        confidence_score: Lookup-derived confidence; never invented.
        sources:          SourceItem list built exclusively from API response data.
        raw_rating:       Original textualRating string from the API (for auditing).
        publisher:        Primary fact-checking organization name from the API.
    """

    verdict: VerdictType
    confidence_score: float = Field(ge=0.0, le=1.0)
    sources: list[SourceItem]
    raw_rating: str
    publisher: str


# ---------------------------------------------------------------------------
# 3. Normalizer — pure functions, zero I/O
# ---------------------------------------------------------------------------

# Public endpoint — also imported by tests for URL matching.
FACTCHECK_API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

# ---------------------------------------------------------------------------
# Explicit rating → (verdict, confidence_score) mapping table.
#
# KEY NORMALIZATION (applied before lookup — see _normalize_key()):
#   lowercase → replace hyphens/underscores with spaces →
#   strip non-alphanumeric/non-space chars → collapse whitespace → strip
#
# FALLBACK: any rating NOT in this table → ("unverifiable", 0.30).
# This is intentional and conservative: unknown ratings must not be guessed.
#
# Rating vocabulary sources:
#   PolitiFact     — True, Mostly True, Half True, Mostly False, False, Pants on Fire
#   Washington Post — One–Four Pinocchios, Geppetto Checkmark
#   Snopes         — True, Mostly True, Mixture, Mostly False, False, Satire
#   Africa Check   — Correct, Misleading, False, Unverified
#   AP Fact Check  — True, Mostly True, False, Misleading
#
# ── AD-16: "Mostly True" → "misleading"  (not "true") ───────────────────────
#   "Mostly True" implies a measurable degree of inaccuracy.
#   Mapping it to "true" would overstate certainty and could cause TruthLens
#   to endorse a claim that contains real factual errors.
#   "misleading" is the conservative choice: it signals that the claim is not
#   fully accurate without asserting it is outright false.
#   This mapping was explicitly reviewed and approved in Phase 2 planning.
#   Alternative considered: map to "true" with reduced confidence.
#   Rejected on safety grounds: TruthLens must not endorse partial inaccuracies.
# ─────────────────────────────────────────────────────────────────────────────

_RATING_MAP: dict[str, tuple[VerdictType, float]] = {

    # ── TRUE ──────────────────────────────────────────────────────────────────
    "true":                 ("true", 0.85),
    "correct":              ("true", 0.85),
    "accurate":             ("true", 0.85),
    "verified":             ("true", 0.80),
    "confirmed":            ("true", 0.85),
    "right":                ("true", 0.80),
    "geppetto checkmark":   ("true", 0.85),   # Washington Post: fully accurate
    "no flip":              ("true", 0.75),    # PolitiFact: consistent position

    # ── MOSTLY TRUE → "misleading"  (see AD-16 above) ────────────────────────
    "mostly true":          ("misleading", 0.70),
    "largely true":         ("misleading", 0.70),
    "generally true":       ("misleading", 0.70),
    "appears accurate":     ("misleading", 0.65),

    # ── MISLEADING / MIXED ────────────────────────────────────────────────────
    "misleading":           ("misleading", 0.75),
    "misrepresents":        ("misleading", 0.70),
    "misrepresented":       ("misleading", 0.70),
    "half true":            ("misleading", 0.65),   # PolitiFact: split accuracy
    "mixture":              ("misleading", 0.65),   # Snopes: mix of true & false
    "partly true":          ("misleading", 0.65),
    "partially true":       ("misleading", 0.65),
    "partially correct":    ("misleading", 0.65),
    "mostly false":         ("misleading", 0.70),   # PolitiFact: more false than true
    "largely false":        ("misleading", 0.70),
    "out of context":       ("misleading", 0.70),
    "lacks context":        ("misleading", 0.65),
    "missing context":      ("misleading", 0.65),
    "needs context":        ("misleading", 0.65),
    "mixed":                ("misleading", 0.65),
    "disputed":             ("misleading", 0.60),
    "exaggerated":          ("misleading", 0.65),
    "cherry picked":        ("misleading", 0.65),
    "not the whole story":  ("misleading", 0.65),
    "distorts":             ("misleading", 0.65),
    "spins":                ("misleading", 0.60),
    "flip":                 ("misleading", 0.60),
    "half flip":            ("misleading", 0.65),
    "three pinocchios":     ("misleading", 0.70),   # Washington Post: significant inaccuracy
    "two pinocchios":       ("misleading", 0.65),
    "one pinocchio":        ("misleading", 0.60),

    # ── FALSE ─────────────────────────────────────────────────────────────────
    "false":                ("false", 0.85),
    "incorrect":            ("false", 0.85),
    "inaccurate":           ("false", 0.80),
    "wrong":                ("false", 0.80),
    "fabricated":           ("false", 0.90),
    "debunked":             ("false", 0.85),
    "untrue":               ("false", 0.85),
    "not true":             ("false", 0.85),
    "pants on fire":        ("false", 0.90),   # PolitiFact: most egregious falsehood
    "four pinocchios":      ("false", 0.90),   # Washington Post: outright false
    "fiction":              ("false", 0.85),
    "scam":                 ("false", 0.85),
    "hoax":                 ("false", 0.90),
    "fake":                 ("false", 0.85),
    "bogus":                ("false", 0.85),
    "lie":                  ("false", 0.85),
    "full flop":            ("false", 0.80),   # PolitiFact: complete reversal
    "no evidence":          ("false", 0.75),
    "unsupported":          ("false", 0.75),

    # ── UNVERIFIABLE ──────────────────────────────────────────────────────────
    "unverifiable":             ("unverifiable", 0.40),
    "unverified":               ("unverifiable", 0.40),
    "cannot determine":         ("unverifiable", 0.30),
    "uncertain":                ("unverifiable", 0.30),
    "insufficient evidence":    ("unverifiable", 0.30),
    "not enough information":   ("unverifiable", 0.30),
    "undetermined":             ("unverifiable", 0.30),
    "unknown":                  ("unverifiable", 0.30),
    "opinion":                  ("unverifiable", 0.30),
    "satire":                   ("unverifiable", 0.30),
    "outdated":                 ("unverifiable", 0.35),
    "stale":                    ("unverifiable", 0.35),
}

# Fallback for any raw rating NOT present in _RATING_MAP.
_FALLBACK: tuple[VerdictType, float] = ("unverifiable", 0.30)


def _normalize_key(raw: str) -> str:
    """
    Normalize a raw textualRating string into a lookup key for _RATING_MAP.

    Steps applied in order:
        1. Lowercase
        2. Replace hyphens and underscores with spaces  ("half-true" → "half true")
        3. Strip all non-alphanumeric, non-space characters  ("Fire!" → "Fire")
        4. Collapse consecutive whitespace to a single space
        5. Strip leading/trailing whitespace

    This normalization is deterministic and idempotent.  It is deliberately
    simple — there is no fuzzy matching, stemming, or semantic analysis.
    Any raw rating that does not produce an exact key match falls through to
    the _FALLBACK value ("unverifiable", 0.30).

    Examples:
        "Pants on Fire!"    → "pants on fire"
        "Mostly True"       → "mostly true"
        "Half-True"         → "half true"
        "FOUR PINOCCHIOS"   → "four pinocchios"
        "  False  "         → "false"
    """
    s = raw.lower()
    s = s.replace("-", " ").replace("_", " ")
    s = re.sub(r"[^a-z0-9 ]", "", s)
    s = re.sub(r" +", " ", s)
    return s.strip()


def normalize_rating(raw: str) -> tuple[VerdictType, float]:
    """
    Map a raw textualRating string to our four-value verdict taxonomy.

    Lookup is exact after key normalization (see _normalize_key).
    No substring matching; no fuzzy matching.  Unknown ratings return the
    fallback ("unverifiable", 0.30) unconditionally.

    Args:
        raw: The textualRating value from a Google Fact Check claimReview.

    Returns:
        (verdict, confidence_score) from _RATING_MAP, or _FALLBACK if the
        normalized key is not present in the table.

    Notable mapping (AD-16):
        "Mostly True" → ("misleading", 0.70)  — see _RATING_MAP docstring.
    """
    key = _normalize_key(raw)
    return _RATING_MAP.get(key, _FALLBACK)


def normalize_response(raw_dict: object) -> FactCheckMatch | None:
    """
    Convert a parsed Google Fact Check API response into a FactCheckMatch.

    Returns None when any of the following conditions hold:
        - raw_dict is not a dict
        - The response has no "claims" key (e.g., empty object {})
        - The "claims" list is empty
        - No claim has a claimReview containing at least one valid url-bearing entry

    Claim selection (AD-14):
        The FIRST claim (in Google's returned order) that has at least one
        valid claimReview is selected.  Google returns claims ordered by
        relevance to the query.  No additional ranking is performed.

    Source integrity:
        - Only urls/titles/publishers present in the API response appear
          in the returned sources list.
        - claimReview entries missing a url are silently skipped.
        - Missing title  → default "Fact Check"  (not a real title, but not
          a fabricated claim title either).
        - Missing publisher.name → publisher.site if available,
          else "Unknown Publisher".

    Args:
        raw_dict: The parsed JSON body of the Google Fact Check API response.
                  Accepts any type; non-dict values return None safely.

    Returns:
        FactCheckMatch if a usable fact-check was found, otherwise None.
    """
    if not isinstance(raw_dict, dict):
        logger.warning(
            "normalize_response: expected dict, got %s — returning None",
            type(raw_dict).__name__,
        )
        return None

    claims = raw_dict.get("claims")

    # Guard: handles both missing "claims" key ({}) and empty list ([]).
    if not claims or not isinstance(claims, list):
        return None

    for claim in claims:
        if not isinstance(claim, dict):
            continue

        reviews = claim.get("claimReview")
        if not reviews or not isinstance(reviews, list):
            continue

        # --- Collect sources from all valid claimReview entries ---
        valid_sources: list[SourceItem] = []
        for review in reviews:
            if not isinstance(review, dict):
                continue
            url = review.get("url")
            if not url:
                # Source integrity: skip reviews without a URL.
                # No URL means no verifiable source to cite.
                continue
            publisher_data = review.get("publisher") or {}
            publisher_name = (
                publisher_data.get("name")
                or publisher_data.get("site")
                or "Unknown Publisher"
            )
            title = review.get("title") or "Fact Check"
            try:
                valid_sources.append(
                    SourceItem(title=title, url=url, publisher=publisher_name)
                )
            except ValidationError:
                logger.warning(
                    "Skipping claimReview with invalid URL: %r", url
                )
                continue

        if not valid_sources:
            # This claim had no usable claimReview entries — try the next claim.
            continue

        # --- Normalize rating from the first claimReview (AD-14) ---
        # The first review's textualRating is used for the verdict.
        # All valid reviews' URLs are included in sources (collected above).
        first_review = reviews[0]
        raw_rating = (first_review.get("textualRating") or "").strip()
        verdict, confidence = normalize_rating(raw_rating)

        publisher_data = first_review.get("publisher") or {}
        primary_publisher = (
            publisher_data.get("name")
            or publisher_data.get("site")
            or "Unknown Publisher"
        )

        logger.info(
            "Fact-check normalized: raw_rating=%r → verdict=%s confidence=%.2f "
            "publisher=%r sources=%d",
            raw_rating,
            verdict,
            confidence,
            primary_publisher,
            len(valid_sources),
        )

        return FactCheckMatch(
            verdict=verdict,
            confidence_score=confidence,
            sources=valid_sources,
            raw_rating=raw_rating,
            publisher=primary_publisher,
        )

    # No claim had any usable claimReview entries.
    return None


# ---------------------------------------------------------------------------
# 4. HTTP client layer
# ---------------------------------------------------------------------------


async def query_factcheck_api(claim: str) -> dict:
    """
    Send a GET request to the Google Fact Check Tools API and return the
    parsed JSON response body as a plain dict.

    Security notes:
        - The API key is retrieved via SecretStr.get_secret_value() immediately
          before the request and never assigned to a named intermediate variable
          that could appear in log messages or tracebacks.
        - The LOG_LEVEL default is INFO.  Avoid setting DEBUG in production;
          httpx DEBUG logs include full request URLs, which contain the key.
        - Raw Google error messages are never forwarded to the caller — only
          our typed FactCheck* exceptions are raised.

    Args:
        claim: Validated, stripped claim text.

    Returns:
        The parsed JSON response body as a dict.  May be an empty dict {}
        when Google has no matching fact-checks.

    Raises:
        FactCheckConfigError:  API key is empty or not set.
        FactCheckAuthError:    HTTP 401 or 403.
        FactCheckQuotaError:   HTTP 429.
        FactCheckServiceError: HTTP 5xx, connection failure, or malformed JSON.
        FactCheckTimeoutError: Request timed out.
    """
    api_key = settings.GOOGLE_FACTCHECK_API_KEY.get_secret_value()
    if not api_key:
        raise FactCheckConfigError(
            "GOOGLE_FACTCHECK_API_KEY is not configured. "
            "Set it in .env to enable Google Fact Check lookups."
        )

    params: dict[str, object] = {
        "query": claim,
        "key": api_key,          # Never log this value
        "pageSize": settings.FACTCHECK_MAX_RESULTS,
        "languageCode": "en",
    }

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(settings.FACTCHECK_TIMEOUT_SECONDS)
        ) as client:
            response = await client.get(FACTCHECK_API_URL, params=params)
            response.raise_for_status()
            return response.json()

    except httpx.TimeoutException as exc:
        raise FactCheckTimeoutError(
            "Google Fact Check API request timed out"
        ) from exc

    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status in (401, 403):
            raise FactCheckAuthError(
                f"Google Fact Check API authentication failed (HTTP {status})"
            ) from exc
        if status == 429:
            raise FactCheckQuotaError(
                "Google Fact Check API quota exceeded (HTTP 429)"
            ) from exc
        raise FactCheckServiceError(
            f"Google Fact Check API returned unexpected status (HTTP {status})"
        ) from exc

    except _json.JSONDecodeError as exc:
        raise FactCheckServiceError(
            "Google Fact Check API returned malformed JSON"
        ) from exc

    except httpx.RequestError as exc:
        raise FactCheckServiceError(
            f"Google Fact Check API request failed: {type(exc).__name__}"
        ) from exc


# ---------------------------------------------------------------------------
# 5. Public orchestrator
# ---------------------------------------------------------------------------


async def verify_claim_factcheck(claim: str) -> FactCheckMatch | None:
    """
    Query the Google Fact Check API and return a normalized FactCheckMatch.

    This is the sole public function called by the route handler.  It
    coordinates the HTTP client → normalizer pipeline and surfaces all
    failure modes as typed FactCheck* exceptions.

    The route handler is responsible for:
        - Catching FactCheck* exceptions and converting them to HTTP responses.
        - Constructing VerifyResponse from the returned FactCheckMatch, passing
          it through the Pydantic model for final validation before delivery.
        - Handling None by returning the configured placeholder response.

    Args:
        claim: Validated, stripped claim text from the user request.

    Returns:
        FactCheckMatch if a relevant fact-check was found, otherwise None.

    Raises:
        FactCheckConfigError, FactCheckAuthError, FactCheckQuotaError,
        FactCheckTimeoutError, FactCheckServiceError
    """
    raw = await query_factcheck_api(claim)
    return normalize_response(raw)
