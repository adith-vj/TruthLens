"""
services/video_verify.py - Phase 5.5 video-specific verification orchestration.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime

import httpx

from app.core.config import settings
from app.core.logging import get_logger
from app.models.verification import SourceItem
from app.models.video import CandidateClaim
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
    LLMParseError,
    LLMQuotaError,
    LLMServiceError,
    LLMTimeoutError,
    GEMINI_VERIFY_URL,
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

GEMINI_FIRST_PASS_URL = GEMINI_VERIFY_URL

_TEMPORAL_KEYWORDS: frozenset = frozenset({
    "current", "currently", "latest", "today",
    "this week", "this month", "this year",
    "recent", "recently", "as of", "ongoing",
})

_TEMPORAL_KEYWORD_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(kw) for kw in sorted(_TEMPORAL_KEYWORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

_YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")

_VALID_FIRST_PASS_VERDICTS: frozenset = frozenset(
    {"true", "false", "misleading", "unverifiable", "uncertain"}
)


@dataclass
class UsageMetrics:
    google_factcheck_calls: int = 0
    gemini_first_pass_calls: int = 0
    gemini_evidence_calls: int = 0
    tavily_calls: int = 0

    def __add__(self, other: "UsageMetrics") -> "UsageMetrics":
        return UsageMetrics(
            google_factcheck_calls=self.google_factcheck_calls + other.google_factcheck_calls,
            gemini_first_pass_calls=self.gemini_first_pass_calls + other.gemini_first_pass_calls,
            gemini_evidence_calls=self.gemini_evidence_calls + other.gemini_evidence_calls,
            tavily_calls=self.tavily_calls + other.tavily_calls,
        )


@dataclass
class FirstPassResult:
    verdict: str
    confidence: float
    needs_web_search: bool
    reason: str = ""


@dataclass
class VideoVerifyResult:
    verdict: str
    confidence_score: float
    sources: list = field(default_factory=list)
    metrics: UsageMetrics = field(default_factory=UsageMetrics)


_verify_cache: dict = {}


def _make_cache_key(video_id: str, claim_text: str, context: str) -> tuple:
    claim_hash = hashlib.sha256(claim_text.encode()).hexdigest()[:16]
    context_hash = hashlib.sha256(context.encode()).hexdigest()[:8]
    version = settings.VIDEO_VERIFICATION_PIPELINE_VERSION
    return (video_id, claim_hash, context_hash, version)


def clear_verify_cache() -> None:
    _verify_cache.clear()


def _is_temporal_claim(claim_text: str) -> bool:
    if _TEMPORAL_KEYWORD_PATTERN.search(claim_text):
        return True
    current_year = datetime.now().year
    year_range = range(current_year - 2, current_year + 2)
    for match in _YEAR_PATTERN.finditer(claim_text):
        if int(match.group()) in year_range:
            return True
    return False


def _is_niche_claim(checkability_score: float) -> bool:
    return checkability_score >= settings.NICHE_CLAIM_CHECKABILITY_THRESHOLD


def should_skip_tavily(
    claim_text: str,
    first_pass: FirstPassResult,
    checkability_score: float,
) -> bool:
    threshold = settings.GEMINI_FIRST_PASS_CONFIDENCE_THRESHOLD
    if _is_temporal_claim(claim_text):
        logger.debug("Tavily required: temporal claim detected")
        return False
    if _is_niche_claim(checkability_score):
        logger.debug("Tavily required: niche claim (checkability=%.3f)", checkability_score)
        return False
    if first_pass.needs_web_search:
        logger.debug("Tavily required: Gemini requested web search")
        return False
    if first_pass.confidence < threshold:
        logger.debug("Tavily required: confidence %.3f < threshold %.3f", first_pass.confidence, threshold)
        return False
    return True


_FIRST_PASS_PROMPT = """\
You are a fact-checking assistant performing an initial assessment of a claim.

Claim: "{claim}"
{context_block}
Assess this claim using your pretrained knowledge.

Respond with a JSON object:
{{
  "verdict": "<one of: true, false, misleading, uncertain, unverifiable>",
  "confidence": <float between 0.0 and 1.0>,
  "needs_web_search": <true or false>,
  "reason": "<one sentence explanation>"
}}

Rules:
1. verdict must be exactly one of: "true", "false", "misleading", "uncertain", "unverifiable".
   - Use "uncertain" when you have partial or conflicting knowledge.
   - Use "unverifiable" when the claim cannot be assessed from pretrained knowledge.
   - Never invent facts not present in your training data.
2. confidence must be a number between 0.0 and 1.0.
   - Use low confidence (< 0.5) when you are genuinely unsure.
   - Use values > 0.80 only when you have strong, reliable knowledge of this specific fact.
3. needs_web_search must be true when any of these apply:
   - The claim involves current events, recent statistics, or recent activities.
   - The claim involves specific recent dates (within the last 2-3 years).
   - The claim involves niche, unusual, or highly specific facts you are not certain about.
   - You have any doubt about the accuracy or currency of your knowledge.
   - confidence < 0.80
4. Return ONLY valid JSON. No explanation outside the JSON object.\
"""


def _build_first_pass_prompt(claim: str, context: str) -> str:
    context_block = f"Context: {context}\n" if context else ""
    return _FIRST_PASS_PROMPT.format(claim=claim, context_block=context_block)


def _parse_first_pass_response(raw_json: dict) -> FirstPassResult:
    try:
        candidates = raw_json.get("candidates", [])
        if not candidates:
            raise ValueError("No candidates")
        text = candidates[0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Bad structure: {exc}") from exc

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Non-JSON: {text[:200]!r}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"Not a dict: {type(parsed).__name__}")

    verdict = parsed.get("verdict", "uncertain")
    if verdict not in _VALID_FIRST_PASS_VERDICTS:
        verdict = "uncertain"

    confidence_raw = parsed.get("confidence", 0.0)
    if not isinstance(confidence_raw, (int, float)):
        confidence_raw = 0.0
    confidence = float(max(0.0, min(1.0, confidence_raw)))

    needs_web = bool(parsed.get("needs_web_search", True))
    reason = str(parsed.get("reason", ""))

    return FirstPassResult(verdict=verdict, confidence=confidence, needs_web_search=needs_web, reason=reason)


async def _gemini_first_pass(claim: str, context: str) -> FirstPassResult:
    api_key = settings.GEMINI_API_KEY.get_secret_value()
    if not api_key:
        logger.warning("GEMINI_API_KEY not configured - first-pass skipped")
        return FirstPassResult(verdict="uncertain", confidence=0.0, needs_web_search=True)

    prompt = _build_first_pass_prompt(claim, context)
    request_body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json"},
    }

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(settings.LLM_TIMEOUT_SECONDS)) as client:
            response = await client.post(
                GEMINI_FIRST_PASS_URL,
                json=request_body,
                params={"key": api_key},
            )
    except httpx.TimeoutException:
        logger.warning("Gemini first-pass timed out")
        return FirstPassResult(verdict="uncertain", confidence=0.0, needs_web_search=True)
    except httpx.RequestError as exc:
        logger.warning("Gemini first-pass request error: %s", exc)
        return FirstPassResult(verdict="uncertain", confidence=0.0, needs_web_search=True)

    if response.status_code == 429:
        logger.warning("Gemini first-pass quota exceeded")
        raise LLMQuotaError("Gemini first-pass quota exceeded (HTTP 429)")

    if response.status_code >= 400:
        logger.error("Gemini first-pass HTTP %d", response.status_code)
        return FirstPassResult(verdict="uncertain", confidence=0.0, needs_web_search=True)

    try:
        raw_json = response.json()
        result = _parse_first_pass_response(raw_json)
    except Exception as exc:
        logger.error("Gemini first-pass parse error: %s", exc)
        return FirstPassResult(verdict="uncertain", confidence=0.0, needs_web_search=True)

    logger.info(
        "Gemini first-pass: verdict=%s confidence=%.2f needs_web_search=%s",
        result.verdict, result.confidence, result.needs_web_search,
    )
    return result


def _build_sources(evidence: list, source_indices: list) -> list:
    seen_urls: set = set()
    sources = []
    for idx in source_indices:
        if not (0 <= idx < len(evidence)):
            continue
        result = evidence[idx]
        url_str = str(result.url)
        if url_str in seen_urls:
            continue
        seen_urls.add(url_str)
        sources.append(SourceItem(
            title=result.title,
            url=result.url,
            publisher=result.publisher or "",
        ))
    return sources


async def verify_video_claim(
    claim: CandidateClaim,
    video_id: str,
    context: str = "",
) -> VideoVerifyResult:
    claim_text = claim.text.strip()
    metrics = UsageMetrics()

    cache_key = _make_cache_key(video_id, claim_text, context)
    if cache_key in _verify_cache:
        logger.debug("Video verify cache hit: %s...", claim_text[:60])
        return _verify_cache[cache_key]

    def _cache_and_return(result: VideoVerifyResult) -> VideoVerifyResult:
        _verify_cache[cache_key] = result
        return result

    claim_type_val = claim.claim_type.lower()
    if claim_type_val in ("opinion", "advertisement"):
        logger.info("Video verify: early exit for claim_type=%s", claim_type_val)
        return _cache_and_return(VideoVerifyResult(
            verdict="unverifiable", confidence_score=0.0, sources=[], metrics=metrics,
        ))

    metrics.google_factcheck_calls += 1
    match = None
    try:
        match = await verify_claim_factcheck(claim_text)
    except FactCheckConfigError:
        logger.warning("Fact Check API key not configured - falling through")
        metrics.google_factcheck_calls = 0
    except (FactCheckAuthError, FactCheckQuotaError, FactCheckTimeoutError, FactCheckServiceError) as exc:
        logger.warning("Fact Check error (%s) - falling through", type(exc).__name__)

    if match is not None:
        confidence = match.confidence_score
        if claim_type_val == "ambiguous":
            confidence = max(0.0, min(1.0, confidence * 0.7))
        logger.info("Video verify: fact-check hit verdict=%s conf=%.2f", match.verdict, confidence)
        return _cache_and_return(VideoVerifyResult(
            verdict=match.verdict, confidence_score=confidence, sources=match.sources, metrics=metrics,
        ))

    metrics.gemini_first_pass_calls += 1
    try:
        first_pass = await _gemini_first_pass(claim_text, context)
    except LLMQuotaError:
        logger.warning("LLM first-pass quota exceeded - unverifiable")
        return _cache_and_return(VideoVerifyResult(
            verdict="unverifiable", confidence_score=0.0, sources=[], metrics=metrics,
        ))
    logger.info(
        "Video verify: first-pass verdict=%s conf=%.2f needs_web=%s reason=%s",
        first_pass.verdict, first_pass.confidence, first_pass.needs_web_search, first_pass.reason[:80],
    )

    if should_skip_tavily(claim_text, first_pass, claim.checkability_score):
        logger.info(
            "Video verify: Tavily skipped (verdict=%s conf=%.2f)",
            first_pass.verdict, first_pass.confidence,
        )
        final_verdict = first_pass.verdict if first_pass.verdict != "uncertain" else "unverifiable"
        return _cache_and_return(VideoVerifyResult(
            verdict=final_verdict, confidence_score=first_pass.confidence, sources=[], metrics=metrics,
        ))

    metrics.tavily_calls += 1
    evidence = []
    try:
        evidence = await search_evidence(claim_text)
        logger.info("Video verify: Tavily returned %d results", len(evidence))
    except SearchConfigError:
        logger.warning("TAVILY_API_KEY not configured - using first-pass result")
        final_verdict = first_pass.verdict if first_pass.verdict != "uncertain" else "unverifiable"
        return _cache_and_return(VideoVerifyResult(
            verdict=final_verdict, confidence_score=first_pass.confidence, sources=[], metrics=metrics,
        ))
    except (SearchQuotaError, SearchTimeoutError, SearchServiceError) as exc:
        logger.warning("Tavily error (%s) - unverifiable", type(exc).__name__)
        return _cache_and_return(VideoVerifyResult(
            verdict="unverifiable", confidence_score=0.0, sources=[], metrics=metrics,
        ))

    if not evidence:
        logger.info("Video verify: Tavily returned no evidence - unverifiable")
        return _cache_and_return(VideoVerifyResult(
            verdict="unverifiable", confidence_score=0.0, sources=[], metrics=metrics,
        ))

    metrics.gemini_evidence_calls += 1
    try:
        llm_verdict = await verify_with_llm(claim_text, evidence)
    except (LLMConfigError, LLMQuotaError, LLMTimeoutError, LLMServiceError, LLMParseError) as exc:
        logger.warning("LLM evidence evaluation failed (%s) - unverifiable", type(exc).__name__)
        return _cache_and_return(VideoVerifyResult(
            verdict="unverifiable", confidence_score=0.0, sources=[], metrics=metrics,
        ))

    sources = _build_sources(evidence, llm_verdict.source_indices)
    logger.info(
        "Video verify: evidence evaluation verdict=%s conf=%.2f sources=%d",
        llm_verdict.verdict, llm_verdict.confidence_score, len(sources),
    )
    return _cache_and_return(VideoVerifyResult(
        verdict=llm_verdict.verdict,
        confidence_score=llm_verdict.confidence_score,
        sources=sources,
        metrics=metrics,
    ))
