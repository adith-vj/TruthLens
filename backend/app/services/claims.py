"""
services/claims.py — Phase 5.4 Claim Extraction & Prioritization.

Request strategy (v2, rate-limit-aware)
----------------------------------------
Previous approach: 1 Gemini request per 60-second chunk → ~15 requests for a
15-minute video PLUS one classify_claim() Gemini call per extracted claim →
~90 total Gemini requests per video. This saturated the free-tier quota.

Current approach:
  1. ALL transcript chunks are sent in ONE Gemini request.
     A 15-minute transcript is ~10k tokens — well within gemini-3.6-flash's
     1M-token context window.  For safety we split into at most MAX_BATCHES
     batches (default 2) only if the transcript is unusually long.
  2. The extraction prompt now asks the model to also output claim_type
     ("factual_claim" | "opinion" | "advertisement" | "ambiguous") inline,
     eliminating all subsequent classify_claim() Gemini calls.
  3. The existing rule-based heuristics from classifier._classify_by_rules()
     are applied synchronously AFTER extraction as a cheap override (zero I/O).
     This keeps the classification layer honest without any extra API calls.

Expected Gemini calls per video:
  10-minute video  → 1 call
  20-minute video  → 1 call
  60-minute video  → 1 call  (token limit only reached at ~6 hours of speech)
  >6-hour video    → 2 calls

Preserved from v1:
  - timestamp mapping (exact segment boundaries in JSON output)
  - chunk-boundary safety (no segment ever crosses a batch boundary)
  - deterministic Python scoring (checkability + claim_type multiplier)
  - SequenceMatcher deduplication (same threshold and merge logic)
  - Top-12 limit (configurable top_n parameter)
  - in-memory cache in api/video.py (unchanged)
"""

from __future__ import annotations

import asyncio
import json
import logging
from difflib import SequenceMatcher
from typing import Any

import httpx

from app.core.config import settings
from app.models.video import CandidateClaim
from app.services.classifier import _classify_by_rules  # rule-only, zero I/O
from app.services.youtube_transcript import get_transcript

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GEMINI_GENERATE_URL = (
    "https://generativelanguage.googleapis.com"
    "/v1beta/models/gemini-3.6-flash:generateContent"
)

# Target: keep each batch under this many transcript characters (~750k tokens).
# At ~4 chars/token, 15-min speech ≈ 15_000 chars — far below the limit.
# We set a conservative ceiling that allows even a 3-hour video in one shot.
_MAX_CHARS_PER_BATCH = 2_800_000  # ≈ 700k tokens at 4 chars/token

# Hard ceiling on batch count so the loop cannot run away.
MAX_BATCHES = 4

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_EXTRACT_PROMPT = """\
You are a factual claim extraction AI.  Read the full transcript below and
extract every externally checkable factual assertion that could be verified
against real-world sources.

Rules:
- A claim must be a *specific* assertion about the world (event, statistic,
  attribution, date, quantity, scientific finding, etc.).
- Do NOT extract pure opinions ("I think…"), rhetorical questions, predictions,
  greetings, or filler.
- Do NOT invent or modify timestamps.  Use the [start - end] values from the
  transcript lines containing the claim.
- Include a claim_type field classifying each claim into exactly one of:
    "factual_claim"  — objectively verifiable assertion
    "opinion"        — personal view or preference
    "advertisement"  — promotional / commercial content
    "ambiguous"      — unclear / borderline

Return ONLY a valid JSON array.  No markdown fences. No prose.

Schema for each element:
{{
  "text":        "<polished claim sentence>",
  "start_time":  <float seconds>,
  "end_time":    <float seconds>,
  "claim_type":  "<factual_claim|opinion|advertisement|ambiguous>",
  "has_numbers": <true|false>,
  "has_entities": <true|false>,
  "is_specific":  <true|false>
}}

Transcript:
{transcript_text}
"""

# Valid claim_type values that map to the Phase-3 ClaimType enum values.
_VALID_CLAIM_TYPES = {"factual_claim", "opinion", "advertisement", "ambiguous"}

# ---------------------------------------------------------------------------
# Chunking (kept for correctness / batch splitting; NOT used for API calls)
# ---------------------------------------------------------------------------


def chunk_transcript(
    segments: list[dict[str, Any]],
    max_duration: float = 60.0,
    overlap: float = 10.0,
) -> list[list[dict[str, Any]]]:
    """
    Split segments into overlapping time-windows.

    Kept to support unit-tests and as the batch-boundary splitter.
    In the new strategy, large batches contain *multiple* original chunks;
    this function's output is concatenated, not sent one-per-request.
    """
    chunks: list[list[dict[str, Any]]] = []
    current_chunk: list[dict[str, Any]] = []
    current_start: float | None = None

    for seg in segments:
        if not current_chunk:
            current_start = seg["start_time"]
            current_chunk.append(seg)
            continue

        duration = seg["end_time"] - current_start  # type: ignore[operator]
        if duration > max_duration:
            chunks.append(current_chunk)

            # Start new chunk with overlap
            overlap_target = seg["end_time"] - overlap
            new_chunk: list[dict[str, Any]] = []
            new_start: float | None = None
            for c_seg in current_chunk:
                if c_seg["end_time"] >= overlap_target:
                    if new_start is None:
                        new_start = c_seg["start_time"]
                    new_chunk.append(c_seg)

            if not new_chunk:
                new_start = seg["start_time"]

            current_chunk = new_chunk
            current_chunk.append(seg)
            current_start = new_start
        else:
            current_chunk.append(seg)

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# ---------------------------------------------------------------------------
# Batch builder
# ---------------------------------------------------------------------------


def _build_batches(segments: list[dict[str, Any]]) -> list[str]:
    """
    Render transcript segments into a list of plain-text strings, each
    within _MAX_CHARS_PER_BATCH.

    For almost every real video this returns a single string (1 batch).
    Falls back to splitting on segment boundaries if a video is
    extraordinarily long.
    """
    lines = [
        f"[{s['start_time']} - {s['end_time']}] {s['text']}" for s in segments
    ]

    batches: list[str] = []
    current_lines: list[str] = []
    current_len = 0

    for line in lines:
        line_len = len(line) + 1  # +1 for newline
        if current_lines and current_len + line_len > _MAX_CHARS_PER_BATCH:
            batches.append("\n".join(current_lines))
            current_lines = []
            current_len = 0
            if len(batches) >= MAX_BATCHES:
                # Safety: merge all remaining lines into the last batch.
                break
        current_lines.append(line)
        current_len += line_len

    if current_lines:
        batches.append("\n".join(current_lines))

    return batches


# ---------------------------------------------------------------------------
# Gemini request (one per batch — typically ONE request per video)
# ---------------------------------------------------------------------------


async def _call_gemini(transcript_text: str, api_key: str) -> list[dict[str, Any]]:
    """
    Send a single generateContent request for the given transcript text block.
    Returns the parsed list of raw claim dicts, or [] on any error.
    """
    payload = {
        "contents": [
            {"parts": [{"text": _EXTRACT_PROMPT.format(transcript_text=transcript_text)}]}
        ],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
        },
    }

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            response = await client.post(
                GEMINI_GENERATE_URL,
                json=payload,
                params={"key": api_key},
            )
            response.raise_for_status()
            data = response.json()

        raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(raw_text)

        if not isinstance(parsed, list):
            logger.error("Gemini returned non-list JSON for claims: %s", type(parsed))
            return []

        return parsed

    except httpx.HTTPStatusError as exc:
        logger.error(
            "Gemini claim extraction HTTP %d: %s",
            exc.response.status_code,
            exc.response.text[:300],
        )
        return []
    except Exception as exc:
        logger.error("Gemini claim extraction failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Scoring helpers (pure Python, zero I/O)
# ---------------------------------------------------------------------------

_CLAIM_TYPE_MULTIPLIER: dict[str, float] = {
    "factual_claim": 1.0,
    "ambiguous": 0.5,
    "opinion": 0.1,
    "advertisement": 0.1,
}


def _score_claim(raw: dict[str, Any]) -> tuple[float, float]:
    """
    Return (checkability_score, claim_score) from raw extraction dict.
    Deterministic — no I/O.
    """
    has_num = bool(raw.get("has_numbers", False))
    has_ent = bool(raw.get("has_entities", False))
    is_spec = bool(raw.get("is_specific", False))

    checkability = 0.0
    if has_num:
        checkability += 0.35
    if has_ent:
        checkability += 0.35
    if is_spec:
        checkability += 0.30

    claim_type = raw.get("claim_type", "factual_claim")
    if claim_type not in _VALID_CLAIM_TYPES:
        claim_type = "ambiguous"

    # Rule-based override: the classifier keyword rules can flip the type
    # to "opinion" or "advertisement" for clear-cut cases — zero Gemini calls.
    rule_result = _classify_by_rules(raw.get("text", ""))
    if rule_result is not None:
        claim_type = rule_result.value

    multiplier = _CLAIM_TYPE_MULTIPLIER.get(claim_type, 0.5)
    claim_score = checkability * multiplier

    return checkability, claim_score, claim_type  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def process_video_claims(video_id: str, top_n: int = 12) -> list[CandidateClaim]:
    """
    Full Phase 5.4 pipeline for a single video.

    Gemini call budget:
      - 1 request  for the full transcript extraction + inline classification
      - 0 follow-up classification requests (rule-based override only)
    """
    # 1. Fetch transcript (Phase 5.3, synchronous)
    transcript_data = get_transcript(video_id)
    segments = transcript_data.get("segments", [])

    if not segments:
        logger.warning("No transcript segments for video %s", video_id)
        return []

    api_key = settings.GEMINI_API_KEY.get_secret_value()
    if not api_key:
        logger.warning("GEMINI_API_KEY not set — cannot extract claims.")
        return []

    # 2. Build batches (almost always exactly 1)
    batches = _build_batches(segments)
    logger.info(
        "Video %s: %d segments → %d batch(es) to Gemini",
        video_id,
        len(segments),
        len(batches),
    )

    # 3. Send batches sequentially (avoids parallel 429s on free tier)
    raw_claims: list[dict[str, Any]] = []
    for i, batch_text in enumerate(batches):
        logger.info("Sending batch %d/%d to Gemini", i + 1, len(batches))
        results = await _call_gemini(batch_text, api_key)
        raw_claims.extend(results)

    if not raw_claims:
        logger.warning("No claims extracted for video %s", video_id)
        return []

    logger.info("Extracted %d raw candidates for video %s", len(raw_claims), video_id)

    # 4. Score & build CandidateClaim objects (pure Python, zero I/O)
    processed_claims: list[CandidateClaim] = []
    for raw in raw_claims:
        text = raw.get("text", "").strip()
        if not text:
            continue

        start_time = float(raw.get("start_time", 0))
        end_time = float(raw.get("end_time", 0))

        checkability, claim_score, claim_type = _score_claim(raw)

        processed_claims.append(
            CandidateClaim(
                text=text,
                start_time=start_time,
                end_time=end_time,
                claim_type=claim_type,
                checkability_score=checkability,
                claim_score=claim_score,
            )
        )

    # 5. Deduplicate (same SequenceMatcher logic, threshold 0.8)
    unique_claims: list[CandidateClaim] = []
    for claim in processed_claims:
        is_dup = False
        for u_claim in unique_claims:
            ratio = SequenceMatcher(
                None, claim.text.lower(), u_claim.text.lower()
            ).ratio()
            if ratio > 0.8:
                is_dup = True
                # Keep strongest wording
                if len(claim.text) > len(u_claim.text):
                    u_claim.text = claim.text
                # Merge timestamps
                u_claim.start_time = min(u_claim.start_time, claim.start_time)
                u_claim.end_time = max(u_claim.end_time, claim.end_time)
                u_claim.claim_score = max(u_claim.claim_score, claim.claim_score)
                break
        if not is_dup:
            unique_claims.append(claim)

    logger.info(
        "After dedup: %d unique claims for video %s", len(unique_claims), video_id
    )

    # 6. Rank by claim_score descending
    unique_claims.sort(key=lambda x: x.claim_score, reverse=True)

    # 7. Top-N
    return unique_claims[:top_n]
