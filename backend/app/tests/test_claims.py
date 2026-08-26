"""
Tests for Phase 5.4 Claim Extraction & Prioritization (v2, batched strategy).

All Gemini HTTP calls are mocked via patch so no real API key is needed.
The rule-based classifier (_classify_by_rules) runs for real — it's pure Python.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from app.services.claims import (
    chunk_transcript,
    _build_batches,
    _score_claim,
    process_video_claims,
)
from app.models.video import CandidateClaim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seg(text, start, end):
    return {"text": text, "start_time": float(start), "end_time": float(end)}


def _raw_claim(text, start, end, claim_type="factual_claim",
               has_numbers=False, has_entities=False, is_specific=False):
    return {
        "text": text,
        "start_time": float(start),
        "end_time": float(end),
        "claim_type": claim_type,
        "has_numbers": has_numbers,
        "has_entities": has_entities,
        "is_specific": is_specific,
    }


# ---------------------------------------------------------------------------
# 1. chunk_transcript — unchanged logic
# ---------------------------------------------------------------------------

def test_transcript_chunking():
    # 200 segments of 1 second each → 200 seconds total
    segments = [_seg(f"Word {i}", i, i + 1) for i in range(200)]
    chunks = chunk_transcript(segments, max_duration=60.0, overlap=10.0)
    # With 200s and 60s windows (10s overlap), expect 4 chunks
    assert len(chunks) == 4

    # Chunk 2 must overlap with end of chunk 1 (overlap ≥ 9 s)
    end_of_chunk1 = chunks[0][-1]["end_time"]
    start_of_chunk2 = chunks[1][0]["start_time"]
    assert start_of_chunk2 <= end_of_chunk1 - 9.0


# ---------------------------------------------------------------------------
# 2. _build_batches — single batch for normal videos
# ---------------------------------------------------------------------------

def test_build_batches_single_for_small_video():
    segments = [_seg(f"Word {i}", i, i + 1) for i in range(500)]
    batches = _build_batches(segments)
    # 500 segments × ~25 chars each ≈ 12500 chars — well within 1 batch
    assert len(batches) == 1


def test_build_batches_splits_on_oversized_input():
    # Create segments whose total text exceeds _MAX_CHARS_PER_BATCH
    from app.services.claims import _MAX_CHARS_PER_BATCH
    char_per_seg = 100
    n = (_MAX_CHARS_PER_BATCH // char_per_seg) + 500
    segments = [_seg("x" * char_per_seg, i, i + 1) for i in range(n)]
    batches = _build_batches(segments)
    assert len(batches) >= 2


# ---------------------------------------------------------------------------
# 3. _score_claim — deterministic, no I/O
# ---------------------------------------------------------------------------

def test_score_claim_high_checkability():
    raw = _raw_claim("A", 0, 1, claim_type="factual_claim",
                     has_numbers=True, has_entities=True, is_specific=True)
    checkability, claim_score, claim_type = _score_claim(raw)
    assert checkability == pytest.approx(1.0)
    assert claim_score == pytest.approx(1.0)
    assert claim_type == "factual_claim"


def test_score_claim_ambiguous_penalty():
    raw = _raw_claim("B", 0, 1, claim_type="ambiguous",
                     has_numbers=False, has_entities=False, is_specific=False)
    checkability, claim_score, claim_type = _score_claim(raw)
    assert checkability == pytest.approx(0.0)
    assert claim_score == pytest.approx(0.0)


def test_score_claim_opinion_penalty():
    # Rule-based override: "I think" → opinion regardless of model label
    raw = _raw_claim("I think the sky is blue", 0, 1, claim_type="factual_claim",
                     has_numbers=False, has_entities=True, is_specific=True)
    checkability, claim_score, claim_type = _score_claim(raw)
    # _classify_by_rules should catch "I think" and return OPINION
    assert claim_type == "opinion"
    assert claim_score == pytest.approx(checkability * 0.1)


# ---------------------------------------------------------------------------
# 4. process_video_claims — full pipeline with mocked Gemini
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_process_video_claims_happy_path():
    mock_transcript = {
        "segments": [
            _seg("The company launched the product in 2024.", 10.5, 15.2),
            _seg("Revenues increased by 30%.", 18.0, 22.0),
            _seg("I think it's great.", 22.0, 25.0),
        ]
    }

    gemini_response = [
        _raw_claim("The company launched the product in 2024.", 10.5, 15.2,
                   claim_type="factual_claim", has_numbers=True, has_entities=True),
        _raw_claim("Revenues increased by 30%.", 18.0, 22.0,
                   claim_type="factual_claim", has_numbers=True, is_specific=True),
    ]

    with patch("app.services.claims.get_transcript", return_value=mock_transcript), \
         patch("app.services.claims.settings") as mock_settings, \
         patch("app.services.claims._call_gemini", new=AsyncMock(return_value=gemini_response)):

        mock_settings.GEMINI_API_KEY.get_secret_value.return_value = "test-key"
        claims = await process_video_claims("FAKE_VIDEO_ID")

    assert len(claims) == 2
    texts = [c.text for c in claims]
    assert "Revenues increased by 30%." in texts
    assert "The company launched the product in 2024." in texts


@pytest.mark.asyncio
async def test_process_video_claims_deduplication():
    mock_transcript = {"segments": [_seg("x", 0, 1)]}

    # Two very similar claims — should be deduped
    gemini_response = [
        _raw_claim("The company launched the product in 2024.", 10.5, 15.2,
                   claim_type="factual_claim", has_numbers=True),
        _raw_claim("Company launched product in 2024.", 15.2, 18.0,
                   claim_type="factual_claim", has_numbers=True),
        _raw_claim("Totally different claim here.", 20.0, 22.0,
                   claim_type="factual_claim"),
    ]

    with patch("app.services.claims.get_transcript", return_value=mock_transcript), \
         patch("app.services.claims.settings") as mock_settings, \
         patch("app.services.claims._call_gemini", new=AsyncMock(return_value=gemini_response)):

        mock_settings.GEMINI_API_KEY.get_secret_value.return_value = "test-key"
        claims = await process_video_claims("FAKE_VIDEO_ID")

    assert len(claims) == 2  # deduped from 3 → 2
    merged = next(c for c in claims if "2024" in c.text)
    # Longest wording kept
    assert merged.text == "The company launched the product in 2024."
    # Timestamps merged
    assert merged.start_time == 10.5
    assert merged.end_time == 18.0


@pytest.mark.asyncio
async def test_process_video_claims_no_api_key():
    mock_transcript = {"segments": [_seg("anything", 0, 1)]}
    with patch("app.services.claims.get_transcript", return_value=mock_transcript), \
         patch("app.services.claims.settings") as mock_settings:
        mock_settings.GEMINI_API_KEY.get_secret_value.return_value = ""
        claims = await process_video_claims("FAKE_VIDEO_ID")
    assert claims == []


@pytest.mark.asyncio
async def test_process_video_claims_gemini_returns_empty():
    mock_transcript = {"segments": [_seg("anything", 0, 1)]}
    with patch("app.services.claims.get_transcript", return_value=mock_transcript), \
         patch("app.services.claims.settings") as mock_settings, \
         patch("app.services.claims._call_gemini", new=AsyncMock(return_value=[])):
        mock_settings.GEMINI_API_KEY.get_secret_value.return_value = "test-key"
        claims = await process_video_claims("FAKE_VIDEO_ID")
    assert claims == []


@pytest.mark.asyncio
async def test_process_video_claims_single_gemini_call_for_normal_video():
    """Verify that a normal 15-minute video triggers exactly 1 Gemini call."""
    # 15 minutes = 900 segments of 1 second each
    segments = [_seg(f"Sentence {i}", i, i + 1) for i in range(900)]
    mock_transcript = {"segments": segments}

    call_count = []

    async def fake_call_gemini(text, key):
        call_count.append(1)
        return []

    with patch("app.services.claims.get_transcript", return_value=mock_transcript), \
         patch("app.services.claims.settings") as mock_settings, \
         patch("app.services.claims._call_gemini", side_effect=fake_call_gemini):
        mock_settings.GEMINI_API_KEY.get_secret_value.return_value = "test-key"
        await process_video_claims("FAKE_VIDEO_ID")

    assert len(call_count) == 1, (
        f"Expected 1 Gemini call, got {len(call_count)}"
    )
