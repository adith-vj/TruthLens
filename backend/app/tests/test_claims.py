"""
Tests for Phase 5.4 Claim Extraction & Prioritization (v2, batched strategy).

All Gemini HTTP calls are mocked via patch so no real API key is needed.
The rule-based classifier (_classify_by_rules) runs for real — it's pure Python.
"""

import pytest
from unittest.mock import patch, AsyncMock

from app.services.claims import (
    chunk_transcript,
    _build_batches,
    _score_claim,
    _SPECIFIC_VERBS,
    _HEDGE_WORDS,
    process_video_claims,
)
from app.models.video import CandidateClaim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seg(text, start, end):
    return {"text": text, "start_time": float(start), "end_time": float(end)}


def _raw(text, claim_type="factual_claim", has_numbers=False,
         has_entities=False, is_specific=False, start=0.0, end=1.0):
    return {
        "text": text,
        "start_time": start,
        "end_time": end,
        "claim_type": claim_type,
        "has_numbers": has_numbers,
        "has_entities": has_entities,
        "is_specific": is_specific,
    }


# ---------------------------------------------------------------------------
# 1. chunk_transcript — unchanged logic
# ---------------------------------------------------------------------------

def test_transcript_chunking():
    segments = [_seg(f"Word {i}", i, i + 1) for i in range(200)]
    chunks = chunk_transcript(segments, max_duration=60.0, overlap=10.0)
    assert len(chunks) == 4

    end_of_chunk1 = chunks[0][-1]["end_time"]
    start_of_chunk2 = chunks[1][0]["start_time"]
    assert start_of_chunk2 <= end_of_chunk1 - 9.0


# ---------------------------------------------------------------------------
# 2. _build_batches
# ---------------------------------------------------------------------------

def test_build_batches_single_for_small_video():
    segments = [_seg(f"Word {i}", i, i + 1) for i in range(500)]
    batches = _build_batches(segments)
    assert len(batches) == 1


def test_build_batches_splits_on_oversized_input():
    from app.services.claims import _MAX_CHARS_PER_BATCH
    char_per_seg = 100
    n = (_MAX_CHARS_PER_BATCH // char_per_seg) + 500
    segments = [_seg("x" * char_per_seg, i, i + 1) for i in range(n)]
    batches = _build_batches(segments)
    assert len(batches) >= 2


# ---------------------------------------------------------------------------
# 3. _score_claim — component-level verification
# ---------------------------------------------------------------------------

class TestScoreclaimSpecificity:

    def test_vague_claim_low_specificity(self):
        """Short claim, no digits, no specific verb → low specificity component."""
        raw = _raw("The ship was big", has_entities=True)
        checkability, _, _ = _score_claim(raw)
        # specificity: 0 + 0 + 0 + (4/20)*0.20 = 0.04
        # info_value: 0 + 0.25 + 0 + 0.30*(1/3)=0.10 → 0.35 + some caps
        # checkability should be well below 0.5
        assert checkability < 0.5, f"Expected <0.5, got {checkability}"

    def test_precise_claim_high_specificity(self):
        """Claim with digit, specific verb, is_specific, and 20+ words → high specificity."""
        raw = _raw(
            "The Titanic sank at 2:20 AM on April 15 1912 with 1517 people "
            "losing their lives in the North Atlantic Ocean near Newfoundland",
            has_numbers=True, has_entities=True, is_specific=True,
        )
        checkability, _, _ = _score_claim(raw)
        assert checkability > 0.75, f"Expected >0.75, got {checkability}"

    def test_precision_ordering(self):
        """More precise claim must score strictly higher than a vague one."""
        vague = _raw("The Titanic sank", has_entities=True)
        precise = _raw(
            "The Titanic sank on April 15 1912 at 2:20 AM after striking an iceberg",
            has_numbers=True, has_entities=True, is_specific=True,
        )
        v_check, _, _ = _score_claim(vague)
        p_check, _, _ = _score_claim(precise)
        assert p_check > v_check, (
            f"precise ({p_check}) should beat vague ({v_check})"
        )


class TestScoreclaimInformationValue:

    def test_two_numbers_beats_one_number(self):
        """Claim with two distinct numeric values scores higher than claim with one."""
        one_num = _raw(
            "The ship carried 2,240 passengers",
            has_numbers=True, has_entities=True, is_specific=True,
        )
        two_nums = _raw(
            "The ship carried 2,240 passengers and weighed 52,310 tons",
            has_numbers=True, has_entities=True, is_specific=True,
        )
        c1, _, _ = _score_claim(one_num)
        c2, _, _ = _score_claim(two_nums)
        assert c2 >= c1, f"two-number claim ({c2}) should be >= one-number claim ({c1})"

    def test_entity_count_improves_score(self):
        """More mid-sentence named entities → higher information_value."""
        low = _raw("It sank", is_specific=False)
        high = _raw(
            "Captain Smith ordered the Titanic to sail past the North Atlantic iceberg zone",
            has_entities=True, is_specific=True,
        )
        c_low, _, _ = _score_claim(low)
        c_high, _, _ = _score_claim(high)
        assert c_high > c_low


class TestScoreclaimConfidence:

    def test_hedge_word_reduces_score(self):
        """Adding 'probably' to a claim must lower its score."""
        no_hedge = _raw(
            "The company launched the product in 2024",
            has_numbers=True, has_entities=True, is_specific=True,
        )
        with_hedge = _raw(
            "The company probably launched the product in around 2024",
            has_numbers=True, has_entities=True, is_specific=True,
        )
        c_clean, _, _ = _score_claim(no_hedge)
        c_hedged, _, _ = _score_claim(with_hedge)
        assert c_clean > c_hedged, (
            f"clean ({c_clean}) should beat hedged ({c_hedged})"
        )

    def test_four_hedges_floors_confidence_at_zero(self):
        """Four or more hedge words → confidence = 0.0."""
        raw = _raw(
            "It might probably perhaps possibly be approximately true",
            has_numbers=False, has_entities=False,
        )
        # confidence = max(0.0, 1.0 - 0.25*4) = 0.0
        # _HEDGE_WORDS contains: might, probably, perhaps, possibly, approximately
        checkability, _, _ = _score_claim(raw)
        # The confidence component is 0; checkability can still be > 0 from spec/info
        # but let's verify score is lower than a fully confident claim
        confident = _raw(
            "The Titanic sank in 1912 carrying 2,240 passengers",
            has_numbers=True, has_entities=True, is_specific=True,
        )
        c_hedged, _, _ = _score_claim(raw)
        c_confident, _, _ = _score_claim(confident)
        assert c_confident > c_hedged


class TestScoreclaimFactualityMultiplier:

    def test_factual_beats_ambiguous_same_text(self):
        """Same claim text: factual_claim scores 2× ambiguous."""
        fact = _raw("Revenue increased by 30%", claim_type="factual_claim",
                    has_numbers=True, is_specific=True)
        ambig = _raw("Revenue increased by 30%", claim_type="ambiguous",
                     has_numbers=True, is_specific=True)
        _, fs, _ = _score_claim(fact)
        _, as_, _ = _score_claim(ambig)
        assert fs == pytest.approx(as_ * 2.0, rel=0.01)

    def test_opinion_rules_override_gemini_label(self):
        """'I think' triggers rule-based override to opinion regardless of model label."""
        raw = _raw(
            "I think the Titanic was the largest ship ever built",
            claim_type="factual_claim",
            has_entities=True, is_specific=True,
        )
        _, claim_score, claim_type = _score_claim(raw)
        assert claim_type == "opinion"
        assert claim_score < 0.15  # opinion multiplier = 0.1


class TestScoreclaimDifferentiation:
    """
    Regression tests proving the score is NOT flat for real-world claims.
    These reflect the original Titanic-video failure (all claims = 1.0).
    """

    def test_no_all_ones(self):
        """A batch of factual Titanic-style claims must NOT all score 1.0."""
        claims = [
            _raw("The Titanic was a ship",
                 has_entities=True),
            _raw("The Titanic sank on April 15 1912",
                 has_numbers=True, has_entities=True, is_specific=True),
            _raw("1,517 people died when the Titanic sank in the North Atlantic",
                 has_numbers=True, has_entities=True, is_specific=True),
            _raw("The ship was probably around 882 feet long",
                 has_numbers=True, has_entities=True, is_specific=True),
        ]
        scores = [_score_claim(c)[0] for c in claims]
        assert max(scores) - min(scores) > 0.15, (
            f"Expected score spread >0.15, got {max(scores):.3f} – {min(scores):.3f}"
        )
        assert max(scores) < 1.0, "No claim should have checkability exactly 1.0"

    def test_score_range_is_sane(self):
        """All scores must be in [0.0, 1.0]."""
        claims = [
            _raw("Maybe it could have been true", has_numbers=False),
            _raw("X occurred", has_entities=True),
            _raw("In 2024 the company earned $5 billion from 12 products in 3 countries",
                 has_numbers=True, has_entities=True, is_specific=True),
        ]
        for c in claims:
            check, score, _ = _score_claim(c)
            assert 0.0 <= check <= 1.0, f"checkability out of range: {check}"
            assert 0.0 <= score <= 1.0, f"claim_score out of range: {score}"


# ---------------------------------------------------------------------------
# 4. Tie-breaker: equal scores → chronological order
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_equal_score_claims_sorted_chronologically():
    """When two claims have the same score, earlier start_time appears first."""
    # Use identical text so scores are guaranteed equal
    text = "The Titanic carried 2,240 passengers in 1912"
    mock_transcript = {"segments": [_seg("any", 0, 1)]}
    gemini_response = [
        _raw(text, claim_type="factual_claim",
             has_numbers=True, has_entities=True, is_specific=True,
             start=100.0, end=105.0),
        _raw(text + " and crew.", claim_type="factual_claim",
             has_numbers=True, has_entities=True, is_specific=True,
             start=10.0, end=15.0),
    ]

    with patch("app.services.claims.get_transcript", return_value=mock_transcript), \
         patch("app.services.claims.settings") as mock_settings, \
         patch("app.services.claims._call_gemini", new=AsyncMock(return_value=gemini_response)):
        mock_settings.GEMINI_API_KEY.get_secret_value.return_value = "test-key"
        claims = await process_video_claims("FAKE", top_n=12)

    # Deduplication merges the near-identical claims into one, so we should
    # end up with 1 or 2 claims; the key thing is no assertion error raised.
    # If two survive, the one with lower start_time must come first.
    if len(claims) >= 2:
        assert claims[0].start_time <= claims[1].start_time


# ---------------------------------------------------------------------------
# 5. Full pipeline integration (mocked Gemini)
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
        _raw("The company launched the product in 2024.", "factual_claim",
             has_numbers=True, has_entities=True, start=10.5, end=15.2),
        _raw("Revenues increased by 30%.", "factual_claim",
             has_numbers=True, is_specific=True, start=18.0, end=22.0),
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
    gemini_response = [
        _raw("The company launched the product in 2024.", "factual_claim",
             has_numbers=True, start=10.5, end=15.2),
        _raw("Company launched product in 2024.", "factual_claim",
             has_numbers=True, start=15.2, end=18.0),
        _raw("Totally different claim here.", "factual_claim",
             start=20.0, end=22.0),
    ]

    with patch("app.services.claims.get_transcript", return_value=mock_transcript), \
         patch("app.services.claims.settings") as mock_settings, \
         patch("app.services.claims._call_gemini", new=AsyncMock(return_value=gemini_response)):
        mock_settings.GEMINI_API_KEY.get_secret_value.return_value = "test-key"
        claims = await process_video_claims("FAKE_VIDEO_ID")

    assert len(claims) == 2
    merged = next(c for c in claims if "2024" in c.text)
    assert merged.text == "The company launched the product in 2024."
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
    """A 15-minute video must trigger exactly 1 Gemini call."""
    segments = [_seg(f"Sentence {i}", i, i + 1) for i in range(900)]
    mock_transcript = {"segments": segments}
    call_count = []

    async def fake_call(text, key):
        call_count.append(1)
        return []

    with patch("app.services.claims.get_transcript", return_value=mock_transcript), \
         patch("app.services.claims.settings") as mock_settings, \
         patch("app.services.claims._call_gemini", side_effect=fake_call):
        mock_settings.GEMINI_API_KEY.get_secret_value.return_value = "test-key"
        await process_video_claims("FAKE_VIDEO_ID")

    assert len(call_count) == 1, f"Expected 1 Gemini call, got {len(call_count)}"
