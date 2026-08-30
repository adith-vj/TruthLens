"""
tests/test_video_verify.py - Tests for Phase 5.5 video-specific verification.

All external services are mocked. No real API calls are made.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

from app.services.video_verify import (
    verify_video_claim,
    should_skip_tavily,
    _is_temporal_claim,
    _is_niche_claim,
    FirstPassResult,
    UsageMetrics,
    clear_verify_cache,
    _parse_first_pass_response,
    _gemini_first_pass,
)
from app.models.video import CandidateClaim
from app.models.verification import SourceItem
from app.services.factcheck import FactCheckMatch, FactCheckConfigError, FactCheckAuthError, FactCheckQuotaError
from app.services.search import SearchResult
from app.services.llm import LLMVerdict
from pydantic import AnyUrl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _claim(
    text="The Titanic sank in 1912.",
    ctype="factual_claim",
    checkability=0.80,
    claim_score=0.80,
    start=10.0,
    end=20.0,
):
    return CandidateClaim(
        text=text,
        claim_type=ctype,
        checkability_score=checkability,
        claim_score=claim_score,
        start_time=start,
        end_time=end,
    )


def _fc_match(verdict="true", conf=0.9):
    return FactCheckMatch(
        verdict=verdict,
        confidence_score=conf,
        sources=[SourceItem(title="FC", url=AnyUrl("https://fc.example.com"), publisher="FC")],
        raw_rating="true",
        publisher="FactCheck.org",
    )


def _search_result(title="Evidence", url="https://ev.example.com", snippet="Supporting text."):
    return SearchResult(title=title, url=AnyUrl(url), snippet=snippet, publisher="ev.example.com")


def _llm_verdict(verdict="true", conf=0.85, source_indices=None):
    indices = [0] if source_indices is None else source_indices
    return LLMVerdict(verdict=verdict, confidence_score=conf, source_indices=indices)



def _first_pass(verdict="true", confidence=0.85, needs_web=False, reason="Well-established historical fact."):
    return FirstPassResult(verdict=verdict, confidence=confidence, needs_web_search=needs_web, reason=reason)


@pytest.fixture(autouse=True)
def clear_cache():
    clear_verify_cache()
    yield
    clear_verify_cache()


# ---------------------------------------------------------------------------
# Temporal heuristic tests
# ---------------------------------------------------------------------------

def test_temporal_current():
    assert _is_temporal_claim("The current president is X.") is True

def test_temporal_currently():
    assert _is_temporal_claim("The bridge is currently under construction.") is True

def test_temporal_latest():
    assert _is_temporal_claim("The latest report shows 5% growth.") is True

def test_temporal_today():
    assert _is_temporal_claim("Today the company announced layoffs.") is True

def test_temporal_recent():
    assert _is_temporal_claim("A recent study found the drug effective.") is True

def test_temporal_recently():
    assert _is_temporal_claim("Tesla recently launched a new model.") is True

def test_temporal_as_of():
    assert _is_temporal_claim("As of last month, the rate was 3.5%.") is True

def test_temporal_this_year():
    assert _is_temporal_claim("This year revenue hit a record.") is True

def test_temporal_this_month():
    assert _is_temporal_claim("This month inflation rose 0.2%.") is True

def test_temporal_this_week():
    assert _is_temporal_claim("This week the bill passed.") is True

def test_temporal_ongoing():
    assert _is_temporal_claim("The ongoing conflict has displaced millions.") is True

def test_temporal_recent_year(monkeypatch):
    from datetime import datetime
    # Pretend current year = 2025; "2024" should trigger
    monkeypatch.setattr("app.services.video_verify.datetime", type("FakeDT", (), {"now": staticmethod(lambda: type("FN", (), {"year": 2025})())}))
    # Import fresh to test with monkeypatched datetime - test via direct regex
    from app.services.video_verify import _YEAR_PATTERN
    import re
    matches = [int(m.group()) for m in _YEAR_PATTERN.finditer("In 2024 the company posted losses.")]
    assert 2024 in matches

def test_temporal_old_year_not_triggered():
    # "1912" is historical — should NOT trigger the year heuristic
    assert _is_temporal_claim("The Titanic sank in 1912 killing 1,517 people.") is False

def test_temporal_no_trigger():
    assert _is_temporal_claim("The Titanic sank in 1912.") is False

def test_temporal_excluded_words():
    # "now", "just", "new", "newly" should NOT trigger
    assert _is_temporal_claim("Now the new model is available.") is False
    assert _is_temporal_claim("Just announced — the newly elected mayor.") is False


# ---------------------------------------------------------------------------
# Niche heuristic tests
# ---------------------------------------------------------------------------

def test_niche_above_threshold():
    assert _is_niche_claim(0.91) is True

def test_niche_at_threshold():
    assert _is_niche_claim(0.90) is True

def test_niche_below_threshold():
    assert _is_niche_claim(0.89) is False

def test_niche_ordinary_score():
    assert _is_niche_claim(0.75) is False


# ---------------------------------------------------------------------------
# should_skip_tavily tests
# ---------------------------------------------------------------------------

def test_skip_tavily_all_conditions():
    """All conditions met → skip Tavily."""
    fp = _first_pass(confidence=0.85, needs_web=False)
    assert should_skip_tavily("The Titanic sank in 1912.", fp, 0.80) is True

def test_skip_tavily_low_confidence():
    """Confidence < 0.80 → do not skip."""
    fp = _first_pass(confidence=0.75, needs_web=False)
    assert should_skip_tavily("The Titanic sank in 1912.", fp, 0.80) is False

def test_skip_tavily_needs_web():
    """Gemini says needs_web_search=True → do not skip."""
    fp = _first_pass(confidence=0.90, needs_web=True)
    assert should_skip_tavily("The Titanic sank in 1912.", fp, 0.80) is False

def test_skip_tavily_temporal_overrides():
    """Temporal claim → do not skip even at high confidence."""
    fp = _first_pass(confidence=0.95, needs_web=False)
    assert should_skip_tavily("The current CEO is Tim Cook.", fp, 0.80) is False

def test_skip_tavily_niche_overrides():
    """High checkability (niche) → do not skip even at high confidence."""
    fp = _first_pass(confidence=0.92, needs_web=False)
    assert should_skip_tavily("The Titanic sank in 1912.", fp, 0.91) is False

def test_skip_tavily_uncertain_verdict():
    """Uncertain verdict but otherwise eligible — should still skip (uncertain → unverifiable in final)."""
    fp = _first_pass(verdict="uncertain", confidence=0.85, needs_web=False)
    # needs_web=False and conf>=0.80 and no temporal/niche → skip
    assert should_skip_tavily("The Titanic sank in 1912.", fp, 0.80) is True


# ---------------------------------------------------------------------------
# verify_video_claim integration tests (all external services mocked)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_factcheck_hit_skips_gemini_and_tavily():
    """Test 1: Google Fact Check hit → 0 Gemini, 0 Tavily."""
    with patch("app.services.video_verify.verify_claim_factcheck", return_value=_fc_match()) as fc_mock, \
         patch("app.services.video_verify._gemini_first_pass") as gfp_mock, \
         patch("app.services.video_verify.search_evidence") as se_mock:
        result = await verify_video_claim(_claim(), video_id="v1")
        fc_mock.assert_awaited_once()
        gfp_mock.assert_not_called()
        se_mock.assert_not_called()
        assert result.verdict == "true"
        assert result.metrics.google_factcheck_calls == 1
        assert result.metrics.gemini_first_pass_calls == 0
        assert result.metrics.tavily_calls == 0


@pytest.mark.asyncio
async def test_no_factcheck_gemini_first_pass_runs():
    """Test 2: No Fact Check match → Gemini first-pass runs."""
    with patch("app.services.video_verify.verify_claim_factcheck", return_value=None), \
         patch("app.services.video_verify._gemini_first_pass", return_value=_first_pass()) as gfp_mock, \
         patch("app.services.video_verify.search_evidence") as se_mock:
        result = await verify_video_claim(_claim(), video_id="v1")
        gfp_mock.assert_awaited_once()
        assert result.metrics.gemini_first_pass_calls == 1


@pytest.mark.asyncio
async def test_high_confidence_no_temporal_skips_tavily():
    """Test 3: conf>=0.80, eligible claim → Tavily skipped."""
    with patch("app.services.video_verify.verify_claim_factcheck", return_value=None), \
         patch("app.services.video_verify._gemini_first_pass", return_value=_first_pass(confidence=0.88, needs_web=False)), \
         patch("app.services.video_verify.search_evidence") as se_mock:
        result = await verify_video_claim(_claim(checkability=0.80), video_id="v1")
        se_mock.assert_not_called()
        assert result.metrics.tavily_calls == 0
        assert result.verdict == "true"


@pytest.mark.asyncio
async def test_low_confidence_forces_tavily():
    """Test 4: conf<0.80 → Tavily required."""
    evidence = [_search_result()]
    with patch("app.services.video_verify.verify_claim_factcheck", return_value=None), \
         patch("app.services.video_verify._gemini_first_pass", return_value=_first_pass(confidence=0.70, needs_web=False)), \
         patch("app.services.video_verify.search_evidence", return_value=evidence), \
         patch("app.services.video_verify.verify_with_llm", return_value=_llm_verdict()):
        result = await verify_video_claim(_claim(checkability=0.80), video_id="v1")
        assert result.metrics.tavily_calls == 1


@pytest.mark.asyncio
async def test_temporal_claim_forces_tavily_even_high_confidence():
    """Test 5+6: Temporal claim → Tavily required despite high Gemini confidence."""
    temporal = _claim(text="The current president of the US is Joe Biden.")
    evidence = [_search_result()]
    with patch("app.services.video_verify.verify_claim_factcheck", return_value=None), \
         patch("app.services.video_verify._gemini_first_pass", return_value=_first_pass(confidence=0.92, needs_web=False)), \
         patch("app.services.video_verify.search_evidence", return_value=evidence), \
         patch("app.services.video_verify.verify_with_llm", return_value=_llm_verdict()):
        result = await verify_video_claim(temporal, video_id="v1")
        assert result.metrics.tavily_calls == 1


@pytest.mark.asyncio
async def test_niche_claim_forces_tavily():
    """Test 7: Niche claim (checkability>=0.90) → Tavily required."""
    evidence = [_search_result()]
    with patch("app.services.video_verify.verify_claim_factcheck", return_value=None), \
         patch("app.services.video_verify._gemini_first_pass", return_value=_first_pass(confidence=0.91, needs_web=False)), \
         patch("app.services.video_verify.search_evidence", return_value=evidence), \
         patch("app.services.video_verify.verify_with_llm", return_value=_llm_verdict()):
        result = await verify_video_claim(_claim(checkability=0.91), video_id="v1")
        assert result.metrics.tavily_calls == 1


@pytest.mark.asyncio
async def test_niche_heuristic_absent_confidence_threshold_still_operates():
    """Test 8: Below niche threshold, confidence threshold still enforced."""
    evidence = [_search_result()]
    with patch("app.services.video_verify.verify_claim_factcheck", return_value=None), \
         patch("app.services.video_verify._gemini_first_pass", return_value=_first_pass(confidence=0.79, needs_web=False)), \
         patch("app.services.video_verify.search_evidence", return_value=evidence), \
         patch("app.services.video_verify.verify_with_llm", return_value=_llm_verdict()):
        result = await verify_video_claim(_claim(checkability=0.80), video_id="v1")
        assert result.metrics.tavily_calls == 1  # Low confidence forced it


@pytest.mark.asyncio
async def test_gemini_uncertain_forces_tavily():
    """Test 9: Gemini uncertain verdict → needs_web_search=True → Tavily."""
    evidence = [_search_result()]
    with patch("app.services.video_verify.verify_claim_factcheck", return_value=None), \
         patch("app.services.video_verify._gemini_first_pass", return_value=_first_pass(verdict="uncertain", confidence=0.60, needs_web=True)), \
         patch("app.services.video_verify.search_evidence", return_value=evidence), \
         patch("app.services.video_verify.verify_with_llm", return_value=_llm_verdict()):
        result = await verify_video_claim(_claim(checkability=0.80), video_id="v1")
        assert result.metrics.tavily_calls == 1


@pytest.mark.asyncio
async def test_tavily_evidence_supports_true():
    """Test 10: Tavily evidence supports → true."""
    evidence = [_search_result()]
    with patch("app.services.video_verify.verify_claim_factcheck", return_value=None), \
         patch("app.services.video_verify._gemini_first_pass", return_value=_first_pass(confidence=0.60, needs_web=True)), \
         patch("app.services.video_verify.search_evidence", return_value=evidence), \
         patch("app.services.video_verify.verify_with_llm", return_value=_llm_verdict(verdict="true", conf=0.90)):
        result = await verify_video_claim(_claim(), video_id="v1")
        assert result.verdict == "true"
        assert result.metrics.gemini_evidence_calls == 1


@pytest.mark.asyncio
async def test_tavily_evidence_contradicts_false():
    """Test 11: Tavily evidence contradicts → false."""
    evidence = [_search_result()]
    with patch("app.services.video_verify.verify_claim_factcheck", return_value=None), \
         patch("app.services.video_verify._gemini_first_pass", return_value=_first_pass(confidence=0.60, needs_web=True)), \
         patch("app.services.video_verify.search_evidence", return_value=evidence), \
         patch("app.services.video_verify.verify_with_llm", return_value=_llm_verdict(verdict="false", conf=0.85, source_indices=[0])):
        result = await verify_video_claim(_claim(), video_id="v1")
        assert result.verdict == "false"


@pytest.mark.asyncio
async def test_empty_tavily_results_unverifiable():
    """Test 12: Tavily returns empty → unverifiable."""
    with patch("app.services.video_verify.verify_claim_factcheck", return_value=None), \
         patch("app.services.video_verify._gemini_first_pass", return_value=_first_pass(confidence=0.60, needs_web=True)), \
         patch("app.services.video_verify.search_evidence", return_value=[]), \
         patch("app.services.video_verify.verify_with_llm") as llm_mock:
        result = await verify_video_claim(_claim(), video_id="v1")
        assert result.verdict == "unverifiable"
        llm_mock.assert_not_called()  # LLM not called for empty evidence


@pytest.mark.asyncio
async def test_irrelevant_source_not_counted_in_sources():
    """Test 13: LLM returns empty source_indices → no sources in result."""
    evidence = [_search_result()]
    with patch("app.services.video_verify.verify_claim_factcheck", return_value=None), \
         patch("app.services.video_verify._gemini_first_pass", return_value=_first_pass(confidence=0.60, needs_web=True)), \
         patch("app.services.video_verify.search_evidence", return_value=evidence), \
         patch("app.services.video_verify.verify_with_llm", return_value=_llm_verdict(verdict="unverifiable", conf=0.3, source_indices=[])):
        result = await verify_video_claim(_claim(), video_id="v1")
        assert result.sources == []


@pytest.mark.asyncio
async def test_cached_result_not_verified_twice():
    """Test 14+15: Same claim same context → cached, no second API calls."""
    evidence = [_search_result()]
    call_count = {"n": 0}

    async def counting_fc(text):
        call_count["n"] += 1
        return None

    with patch("app.services.video_verify.verify_claim_factcheck", side_effect=counting_fc), \
         patch("app.services.video_verify._gemini_first_pass", return_value=_first_pass(confidence=0.85, needs_web=False)):
        r1 = await verify_video_claim(_claim(), video_id="v1", context="ctx")
        r2 = await verify_video_claim(_claim(), video_id="v1", context="ctx")
        # Only called once; second call used cache
        assert call_count["n"] == 1
        assert r1.verdict == r2.verdict


@pytest.mark.asyncio
async def test_pipeline_version_invalidates_cache():
    """Test 16: Changing pipeline version invalidates cache."""
    with patch("app.services.video_verify.verify_claim_factcheck", return_value=None), \
         patch("app.services.video_verify._gemini_first_pass", return_value=_first_pass(confidence=0.85, needs_web=False)), \
         patch("app.services.video_verify.search_evidence", return_value=[]):
        # Warm the cache
        r1 = await verify_video_claim(_claim(), video_id="v1", context="ctx")

    # Bump pipeline version
    with patch("app.core.config.settings.VIDEO_VERIFICATION_PIPELINE_VERSION", "v3"), \
         patch("app.services.video_verify.verify_claim_factcheck", return_value=None) as fc2, \
         patch("app.services.video_verify._gemini_first_pass", return_value=_first_pass(confidence=0.85, needs_web=False)):
        r2 = await verify_video_claim(_claim(), video_id="v1", context="ctx")
        # Should have re-run (different cache key)
        fc2.assert_awaited_once()


@pytest.mark.asyncio
async def test_video_context_passed_to_first_pass():
    """Test 17: Context string reaches Gemini first-pass."""
    captured = {}

    async def capture_fp(claim, context):
        captured["context"] = context
        return _first_pass()

    with patch("app.services.video_verify.verify_claim_factcheck", return_value=None), \
         patch("app.services.video_verify._gemini_first_pass", side_effect=capture_fp):
        await verify_video_claim(_claim(), video_id="v1", context="Timestamp: 1:30-1:45")
    assert "1:30-1:45" in captured.get("context", "")


@pytest.mark.asyncio
async def test_opinion_claim_skips_all_external_calls():
    """Test 18: opinion → early exit, no external calls."""
    with patch("app.services.video_verify.verify_claim_factcheck") as fc, \
         patch("app.services.video_verify._gemini_first_pass") as gfp, \
         patch("app.services.video_verify.search_evidence") as se:
        result = await verify_video_claim(_claim(ctype="opinion"), video_id="v1")
        fc.assert_not_called()
        gfp.assert_not_called()
        se.assert_not_called()
        assert result.verdict == "unverifiable"
        assert result.metrics.google_factcheck_calls == 0


@pytest.mark.asyncio
async def test_advertisement_claim_skips_all_external_calls():
    """Test 18b: advertisement → early exit."""
    with patch("app.services.video_verify.verify_claim_factcheck") as fc, \
         patch("app.services.video_verify._gemini_first_pass") as gfp:
        result = await verify_video_claim(_claim(ctype="advertisement"), video_id="v1")
        fc.assert_not_called()
        gfp.assert_not_called()


@pytest.mark.asyncio
async def test_manual_verify_route_unchanged():
    """Test 19: /api/verify still calls the old pipeline (Tavily→Gemini, no first-pass)."""
    from app.services.llm import verify_with_llm as real_verify_with_llm
    from app.services.search import search_evidence as real_search_evidence

    evidence = [_search_result()]
    with patch("app.api.verify.search_evidence", return_value=evidence) as se_mock, \
         patch("app.api.verify.verify_with_llm", return_value=_llm_verdict()) as llm_mock, \
         patch("app.api.verify.verify_claim_factcheck", return_value=None), \
         patch("app.api.verify.classify_claim", return_value=MagicMock(value="factual_claim")):
        from app.api.verify import verify_claim as route_verify_claim
        from app.models.verification import VerifyRequest
        # The route verify_claim is a FastAPI route function; call it directly
        req = VerifyRequest(text="The Titanic sank in 1912.")
        result = await route_verify_claim(req)
        # Must call Tavily (old path) — no first-pass
        se_mock.assert_awaited_once()
        llm_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_usage_metrics_count_correctly():
    """Test 20: Usage metrics count each call type correctly."""
    evidence = [_search_result()]
    with patch("app.services.video_verify.verify_claim_factcheck", return_value=None), \
         patch("app.services.video_verify._gemini_first_pass", return_value=_first_pass(confidence=0.60, needs_web=True)), \
         patch("app.services.video_verify.search_evidence", return_value=evidence), \
         patch("app.services.video_verify.verify_with_llm", return_value=_llm_verdict()):
        result = await verify_video_claim(_claim(), video_id="v1")
        m = result.metrics
        assert m.google_factcheck_calls == 1
        assert m.gemini_first_pass_calls == 1
        assert m.tavily_calls == 1
        assert m.gemini_evidence_calls == 1


@pytest.mark.asyncio
async def test_job_ttl_not_broken():
    """Test 21: Existing TTL behavior still works (jobs dict remains functional)."""
    from app.services.video_analysis import _jobs, _jobs_by_video, _cleanup_old_jobs, VIDEO_JOB_TTL_SECONDS
    import time
    from app.models.video import VideoAnalysisJobState, VideoUsageMetrics

    _jobs.clear()
    _jobs_by_video.clear()

    old_job = VideoAnalysisJobState(
        job_id="old-1",
        video_id="old-video",
        status="completed",
        total_claims=0,
        completed_claims=0,
        failed_claims=0,
        results=[],
        created_at=time.time() - VIDEO_JOB_TTL_SECONDS - 100,
        updated_at=time.time() - VIDEO_JOB_TTL_SECONDS - 100,
    )
    _jobs["old-1"] = old_job
    _jobs_by_video["old-video"] = "old-1"

    _cleanup_old_jobs()

    assert "old-1" not in _jobs
    assert "old-video" not in _jobs_by_video
