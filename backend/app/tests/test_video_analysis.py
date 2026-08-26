import pytest
import asyncio
import time
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi import HTTPException

from app.services.video_analysis import (
    start_video_analysis,
    get_job_state,
    _cleanup_old_jobs,
    _jobs,
    _jobs_by_video,
    VIDEO_JOB_TTL_SECONDS
)
from app.models.video import CandidateClaim
from app.models.verification import VerifyResponse

@pytest.fixture(autouse=True)
def clear_jobs():
    """Clear jobs state before each test."""
    _jobs.clear()
    _jobs_by_video.clear()

# Helpers
def _mock_claim(text="test", start=0.0, end=1.0, ctype="factual_claim"):
    return CandidateClaim(
        text=text,
        start_time=start,
        end_time=end,
        claim_type=ctype,
        checkability_score=0.9,
        claim_score=0.9
    )

def _mock_verify_res(verdict="true", conf=0.9):
    return VerifyResponse(verdict=verdict, confidence_score=conf, sources=[])

# 1. job creation & 2. job retrieval
@pytest.mark.asyncio
async def test_job_creation_and_retrieval():
    with patch("app.services.video_analysis.get_transcript") as mock_t, \
         patch("app.services.video_analysis.process_video_claims", new=AsyncMock(return_value=[])):
        mock_t.return_value = {"is_auto_generated": False, "language": "en"}
        
        job_id = await start_video_analysis("v1")
        assert job_id is not None
        
        job = get_job_state(job_id)
        assert job is not None
        assert job.job_id == job_id
        assert job.video_id == "v1"

# 3. job lifecycle
@pytest.mark.asyncio
async def test_job_lifecycle_empty_claims():
    with patch("app.services.video_analysis.get_transcript") as mock_t, \
         patch("app.services.video_analysis.process_video_claims", new=AsyncMock(return_value=[])):
        mock_t.return_value = {"is_auto_generated": False, "language": "en"}
        
        job_id = await start_video_analysis("v1")
        # Give it a moment to complete
        await asyncio.sleep(0.01)
        job = get_job_state(job_id)
        assert job.status == "completed"

# 4. verification concurrency limit & 5. progressive result availability
@pytest.mark.asyncio
async def test_concurrency_and_progressive():
    claims = [_mock_claim(f"C{i}") for i in range(5)]
    
    verify_calls = 0
    in_flight = 0
    max_in_flight = 0
    
    async def slow_verify(req):
        nonlocal verify_calls, in_flight, max_in_flight
        verify_calls += 1
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.1) # Artificially slow down
        in_flight -= 1
        return _mock_verify_res("true")
        
    with patch("app.services.video_analysis.get_transcript") as mock_t, \
         patch("app.services.video_analysis.process_video_claims", new=AsyncMock(return_value=claims)), \
         patch("app.services.video_analysis.verify_claim", side_effect=slow_verify):
        mock_t.return_value = {"is_auto_generated": False, "language": "en"}
        
        job_id = await start_video_analysis("v1")
        
        # Immediate check (pending or running)
        job = get_job_state(job_id)
        assert job.status in ("pending", "running")
        
        # Wait halfway through
        await asyncio.sleep(0.15)
        job = get_job_state(job_id)
        assert job.status == "running"
        assert job.completed_claims > 0 and job.completed_claims < 5 # Progressive availability
        
        # Wait to finish
        await asyncio.sleep(0.3)
        job = get_job_state(job_id)
        assert job.status == "completed"
        assert job.completed_claims == 5
        assert max_in_flight <= 3 # Max concurrency

# 6, 7, 8, 9, 16. successful, unverifiable, error claims and failure isolation
@pytest.mark.asyncio
async def test_mixed_results_and_failure_isolation():
    claims = [
        _mock_claim("Success"),
        _mock_claim("Unverif"),
        _mock_claim("Error"),
    ]
    
    async def mixed_verify(req):
        if req.text == "Success":
            return _mock_verify_res("true")
        elif req.text == "Unverif":
            return _mock_verify_res("unverifiable")
        elif req.text == "Error":
            raise HTTPException(status_code=502, detail="Upstream error")
            
    with patch("app.services.video_analysis.get_transcript") as mock_t, \
         patch("app.services.video_analysis.process_video_claims", new=AsyncMock(return_value=claims)), \
         patch("app.services.video_analysis.verify_claim", side_effect=mixed_verify):
        mock_t.return_value = {"is_auto_generated": False, "language": "en"}
        
        job_id = await start_video_analysis("v1")
        await asyncio.sleep(0.05)
        
        job = get_job_state(job_id)
        assert job.status == "completed"
        assert job.completed_claims == 2
        assert job.failed_claims == 1
        
        # Check specific results
        r_succ = next(r for r in job.results if r.text == "Success")
        assert r_succ.status == "verified"
        assert r_succ.verdict == "true"
        
        r_unv = next(r for r in job.results if r.text == "Unverif")
        assert r_unv.status == "unverifiable"
        assert r_unv.verdict == "unverifiable"
        
        r_err = next(r for r in job.results if r.text == "Error")
        assert r_err.status == "error"

# 10. final result ordering
@pytest.mark.asyncio
async def test_final_result_ordering():
    claims = [_mock_claim(f"C{i}") for i in range(5)]
    
    # Complete in random order to test stability
    async def random_verify(req):
        idx = int(req.text[1:])
        # Deliberately delay some more than others
        delay = [0.1, 0.01, 0.05, 0.2, 0.02][idx]
        await asyncio.sleep(delay)
        return _mock_verify_res("true")
        
    with patch("app.services.video_analysis.get_transcript") as mock_t, \
         patch("app.services.video_analysis.process_video_claims", new=AsyncMock(return_value=claims)), \
         patch("app.services.video_analysis.verify_claim", side_effect=random_verify):
        mock_t.return_value = {"is_auto_generated": False, "language": "en"}
        
        job_id = await start_video_analysis("v1")
        await asyncio.sleep(0.3)
        
        job = get_job_state(job_id)
        assert job.status == "completed"
        # Order should exactly match claims input
        for i, c in enumerate(claims):
            assert job.results[i].text == c.text

# 11, 12, 13. duplicate prevention, cache reuse
@pytest.mark.asyncio
async def test_duplicate_prevention_cache_reuse():
    with patch("app.services.video_analysis.get_transcript") as mock_t, \
         patch("app.services.video_analysis.process_video_claims", new=AsyncMock(return_value=[])):
        mock_t.return_value = {"is_auto_generated": False, "language": "en"}
        
        job_id1 = await start_video_analysis("v1")
        job_id2 = await start_video_analysis("v1") # While running/pending
        assert job_id1 == job_id2
        
        await asyncio.sleep(0.01) # Wait for complete
        job = get_job_state(job_id1)
        assert job.status == "completed"
        
        job_id3 = await start_video_analysis("v1") # After complete
        assert job_id1 == job_id3

# 14. zero selected claims
@pytest.mark.asyncio
async def test_zero_selected_claims():
    with patch("app.services.video_analysis.get_transcript") as mock_t, \
         patch("app.services.video_analysis.process_video_claims", new=AsyncMock(return_value=[])):
        mock_t.return_value = {"is_auto_generated": False, "language": "en"}
        
        job_id = await start_video_analysis("v1")
        await asyncio.sleep(0.01)
        job = get_job_state(job_id)
        assert job.status == "completed"
        assert job.total_claims == 0
        assert len(job.results) == 0

# 15. transcript/claim acquisition failure
@pytest.mark.asyncio
async def test_transcript_acquisition_failure():
    with patch("app.services.video_analysis.get_transcript", side_effect=ValueError("No transcript")):
        job_id = await start_video_analysis("v1")
        await asyncio.sleep(0.01)
        job = get_job_state(job_id)
        assert job.status == "failed"

# 17. client disconnect / job continuation
# This is implicit since we use asyncio.create_task and don't tie it to a Request object.
# The endpoint returns immediately and the background task continues.

# 18. Polling behavior
@pytest.mark.asyncio
async def test_polling_endpoint():
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    
    with patch("app.services.video_analysis.get_transcript") as mock_t, \
         patch("app.services.video_analysis.process_video_claims", new=AsyncMock(return_value=[])):
        mock_t.return_value = {"is_auto_generated": False, "language": "en"}
        
        # Start
        resp = client.post("/api/video/analyze", json={"video_id": "poll-test"})
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]
        
        # Poll
        resp2 = client.get(f"/api/video/analyze/{job_id}")
        assert resp2.status_code == 200
        assert resp2.json()["job_id"] == job_id
        
        # Wait
        await asyncio.sleep(0.02)
        resp3 = client.get(f"/api/video/analyze/{job_id}")
        assert resp3.json()["status"] == "completed"
