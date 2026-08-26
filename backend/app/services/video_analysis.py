import asyncio
import time
import uuid
import logging
from typing import Dict
from fastapi import HTTPException
from pydantic import BaseModel

from app.models.video import CandidateClaim, VideoAnalysisJobState, VideoAnalysisResult
from app.models.verification import VerifyRequest, VerifyResponse
from app.api.verify import verify_claim
from app.services.claims import process_video_claims
from app.services.youtube_transcript import get_transcript

logger = logging.getLogger(__name__)

# TTL caching
VIDEO_JOB_TTL_SECONDS = 3600
_jobs: Dict[str, VideoAnalysisJobState] = {}
_jobs_by_video: Dict[str, str] = {} # video_id -> job_id

def _cleanup_old_jobs():
    """Removes jobs that are completed or failed and older than TTL."""
    now = time.time()
    to_delete = []
    for j_id, job in _jobs.items():
        if job.status in ("completed", "failed") and (now - job.updated_at > VIDEO_JOB_TTL_SECONDS):
            to_delete.append(j_id)
            
    for j_id in to_delete:
        v_id = _jobs[j_id].video_id
        if _jobs_by_video.get(v_id) == j_id:
            del _jobs_by_video[v_id]
        del _jobs[j_id]

async def start_video_analysis(video_id: str) -> str:
    """
    Initializes a new video analysis job or returns an existing one.
    Starts background verification of extracted claims.
    """
    _cleanup_old_jobs()
    
    if video_id in _jobs_by_video:
        j_id = _jobs_by_video[video_id]
        if _jobs[j_id].status in ("pending", "running", "completed"):
            return j_id
            
    job_id = str(uuid.uuid4())
    job = VideoAnalysisJobState(
        job_id=job_id,
        video_id=video_id,
        status="pending",
        total_claims=0,
        completed_claims=0,
        failed_claims=0,
        results=[],
        created_at=time.time(),
        updated_at=time.time()
    )
    
    _jobs[job_id] = job
    _jobs_by_video[video_id] = job_id
    
    # Start the analysis pipeline in the background
    asyncio.create_task(_run_video_analysis(job_id, video_id))
    
    return job_id

async def _run_video_analysis(job_id: str, video_id: str):
    """Background task to extract and verify claims."""
    job = _jobs.get(job_id)
    if not job:
        return
        
    job.status = "running"
    job.updated_at = time.time()
    
    try:
        # We need video metadata (is_auto_generated, language)
        transcript_data = get_transcript(video_id)
        job.is_auto_generated = transcript_data.get("is_auto_generated", False)
        job.language = transcript_data.get("language", "")
        
        # 1. Phase 5.4 - Extract Claims
        claims = await process_video_claims(video_id, top_n=3)
        if not claims:
            job.status = "completed"
            job.updated_at = time.time()
            return
            
        job.total_claims = len(claims)
        
        # 2. Setup job results in original order
        for c in claims:
            job.results.append(VideoAnalysisResult(
                text=c.text,
                start_time=c.start_time,
                end_time=c.end_time,
                claim_type=c.claim_type,
                claim_score=c.claim_score,
                checkability_score=c.checkability_score,
                status="pending"
            ))
        job.updated_at = time.time()
        
        # 3. Verify each claim independently with concurrency limit
        sem = asyncio.Semaphore(3) # MAX_VIDEO_VERIFICATIONS_CONCURRENT = 3
        
        async def verify_single_claim(index: int, claim: CandidateClaim):
            async with sem:
                res = job.results[index]
                try:
                    # Invoke the existing verify endpoint logic directly
                    v_req = VerifyRequest(text=claim.text)
                    v_res: VerifyResponse = await verify_claim(v_req)
                    
                    if v_res.verdict == "unverifiable":
                        res.status = "unverifiable"
                    else:
                        res.status = "verified"
                        
                    res.verdict = v_res.verdict
                    res.confidence_score = v_res.confidence_score
                    res.sources = v_res.sources
                    job.completed_claims += 1
                except HTTPException as e:
                    res.status = "error"
                    job.failed_claims += 1
                    logger.error(f"HTTPException verifying claim (Job: {job_id}, Index: {index}): {e.detail}")
                except Exception as e:
                    res.status = "error"
                    job.failed_claims += 1
                    logger.error(f"Error verifying claim (Job: {job_id}, Index: {index}): {e}", exc_info=True)
                finally:
                    job.updated_at = time.time()

        tasks = [verify_single_claim(i, c) for i, c in enumerate(claims)]
        await asyncio.gather(*tasks)
        
        job.status = "completed"
        job.updated_at = time.time()
        
    except Exception as e:
        logger.error(f"Video analysis job {job_id} failed: {e}", exc_info=True)
        job.status = "failed"
        job.updated_at = time.time()

def get_job_state(job_id: str) -> VideoAnalysisJobState | None:
    return _jobs.get(job_id)
