import logging
from typing import Dict, Any

import asyncio
from fastapi import APIRouter, HTTPException, status, UploadFile, File

from app.models.video import (
    VideoTranscriptRequest, 
    VideoTranscriptResponse, 
    VideoClaimsRequest, 
    VideoClaimsResponse,
    VideoAnalyzeRequest
)
from app.services.youtube_transcript import get_transcript

logger = logging.getLogger(__name__)

router = APIRouter()

# Very basic in-memory cache for V1
# Keys: video_id, Values: Transcript Response Dict
_transcript_cache: Dict[str, Any] = {}

@router.post("/transcript", response_model=VideoTranscriptResponse)
async def fetch_transcript(request: VideoTranscriptRequest):
    """
    Fetches, normalizes, and caches the transcript for a given YouTube video ID.
    """
    video_id = request.video_id
    
    if video_id in _transcript_cache:
        logger.info(f"Cache hit for transcript of video {video_id}")
        return _transcript_cache[video_id]
        
    logger.info(f"Fetching transcript for video {video_id}")
    
    try:
        # Run synchronous fetching logic
        # get_transcript fetches and normalizes
        result = get_transcript(video_id)
        
        # Cache the result
        _transcript_cache[video_id] = result
        return result
        
    except ValueError as ve:
        # e.g., "No transcript available for this video."
        logger.warning(f"Transcript unavailable for {video_id}: {ve}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(ve)
        )
    except Exception as e:
        logger.error(f"Failed to fetch transcript for {video_id}", exc_info=True)
        # We don't expose stack traces to the frontend
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve transcript due to an internal error."
        )

# In-memory cache for claim extraction
_claims_cache: Dict[str, Any] = {}

@router.post("/claims", response_model=VideoClaimsResponse)
async def extract_video_claims(request: VideoClaimsRequest):
    """
    Extracts, classifies, scores, deduplicates and ranks claims from a video transcript.
    """
    from app.services.claims import process_video_claims
    
    video_id = request.video_id
    
    if video_id in _claims_cache:
        logger.info(f"Cache hit for claims of video {video_id}")
        return _claims_cache[video_id]
        
    logger.info(f"Extracting claims for video {video_id}")
    
    try:
        # Run extraction
        top_claims = await process_video_claims(video_id, top_n=12)
        
        result = VideoClaimsResponse(
            video_id=video_id,
            total_candidates=len(top_claims),
            selected_claims=top_claims
        )
        
        _claims_cache[video_id] = result
        return result
        
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(ve)
        )
    except Exception as e:
        logger.error(f"Failed to extract claims for {video_id}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to extract claims due to an internal error."
        )

@router.post("/analyze", response_model=Dict[str, str])
async def analyze_video(request: VideoAnalyzeRequest):
    """
    Initializes a progressive video analysis job.
    Returns the job_id which can be polled for state and results.
    """
    from app.services.video_analysis import start_video_analysis
    try:
        job_id = await start_video_analysis(request.video_id)
        return {"job_id": job_id}
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(ve)
        )
    except Exception as e:
        logger.error(f"Failed to start video analysis for {request.video_id}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start video analysis."
        )

from app.models.video import VideoAnalysisJobState

@router.get("/analyze/{job_id}", response_model=VideoAnalysisJobState)
async def get_analysis_status(job_id: str):
    """
    Returns the current state of a video analysis job.
    """
    from app.services.video_analysis import get_job_state
    
    job = get_job_state(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found."
        )
        
    return job
        
@router.get("/analyze/video/{video_id}", response_model=VideoAnalysisJobState)
async def get_analysis_by_video(video_id: str):
    """
    Returns the active or completed job for a video_id, if one exists.
    """
    from app.services.video_analysis import _jobs_by_video, get_job_state
    
    if video_id not in _jobs_by_video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No job found for this video."
        )
        
    job_id = _jobs_by_video[video_id]
    job = get_job_state(job_id)
    if not job or job.status == "failed":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No valid job found for this video."
        )
        
    return job


# ---------------------------------------------------------------------------
# Visual Fake / Deepfake Video Analysis (No Audio / Silent Videos)
# ---------------------------------------------------------------------------
from app.models.visual_video import (
    VisualFramesAnalysisRequest,
    VisualUrlAnalysisRequest,
    VisualVideoAnalysisResponse,
)
from app.services.visual_video_analysis import (
    analyze_video_frames,
    decode_base64_image,
    extract_frames_from_video_bytes,
    extract_frames_from_video_file,
)


@router.post("/analyze-visual-frames", response_model=VisualVideoAnalysisResponse)
async def analyze_visual_frames_endpoint(request: VisualFramesAnalysisRequest):
    """
    Analyzes an array of client-extracted video frames without audio.
    Used by browser extension canvas frame capture on any web video.
    """
    try:
        frames = []
        for item in request.frames:
            pil_img = decode_base64_image(item.image_base64)
            frames.append((item.timestamp, pil_img))

        if not frames:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid frames provided for analysis.",
            )

        result = await asyncio.to_thread(analyze_video_frames, frames)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Visual frames analysis failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Visual video analysis failed: {str(e)}",
        )


@router.post("/analyze-visual-file", response_model=VisualVideoAnalysisResponse)
async def analyze_visual_file_endpoint(
    file: UploadFile = File(...), max_frames: int = 16
):
    """
    Upload a silent or audio-free video file (.mp4, .webm, .mov, .avi)
    for multi-signal visual deepfake detection.
    """
    ext = file.filename.split(".")[-1].lower() if file.filename else ""
    if ext not in ["mp4", "webm", "mov", "avi", "mkv"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported video format. Please upload MP4, WebM, MOV, or AVI.",
        )

    try:
        content = await file.read()
        if len(content) > 50 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Video file exceeds maximum size limit (50MB).",
            )

        def _process():
            extracted, dur = extract_frames_from_video_bytes(
                content, max_frames=max_frames
            )
            return analyze_video_frames(extracted, total_duration=dur)

        result = await asyncio.to_thread(_process)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Visual video file analysis failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Video processing failed: {str(e)}",
        )


@router.post("/analyze-visual-url", response_model=VisualVideoAnalysisResponse)
async def analyze_visual_url_endpoint(request: VisualUrlAnalysisRequest):
    """
    Analyzes a video from URL (direct stream or YouTube URL via yt-dlp snippet).
    """
    url = request.video_url.strip()
    max_frames = request.max_frames or 16

    try:
        def _process():
            # If YouTube URL, fetch short snippet via yt-dlp
            if "youtube.com" in url or "youtu.be" in url:
                import tempfile
                import os
                import yt_dlp

                with tempfile.TemporaryDirectory() as tmp_dir:
                    out_template = os.path.join(tmp_dir, "snippet.%(ext)s")
                    ydl_opts = {
                        "format": "bestvideo[height<=480]/worstvideo/worst/best",
                        "outtmpl": out_template,
                        "quiet": True,
                        "no_warnings": True,
                    }
                    try:
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            ydl.download([url])
                    except Exception as yt_err:
                        logger.warning("yt-dlp primary download failed, retrying generic: %s", yt_err)
                        ydl_opts_fallback = {
                            "format": "worst/best",
                            "outtmpl": out_template,
                            "quiet": True,
                            "no_warnings": True,
                        }
                        with yt_dlp.YoutubeDL(ydl_opts_fallback) as ydl:
                            ydl.download([url])

                    # Find downloaded file
                    files = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if os.path.isfile(os.path.join(tmp_dir, f))]
                    if not files:
                        raise ValueError("yt-dlp could not download video from URL.")

                    video_file = files[0]
                    extracted, dur = extract_frames_from_video_file(video_file, max_frames=max_frames)
                    return analyze_video_frames(extracted, total_duration=dur)
            else:
                extracted, dur = extract_frames_from_video_file(url, max_frames=max_frames)
                return analyze_video_frames(extracted, total_duration=dur)

        result = await asyncio.to_thread(_process)
        return result
    except Exception as e:
        logger.error("Visual URL analysis failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze video from URL: {str(e)}",
        )


