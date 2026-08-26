import logging
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.models.video import VideoTranscriptRequest, VideoTranscriptResponse, VideoClaimsRequest, VideoClaimsResponse
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
