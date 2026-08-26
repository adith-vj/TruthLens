from typing import List
from pydantic import BaseModel, Field

class VideoTranscriptRequest(BaseModel):
    video_id: str = Field(..., description="The YouTube video ID to fetch the transcript for.")

class TranscriptSegment(BaseModel):
    text: str
    start_time: float
    end_time: float

class VideoTranscriptResponse(BaseModel):
    video_id: str
    platform: str
    source: str
    language: str
    language_code: str
    is_auto_generated: bool
    segments: List[TranscriptSegment]

class VideoClaimsRequest(BaseModel):
    video_id: str = Field(..., description="The YouTube video ID to analyze claims for.")

class CandidateClaim(BaseModel):
    text: str
    start_time: float
    end_time: float
    claim_type: str
    checkability_score: float
    claim_score: float

class VideoClaimsResponse(BaseModel):
    video_id: str
    total_candidates: int
    selected_claims: List[CandidateClaim]
