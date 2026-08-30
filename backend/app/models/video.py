from typing import List, Any
from pydantic import BaseModel, Field
from app.models.verification import SourceItem

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

class VideoAnalyzeRequest(BaseModel):
    video_id: str = Field(..., description="The YouTube video ID to analyze and verify claims for.")

class VideoUsageMetrics(BaseModel):
    """Accumulated API usage counts for a video analysis job."""
    google_factcheck_calls: int = 0
    gemini_first_pass_calls: int = 0
    gemini_evidence_calls: int = 0
    tavily_calls: int = 0

class VideoAnalysisResult(BaseModel):
    text: str
    start_time: float
    end_time: float
    claim_type: str
    claim_score: float
    checkability_score: float
    status: str = Field(..., description="Status of this specific claim verification (pending, verified, unverifiable, error)")
    verdict: str | None = None
    confidence_score: float | None = None
    sources: List[SourceItem] = Field(default_factory=list)

class VideoAnalysisJobState(BaseModel):
    job_id: str
    video_id: str
    status: str = Field(..., description="Job status: pending, running, completed, failed")
    total_claims: int
    completed_claims: int
    failed_claims: int
    results: List[VideoAnalysisResult]
    created_at: float
    updated_at: float
    is_auto_generated: bool = False
    language: str = ""
    usage_metrics: VideoUsageMetrics = Field(default_factory=VideoUsageMetrics)


