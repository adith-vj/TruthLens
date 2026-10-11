"""
models/visual_video.py — Pydantic models for visual fake/deepfake video analysis.

Defines request/response schemas for analyzing video frames, video files,
and video streams without audio dependencies.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class VideoFrameInput(BaseModel):
    """A single video frame captured at a specific timestamp."""
    timestamp: float = Field(..., description="Timestamp in seconds from video start", ge=0.0)
    image_base64: str = Field(..., description="Base64-encoded JPEG/PNG image data")


class VisualFramesAnalysisRequest(BaseModel):
    """Payload containing extracted video frames sent from client canvas."""
    video_id: Optional[str] = Field(None, description="Optional YouTube or platform video identifier")
    title: Optional[str] = Field(None, description="Optional video title")
    frames: List[VideoFrameInput] = Field(..., min_length=1, max_length=60, description="List of video frames to analyze")


class VisualUrlAnalysisRequest(BaseModel):
    """Request to analyze a video from a direct URL."""
    video_url: str = Field(..., description="HTTP/HTTPS or direct streaming video URL")
    max_frames: Optional[int] = Field(16, ge=4, le=48, description="Maximum number of frames to sample")


class FrameAnalysisResult(BaseModel):
    """Detailed forensic analysis for a single video frame."""
    timestamp: float = Field(..., description="Timestamp in seconds")
    formatted_time: str = Field(..., description="Formatted timestamp MM:SS")
    fake_score: float = Field(..., description="Per-frame composite fake score 0.0 - 1.0", ge=0.0, le=1.0)
    face_detected: bool = Field(False, description="Whether a face was detected in this frame")
    face_count: int = Field(0, description="Number of detected faces")
    face_score: Optional[float] = Field(None, description="Calibrated face deepfake score if face present")
    scene_ai_score: float = Field(..., description="Full scene generative AI score")
    fft_anomaly_score: float = Field(..., description="Frequency domain anomaly score")
    ela_anomaly_score: float = Field(..., description="Error Level Analysis compression score")
    boundary_seam_score: Optional[float] = Field(default=None, description="Face boundary seam blending score if face present")
    temporal_jitter: float = Field(0.0, description="Difference from adjacent frame")
    flagged: bool = Field(False, description="True if frame exceeds fake suspicion thresholds")
    signals: List[str] = Field(default_factory=list, description="Forensic signals triggered for this frame")


class VisualForensicBreakdown(BaseModel):
    """Aggregate risk indicators across forensic dimensions."""
    face_deepfake_risk: float = Field(..., ge=0.0, le=1.0)
    boundary_seam_risk: float = Field(default=0.0, ge=0.0, le=1.0, description="Face mask boundary seam and blending risk")
    scene_ai_generation_risk: float = Field(..., ge=0.0, le=1.0)
    frequency_artifacts_risk: float = Field(..., ge=0.0, le=1.0)
    compression_splicing_risk: float = Field(..., ge=0.0, le=1.0)
    temporal_inconsistency_risk: float = Field(..., ge=0.0, le=1.0)


class VisualVideoAnalysisResponse(BaseModel):
    """Final visual fake video analysis verdict and timeline."""
    status: str = "success"
    verdict: str = Field(..., description="DEEPFAKE | SUSPICIOUS | AUTHENTIC")
    confidence_score: float = Field(..., description="Verdict confidence percentage 0.50 - 0.99")
    composite_score: float = Field(..., description="Overall fake probability 0.0 - 1.0")
    analyzed_frames_count: int = Field(..., description="Total number of frames analyzed")
    duration_seconds: Optional[float] = Field(None, description="Estimated video duration in seconds")
    flagged_segments: List[str] = Field(default_factory=list, description="List of time windows with detected anomalies")
    timeline: List[FrameAnalysisResult] = Field(..., description="Chronological timeline of frame results")
    forensic_breakdown: VisualForensicBreakdown = Field(..., description="Scores for individual forensic signals")
    explanation: str = Field(..., description="Human-readable forensic findings explanation")
