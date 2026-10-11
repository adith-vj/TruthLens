"""
tests/test_visual_video.py — Automated tests for the visual fake video analysis service and API.
"""

from __future__ import annotations

import base64
from io import BytesIO
from unittest.mock import MagicMock

import numpy as np
import pytest
from httpx import AsyncClient
from PIL import Image

from app.models.visual_video import VisualFramesAnalysisRequest, VideoFrameInput
from app.services.visual_video_analysis import (
    analyze_single_frame,
    analyze_video_frames,
    compute_frame_difference,
    decode_base64_image,
    format_timestamp,
)


def _create_dummy_image(color=(128, 128, 128), size=(100, 100)) -> Image.Image:
    """Create a plain solid color PIL image for testing."""
    return Image.new("RGB", size, color)


def _image_to_base64(image: Image.Image) -> str:
    """Convert PIL image to base64 data URI."""
    buf = BytesIO()
    image.save(buf, format="JPEG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def test_format_timestamp():
    assert format_timestamp(0.0) == "0:00"
    assert format_timestamp(65.0) == "1:05"
    assert format_timestamp(134.2) == "2:14"


def test_decode_base64_image():
    img = _create_dummy_image((255, 0, 0))
    b64 = _image_to_base64(img)
    decoded = decode_base64_image(b64)
    assert decoded.size == (100, 100)
    assert decoded.mode == "RGB"


def test_compute_frame_difference():
    img1 = _create_dummy_image((0, 0, 0))
    img2 = _create_dummy_image((0, 0, 0))
    # Identical images should have zero difference
    diff_identical = compute_frame_difference(img1, img2)
    assert diff_identical == pytest.approx(0.0, abs=1e-3)

    # Black vs White should have significant difference
    img3 = _create_dummy_image((255, 255, 255))
    diff_extreme = compute_frame_difference(img1, img3)
    assert diff_extreme > 0.5


def test_compute_face_boundary_seam():
    from app.services.visual_video_analysis import compute_face_boundary_seam
    # Solid image has uniform edge and color
    uniform_img = _create_dummy_image((128, 128, 128), size=(160, 160))
    seam_score = compute_face_boundary_seam(uniform_img)
    assert 0.0 <= seam_score <= 1.0


def test_analyze_video_frames_authentic(monkeypatch):
    """Test analysis when models indicate authentic content."""
    # Mock deepfake model to return AUTHENTIC
    mock_engine = MagicMock()
    mock_engine.return_value = [{"label": "REAL", "score": 0.95}]
    
    from app.services import legacy_models
    monkeypatch.setattr(legacy_models, "get_deepfake_model", lambda: mock_engine)
    monkeypatch.setattr(legacy_models, "get_face_detector", lambda: None)

    frames = [
        (0.0, _create_dummy_image((100, 100, 100))),
        (2.0, _create_dummy_image((102, 102, 102))),
        (4.0, _create_dummy_image((104, 104, 104))),
    ]

    response = analyze_video_frames(frames, total_duration=4.0)
    assert response.status == "success"
    assert response.verdict == "AUTHENTIC"
    assert response.confidence_score >= 0.50
    assert response.analyzed_frames_count == 3
    assert len(response.timeline) == 3


def test_analyze_video_frames_deepfake(monkeypatch):
    """Test analysis when models indicate synthetic AI generation."""
    # Mock deepfake model to return FAKE
    mock_engine = MagicMock()
    mock_engine.return_value = [{"label": "FAKE", "score": 0.98}]
    
    from app.services import legacy_models
    monkeypatch.setattr(legacy_models, "get_deepfake_model", lambda: mock_engine)
    monkeypatch.setattr(legacy_models, "get_face_detector", lambda: None)

    frames = [
        (0.0, _create_dummy_image((100, 100, 100))),
        (3.0, _create_dummy_image((200, 50, 50))),
        (6.0, _create_dummy_image((50, 200, 50))),
    ]

    response = analyze_video_frames(frames, total_duration=6.0)
    assert response.status == "success"
    assert response.verdict == "DEEPFAKE"
    assert response.composite_score >= 0.55
    assert len(response.flagged_segments) > 0


@pytest.mark.asyncio
async def test_api_analyze_visual_frames(async_client: AsyncClient, monkeypatch):
    """Test POST /api/video/analyze-visual-frames endpoint with JSON payload."""
    mock_engine = MagicMock()
    mock_engine.return_value = [{"label": "REAL", "score": 0.92}]
    from app.services import legacy_models
    monkeypatch.setattr(legacy_models, "get_deepfake_model", lambda: mock_engine)
    monkeypatch.setattr(legacy_models, "get_face_detector", lambda: None)

    img = _create_dummy_image((80, 80, 80))
    b64 = _image_to_base64(img)

    payload = {
        "video_id": "test_video_123",
        "title": "Silent Test Video",
        "frames": [
            {"timestamp": 0.0, "image_base64": b64},
            {"timestamp": 2.5, "image_base64": b64},
        ],
    }

    res = await async_client.post("/api/video/analyze-visual-frames", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["verdict"] in ["AUTHENTIC", "SUSPICIOUS", "DEEPFAKE"]
    assert data["analyzed_frames_count"] == 2
    assert "timeline" in data
    assert "forensic_breakdown" in data


@pytest.mark.asyncio
async def test_api_analyze_visual_file(async_client: AsyncClient, monkeypatch, tmp_path):
    """Test POST /api/video/analyze-visual-file endpoint with an MP4 file upload."""
    import cv2
    mock_engine = MagicMock()
    mock_engine.return_value = [{"label": "REAL", "score": 0.90}]
    from app.services import legacy_models
    monkeypatch.setattr(legacy_models, "get_deepfake_model", lambda: mock_engine)
    monkeypatch.setattr(legacy_models, "get_face_detector", lambda: None)

    # Generate small temporary MP4 video
    video_path = str(tmp_path / "test_clip.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, 10, (128, 128))
    for i in range(15):
        frame = np.full((128, 128, 3), 100 + i * 2, dtype=np.uint8)
        out.write(frame)
    out.release()

    with open(video_path, "rb") as f:
        video_bytes = f.read()

    files = {"file": ("test_clip.mp4", video_bytes, "video/mp4")}
    res = await async_client.post("/api/video/analyze-visual-file", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["verdict"] in ["AUTHENTIC", "SUSPICIOUS", "DEEPFAKE"]
    assert data["analyzed_frames_count"] > 0
    assert len(data["timeline"]) > 0

