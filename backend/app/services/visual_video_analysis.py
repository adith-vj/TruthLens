"""
services/visual_video_analysis.py — Multi-signal visual fake video forensic engine.

Inspects video frames for deepfake face manipulation, synthetic/generative AI
visual artifacts (Sora/Runway/Kling/diffusion), frequency anomalies (FFT),
compression splicing (ELA), and inter-frame temporal discontinuities without
relying on audio or speech transcripts.
"""

from __future__ import annotations

import base64
import logging
import math
import os
import tempfile
from io import BytesIO
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from app.models.visual_video import (
    FrameAnalysisResult,
    VideoFrameInput,
    VisualForensicBreakdown,
    VisualVideoAnalysisResponse,
)
from app.services import legacy_models
from app.services.legacy_image import (
    calibrate_score,
    compute_ela_metrics,
    compute_fft_metrics,
    parse_model_scores,
    preprocess_image,
)

logger = logging.getLogger(__name__)


def format_timestamp(seconds: float) -> str:
    """Format seconds into MM:SS notation."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


def decode_base64_image(image_base64: str) -> Image.Image:
    """Decodes a base64 encoded image string into a PIL Image."""
    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]
    raw_bytes = base64.b64decode(image_base64)
    image = Image.open(BytesIO(raw_bytes))
    return preprocess_image(image)


def compute_face_boundary_seam(face_image: Image.Image) -> float:
    """
    Analyzes physical blending boundaries and mask seam discontinuities
    along the outer perimeter of a cropped face (jawline, forehead, hair boundary).
    Returns anomaly risk score 0.0 to 1.0.
    """
    try:
        img_np = np.array(face_image.resize((160, 160)), dtype=np.uint8)
        if img_np.ndim != 3 or img_np.shape[2] < 3:
            return 0.0

        bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Outer perimeter mask (outer 18% ring) vs inner facial core (inner 60%)
        h, w = gray.shape
        outer_mask = np.zeros((h, w), dtype=np.uint8)
        outer_mask[:int(h * 0.18), :] = 1
        outer_mask[int(h * 0.82):, :] = 1
        outer_mask[:, :int(w * 0.18)] = 1
        outer_mask[:, int(w * 0.82):] = 1

        inner_mask = np.zeros((h, w), dtype=np.uint8)
        inner_mask[int(h * 0.25):int(h * 0.75), int(w * 0.25):int(w * 0.75)] = 1

        # Laplacian sharpness difference
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        outer_sharpness = float(np.var(laplacian[outer_mask == 1]))
        inner_sharpness = float(np.var(laplacian[inner_mask == 1]))

        # Face swaps often exhibit blurred boundaries or cut lines
        sharpness_ratio = (inner_sharpness + 1e-5) / (outer_sharpness + 1e-5)
        sharpness_anomaly = 0.0
        if sharpness_ratio > 3.5:
            sharpness_anomaly = min(1.0, (sharpness_ratio - 3.5) / 5.0)
        elif sharpness_ratio < 0.25:
            sharpness_anomaly = min(1.0, (0.25 - sharpness_ratio) * 3.0)

        # Color difference: hue and saturation delta between boundary and center
        inner_h = np.mean(hsv[:, :, 0][inner_mask == 1])
        outer_h = np.mean(hsv[:, :, 0][outer_mask == 1])
        hue_delta = min(abs(inner_h - outer_h), 180 - abs(inner_h - outer_h)) / 90.0

        inner_s = np.mean(hsv[:, :, 1][inner_mask == 1]) / 255.0
        outer_s = np.mean(hsv[:, :, 1][outer_mask == 1]) / 255.0
        sat_delta = abs(inner_s - outer_s)

        color_anomaly = min(1.0, hue_delta * 0.6 + sat_delta * 0.7)
        seam_score = 0.55 * sharpness_anomaly + 0.45 * color_anomaly
        return round(float(np.clip(seam_score, 0.0, 1.0)), 4)
    except Exception as e:
        logger.debug("Boundary seam calculation error: %s", e)
        return 0.0


def extract_frames_from_video_file(
    file_path: str, max_frames: int = 16
) -> Tuple[List[Tuple[float, Image.Image]], float]:
    """
    Extracts motion- and face-adaptive sampled frames from a local video file using OpenCV.
    Returns: (list of (timestamp_seconds, PIL.Image), duration_seconds)
    """
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {file_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0.0:
            fps = 25.0
        duration = total_frames / fps if total_frames > 0 else 0.0

        if total_frames <= 0:
            # Fallback: sequential read if frame count unavailable
            frames = []
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % 15 == 0 and len(frames) < max_frames:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    ts = idx / fps
                    frames.append((round(ts, 2), Image.fromarray(rgb)))
                idx += 1
            return frames, round(idx / fps, 2)

        # Adaptive Motion-Aware Sampling:
        # If total frames exceed budget, sample candidates and pick dynamic/expressive frames
        if total_frames > max_frames * 2:
            candidate_indices = np.linspace(0, total_frames - 1, num=min(total_frames, max_frames * 3), dtype=int)
            candidate_data = []
            prev_small = None

            for c_idx in candidate_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(c_idx))
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 64))
                motion_score = 0.0
                if prev_small is not None:
                    motion_score = float(np.mean(np.abs(small.astype(np.float32) - prev_small.astype(np.float32))))
                prev_small = small
                candidate_data.append((c_idx, frame, motion_score))

            if len(candidate_data) > max_frames:
                bin_size = len(candidate_data) / max_frames
                selected_samples = []
                for b in range(max_frames):
                    start_i = int(b * bin_size)
                    end_i = int((b + 1) * bin_size)
                    bin_candidates = candidate_data[start_i:end_i]
                    if bin_candidates:
                        best = max(bin_candidates, key=lambda x: x[2])
                        selected_samples.append(best)
                candidate_data = selected_samples

            extracted_frames: List[Tuple[float, Image.Image]] = []
            for frame_idx, frame, _ in candidate_data:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                timestamp = round(float(frame_idx) / fps, 2)
                extracted_frames.append((timestamp, pil_img))
            return extracted_frames, round(duration, 2)

        # Standard linear distribution for short clips
        num_samples = min(max_frames, max(1, total_frames))
        sample_indices = np.linspace(0, total_frames - 1, num=num_samples, dtype=int)

        extracted_frames = []
        for frame_idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()
            if ret and frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                timestamp = round(float(frame_idx) / fps, 2)
                extracted_frames.append((timestamp, pil_img))

        return extracted_frames, round(duration, 2)
    finally:
        cap.release()


def extract_frames_from_video_bytes(
    video_bytes: bytes, max_frames: int = 16
) -> Tuple[List[Tuple[float, Image.Image]], float]:
    """Writes video bytes to a temporary file and extracts frames."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        return extract_frames_from_video_file(tmp_path, max_frames=max_frames)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception as e:
                logger.warning("Failed to remove temp video file: %s", e)


def detect_and_crop_faces(image: Image.Image) -> List[Image.Image]:
    """
    Detects faces in a PIL Image using MediaPipe BlazeFace.
    Returns cropped PIL face images with padding.
    """
    crops: List[Image.Image] = []
    detector = legacy_models.get_face_detector()
    if detector is None:
        return crops

    try:
        img_np = np.array(image, dtype=np.uint8)
        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
        result = detector.detect(mp_image)

        if not result or not result.detections:
            return crops

        width, height = image.size
        for det in result.detections:
            bb = det.bounding_box
            # Add 20% margin around face
            pad_x = bb.width * 0.2
            pad_y = bb.height * 0.2

            x1 = max(0, int(bb.origin_x - pad_x))
            y1 = max(0, int(bb.origin_y - pad_y))
            x2 = min(width, int(bb.origin_x + bb.width + pad_x))
            y2 = min(height, int(bb.origin_y + bb.height + pad_y))

            if x2 > x1 + 20 and y2 > y1 + 20:
                face_crop = image.crop((x1, y1, x2, y2))
                crops.append(face_crop)
    except Exception as e:
        logger.warning("Face detection crop error: %s", e)

    return crops


def compute_frame_difference(img1: Image.Image, img2: Image.Image) -> float:
    """
    Measures normalized difference between two consecutive frames
    to capture temporal flicker or sudden boundary jitter.
    """
    try:
        s1 = img1.resize((128, 128)).convert("L")
        s2 = img2.resize((128, 128)).convert("L")
        arr1 = np.array(s1, dtype=np.float32)
        arr2 = np.array(s2, dtype=np.float32)

        diff = np.abs(arr1 - arr2)
        mean_diff = float(np.mean(diff)) / 255.0
        # Normalize and clamp: values > 0.25 indicate sudden scene shifts or flicker
        return min(1.0, mean_diff * 3.0)
    except Exception:
        return 0.0


def analyze_single_frame(
    image: Image.Image,
    timestamp: float,
    prev_image: Optional[Image.Image] = None,
) -> FrameAnalysisResult:
    """Performs multi-signal visual forensic analysis on a single video frame."""
    signals: List[str] = []

    # Downsample for faster neural inference if huge
    w, h = image.size
    if max(w, h) > 768:
        scale = 768.0 / max(w, h)
        infer_img = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    else:
        infer_img = image

    # 1. Full-Scene Generative AI / Synthetic Detection
    deepfake_engine = legacy_models.get_deepfake_model()
    general_results = deepfake_engine(infer_img)
    general_scores = parse_model_scores(general_results, deepfake_engine)
    raw_scene_score = general_scores["fake_score"]
    scene_ai_score = calibrate_score(raw_scene_score, temperature=1.5) if raw_scene_score > 0.60 else raw_scene_score
    if scene_ai_score > 0.72:
        signals.append("synthetic_scene_pattern")

    # 2. Face Deepfake Detection & Boundary Seam Analysis
    face_crops = detect_and_crop_faces(image)
    face_detected = len(face_crops) > 0
    face_score: Optional[float] = None
    boundary_seam_score: Optional[float] = None

    if face_detected:
        face_engine = legacy_models.get_face_model()
        face_scores_list = []
        seam_scores_list = []
        for fc in face_crops[:3]:
            f_res = face_engine(fc)
            p_scores = parse_model_scores(f_res, face_engine)
            face_scores_list.append(p_scores["fake_score"])
            seam_scores_list.append(compute_face_boundary_seam(fc))

        if face_scores_list:
            face_score = max(face_scores_list)
            if face_score > 0.60:
                signals.append("face_deepfake_anomaly")
        if seam_scores_list:
            boundary_seam_score = max(seam_scores_list)
            if boundary_seam_score > 0.60:
                signals.append("face_mask_seam_anomaly")

    # 3. Spatial Frequency & Compression Forensics
    fft_res = compute_fft_metrics(infer_img)
    fft_anomaly = fft_res.get("anomaly_score", 0.0)
    if fft_anomaly > 0.80:
        signals.append("frequency_spectrum_anomaly")

    ela_res = compute_ela_metrics(infer_img)
    ela_anomaly = ela_res.get("anomaly_score", 0.0)
    if ela_anomaly > 0.75:
        signals.append("compression_splicing_anomaly")

    # 4. Temporal Jitter / Inter-frame variance
    temporal_jitter = 0.0
    if prev_image is not None:
        temporal_jitter = compute_frame_difference(image, prev_image)
        if temporal_jitter > 0.55:
            signals.append("temporal_motion_flicker")

    # 5. Composite Frame Score Calculation
    if face_detected and face_score is not None:
        effective_face_risk = face_score
        if boundary_seam_score is not None and boundary_seam_score > 0.50:
            effective_face_risk = max(face_score, 0.65 * face_score + 0.35 * boundary_seam_score)
        neural_weight = 0.35 * scene_ai_score + 0.65 * effective_face_risk
    else:
        neural_weight = scene_ai_score

    forensic_boost = 0.0
    if fft_anomaly > 0.75:
        forensic_boost += 0.03 * (fft_anomaly - 0.75)
    if ela_anomaly > 0.70:
        forensic_boost += 0.03 * (ela_anomaly - 0.70)
    if temporal_jitter > 0.50:
        forensic_boost += 0.04 * (temporal_jitter - 0.50)
    if boundary_seam_score is not None and boundary_seam_score > 0.55:
        forensic_boost += 0.05 * (boundary_seam_score - 0.55)

    frame_fake_score = max(0.0, min(1.0, neural_weight + forensic_boost))
    is_flagged = (frame_fake_score >= 0.60) or (face_score is not None and face_score >= 0.70) or (boundary_seam_score is not None and boundary_seam_score >= 0.75)

    return FrameAnalysisResult(
        timestamp=round(timestamp, 2),
        formatted_time=format_timestamp(timestamp),
        fake_score=round(frame_fake_score, 4),
        face_detected=face_detected,
        face_count=len(face_crops),
        face_score=round(face_score, 4) if face_score is not None else None,
        boundary_seam_score=round(boundary_seam_score, 4) if boundary_seam_score is not None else None,
        scene_ai_score=round(scene_ai_score, 4),
        fft_anomaly_score=round(fft_anomaly, 4),
        ela_anomaly_score=round(ela_anomaly, 4),
        temporal_jitter=round(temporal_jitter, 4),
        flagged=is_flagged,
        signals=signals,
    )


def analyze_video_frames(
    frames: List[Tuple[float, Image.Image]],
    total_duration: Optional[float] = None,
) -> VisualVideoAnalysisResponse:
    """
    Orchestrates high-efficiency batched analysis over an extracted frame sequence.
    Batches neural inferences across frames, measures face mask boundary seams,
    and applies sliding-window temporal burst aggregation.
    """
    if not frames:
        raise ValueError("No video frames provided for analysis.")

    # Sort frames chronologically by timestamp
    frames.sort(key=lambda item: item[0])

    # 1. Feature Pre-Extraction & Non-Neural Signal Computation (Parallelized across worker threads)
    from concurrent.futures import ThreadPoolExecutor

    # Pre-scale infer images and compute temporal jitter sequentially
    prepared_frames = []
    prev_img: Optional[Image.Image] = None
    for idx, (ts, img) in enumerate(frames):
        w, h = img.size
        if max(w, h) > 768:
            scale = 768.0 / max(w, h)
            infer_img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        else:
            infer_img = img
        temporal_jitter = compute_frame_difference(img, prev_img) if prev_img is not None else 0.0
        prev_img = img
        prepared_frames.append((idx, ts, img, infer_img, temporal_jitter))

    def _process_frame_signals(item):
        idx, ts, img, infer_img, temporal_jitter = item
        face_crops = detect_and_crop_faces(img)
        boundary_seams = [compute_face_boundary_seam(fc) for fc in face_crops]
        max_boundary_seam = max(boundary_seams) if boundary_seams else None

        fft_res = compute_fft_metrics(infer_img)
        fft_anomaly = fft_res.get("anomaly_score", 0.0)

        ela_res = compute_ela_metrics(infer_img)
        ela_anomaly = ela_res.get("anomaly_score", 0.0)

        return {
            "idx": idx,
            "timestamp": ts,
            "infer_img": infer_img,
            "face_crops": face_crops,
            "face_count": len(face_crops),
            "boundary_seam": max_boundary_seam,
            "fft_anomaly": fft_anomaly,
            "ela_anomaly": ela_anomaly,
            "temporal_jitter": temporal_jitter,
            "signals": [],
        }

    # Execute CPU signal analysis concurrently across available threads
    max_workers = min(8, len(frames))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        frame_metas = list(executor.map(_process_frame_signals, prepared_frames))

    # Sort back by original frame order
    frame_metas.sort(key=lambda m: m["idx"])
    all_infer_images = [m["infer_img"] for m in frame_metas]

    face_crop_tracking = []
    for m in frame_metas:
        for fc in m["face_crops"][:3]:
            face_crop_tracking.append((m["idx"], fc))

    # 2. Batched Scene-Level Generative AI Inference
    deepfake_engine = legacy_models.get_deepfake_model()
    scene_batch_results = deepfake_engine(all_infer_images, batch_size=8)
    if scene_batch_results and isinstance(scene_batch_results, list) and isinstance(scene_batch_results[0], dict):
        scene_batch_results = [scene_batch_results]

    for idx, meta in enumerate(frame_metas):
        raw_res = scene_batch_results[idx] if idx < len(scene_batch_results) else (scene_batch_results[-1] if scene_batch_results else [])
        scores = parse_model_scores(raw_res, deepfake_engine)
        raw_score = scores["fake_score"]
        # Conservative calibration: compress high scores toward the center.
        # ViT models can be overconfident (>0.90) on hard negatives. Apply a
        # temperature-scaled sigmoid with T=1.5 to reduce false positives.
        score = calibrate_score(raw_score, temperature=1.5) if raw_score > 0.60 else raw_score
        meta["scene_ai_score"] = score
        # Only flag as synthetic scene if score is well above noise floor
        if score > 0.72:
            meta["signals"].append("synthetic_scene_pattern")

    # 3. Batched Face Deepfake Inference
    from collections import defaultdict
    frame_face_scores = defaultdict(list)
    if face_crop_tracking:
        face_engine = legacy_models.get_face_model()
        all_crops_list = [crop for _, crop in face_crop_tracking]
        face_batch_results = face_engine(all_crops_list, batch_size=8)
        if face_batch_results and isinstance(face_batch_results, list) and isinstance(face_batch_results[0], dict):
            face_batch_results = [face_batch_results]

        for crop_idx, (target_idx, _) in enumerate(face_crop_tracking):
            raw_f_res = face_batch_results[crop_idx] if crop_idx < len(face_batch_results) else (face_batch_results[-1] if face_batch_results else [])
            f_scores = parse_model_scores(raw_f_res, face_engine)
            frame_face_scores[target_idx].append(f_scores["fake_score"])

    # 4. Construct Frame Analysis Timeline
    timeline: List[FrameAnalysisResult] = []
    for idx, meta in enumerate(frame_metas):
        f_scores = frame_face_scores[idx]
        face_detected = meta["face_count"] > 0
        face_score = max(f_scores) if f_scores else None
        scene_score = meta["scene_ai_score"]
        boundary_seam = meta["boundary_seam"]
        fft_anomaly = meta["fft_anomaly"]
        ela_anomaly = meta["ela_anomaly"]
        temporal_jitter = meta["temporal_jitter"]
        signals = meta["signals"]

        if face_score is not None and face_score > 0.60:
            signals.append("face_deepfake_anomaly")
        if boundary_seam is not None and boundary_seam > 0.60:
            signals.append("face_mask_seam_anomaly")
        if fft_anomaly > 0.80:
            signals.append("frequency_spectrum_anomaly")
        if ela_anomaly > 0.75:
            signals.append("compression_splicing_anomaly")
        if temporal_jitter > 0.55:
            signals.append("temporal_motion_flicker")

        # Composite frame calculation
        if face_detected and face_score is not None:
            effective_face_risk = face_score
            if boundary_seam is not None and boundary_seam > 0.50:
                effective_face_risk = max(face_score, 0.65 * face_score + 0.35 * boundary_seam)
            neural_weight = 0.35 * scene_score + 0.65 * effective_face_risk
        else:
            # When no face is detected (e.g. landscape, objects, Sora/Runway AI footage),
            # scene_score drives the visual decision directly.
            neural_weight = scene_score

        forensic_boost = 0.0
        if fft_anomaly > 0.75:
            forensic_boost += 0.03 * (fft_anomaly - 0.75)
        if ela_anomaly > 0.70:
            forensic_boost += 0.03 * (ela_anomaly - 0.70)
        if temporal_jitter > 0.50:
            forensic_boost += 0.04 * (temporal_jitter - 0.50)
        if boundary_seam is not None and boundary_seam > 0.55:
            forensic_boost += 0.05 * (boundary_seam - 0.55)

        frame_fake_score = max(0.0, min(1.0, neural_weight + forensic_boost))
        is_flagged = (frame_fake_score >= 0.60) or (face_score is not None and face_score >= 0.70) or (boundary_seam is not None and boundary_seam >= 0.75)

        timeline.append(FrameAnalysisResult(
            timestamp=round(meta["timestamp"], 2),
            formatted_time=format_timestamp(meta["timestamp"]),
            fake_score=round(frame_fake_score, 4),
            face_detected=face_detected,
            face_count=meta["face_count"],
            face_score=round(face_score, 4) if face_score is not None else None,
            boundary_seam_score=round(boundary_seam, 4) if boundary_seam is not None else None,
            scene_ai_score=round(scene_score, 4),
            fft_anomaly_score=round(fft_anomaly, 4),
            ela_anomaly_score=round(ela_anomaly, 4),
            temporal_jitter=round(temporal_jitter, 4),
            flagged=is_flagged,
            signals=signals,
        ))

    # 5. Aggregate Forensic Dimensions
    face_scores_all = [f.face_score for f in timeline if f.face_score is not None]
    avg_face_risk = float(np.mean(face_scores_all)) if face_scores_all else 0.0

    seam_scores_all = [f.boundary_seam_score for f in timeline if f.boundary_seam_score is not None]
    avg_seam_risk = float(np.mean(seam_scores_all)) if seam_scores_all else 0.0

    scene_ai_scores = [f.scene_ai_score for f in timeline]
    avg_scene_ai = float(np.mean(scene_ai_scores)) if scene_ai_scores else 0.0

    fft_scores = [f.fft_anomaly_score for f in timeline]
    avg_fft = float(np.mean(fft_scores)) if fft_scores else 0.0

    ela_scores = [f.ela_anomaly_score for f in timeline]
    avg_ela = float(np.mean(ela_scores)) if ela_scores else 0.0

    jitter_scores = [f.temporal_jitter for f in timeline]
    avg_temporal = float(np.mean(jitter_scores)) if jitter_scores else 0.0

    forensic_breakdown = VisualForensicBreakdown(
        face_deepfake_risk=round(avg_face_risk, 4),
        boundary_seam_risk=round(avg_seam_risk, 4),
        scene_ai_generation_risk=round(avg_scene_ai, 4),
        frequency_artifacts_risk=round(avg_fft, 4),
        compression_splicing_risk=round(avg_ela, 4),
        temporal_inconsistency_risk=round(avg_temporal, 4),
    )

    # 6. Sliding-Window Temporal Burst Aggregation
    frame_scores = [f.fake_score for f in timeline]
    mean_frame_score = float(np.mean(frame_scores))
    flagged_count = sum(1 for f in timeline if f.flagged)
    flagged_ratio = flagged_count / len(timeline)

    sorted_scores = sorted(frame_scores, reverse=True)
    top_k = max(1, int(math.ceil(len(sorted_scores) * 0.35)))
    top_k_mean = float(np.mean(sorted_scores[:top_k]))

    if len(frame_scores) >= 3:
        rolling_scores = [float(np.mean(frame_scores[i:i+3])) for i in range(len(frame_scores) - 2)]
        peak_burst_score = max(rolling_scores)
    else:
        peak_burst_score = max(frame_scores) if frame_scores else 0.0

    composite_score = round(
        0.40 * peak_burst_score + 0.35 * top_k_mean + 0.25 * mean_frame_score, 4
    )

    # Identify continuous flagged segments
    flagged_segments: List[str] = []
    seg_start: Optional[float] = None
    seg_end: Optional[float] = None

    for item in timeline:
        if item.flagged:
            if seg_start is None:
                seg_start = item.timestamp
            seg_end = item.timestamp
        else:
            if seg_start is not None and seg_end is not None:
                flagged_segments.append(
                    f"{format_timestamp(seg_start)} - {format_timestamp(seg_end)}"
                )
                seg_start = None
                seg_end = None
    if seg_start is not None and seg_end is not None:
        flagged_segments.append(
            f"{format_timestamp(seg_start)} - {format_timestamp(seg_end)}"
        )

    # Determine Verdict with tighter thresholds to avoid false 'SUSPICIOUS' clustering
    if composite_score >= 0.60 or flagged_ratio >= 0.40 or (avg_face_risk >= 0.65):
        verdict = "DEEPFAKE"
        conf = 0.55 + ((composite_score - 0.60) / 0.40) * 0.40
    elif composite_score >= 0.48 and flagged_count >= 2:
        verdict = "SUSPICIOUS"
        conf = 0.50 + ((composite_score - 0.48) / 0.12) * 0.25
    else:
        verdict = "AUTHENTIC"
        conf = 0.55 + ((0.48 - composite_score) / 0.48) * 0.40

    confidence_score = round(max(0.50, min(0.99, conf)), 2)

    # Human-readable explanation
    reasons = []
    if verdict == "DEEPFAKE":
        reasons.append(
            f"Visual video forensics detected synthetic manipulation with {int(confidence_score * 100)}% confidence"
        )
        if avg_face_risk > 0.50:
            reasons.append(f"Facial manipulation risk is elevated ({avg_face_risk:.2f})")
        if avg_seam_risk > 0.50:
            reasons.append(f"Face mask boundary seam anomalies detected ({avg_seam_risk:.2f})")
        if avg_scene_ai > 0.50:
            reasons.append(f"Generative AI scene patterns detected ({avg_scene_ai:.2f})")
        if flagged_segments:
            reasons.append(f"Anomalies localized at timestamps: {', '.join(flagged_segments)}")
    elif verdict == "SUSPICIOUS":
        reasons.append(
            f"Video exhibits minor visual inconsistencies ({int(confidence_score * 100)}% confidence)"
        )
        if avg_seam_risk > 0.45:
            reasons.append(f"Subtle face mask blending inconsistencies observed ({avg_seam_risk:.2f})")
        if flagged_segments:
            reasons.append(f"Suspicious variations observed at {', '.join(flagged_segments)}")
        if avg_temporal > 0.35:
            reasons.append(f"Temporal frame jitter detected ({avg_temporal:.2f})")
    else:
        reasons.append(
            f"Video frames demonstrate natural visual continuity and sensor fidelity ({int(confidence_score * 100)}% confidence)"
        )
        if face_scores_all:
            reasons.append(f"Face features appear authentic across {len(face_scores_all)} face samples")

    explanation = ". ".join(reasons) + "."

    dur = total_duration if total_duration is not None else (frames[-1][0] if frames else 0.0)

    return VisualVideoAnalysisResponse(
        status="success",
        verdict=verdict,
        confidence_score=confidence_score,
        composite_score=composite_score,
        analyzed_frames_count=len(timeline),
        duration_seconds=round(dur, 2),
        flagged_segments=flagged_segments,
        timeline=timeline,
        forensic_breakdown=forensic_breakdown,
        explanation=explanation,
    )
