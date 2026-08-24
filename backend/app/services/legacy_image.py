import math
import numpy as np
import cv2
import logging
from io import BytesIO
from PIL import Image, ExifTags, ImageChops, ImageEnhance
import mediapipe as mp

from app.services.legacy_models import (
    deepfake_model,
    face_model,
    face_detector,
)

logger = logging.getLogger(__name__)

def preprocess_image(image: Image.Image) -> Image.Image:
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGBA")
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        return background
    else:
        return image.convert("RGB")

def extract_metadata(image: Image.Image) -> dict:
    metadata = {
        "format": image.format,
        "size": f"{image.size[0]}x{image.size[1]}",
        "exif_present": False,
        "software_tag": "None detected",
        "red_flag": False
    }
    try:
        exif_data = image.getexif()
        if exif_data:
            metadata["exif_present"] = True
            for tag_id, value in exif_data.items():
                tag_name = ExifTags.TAGS.get(tag_id, tag_id)
                if tag_name == "Software":
                    metadata["software_tag"] = str(value)
                    suspicious_tools = ["Photoshop", "Midjourney", "DALL", "Stable Diffusion", "Automatic1111", "ComfyUI"]
                    if any(tool.lower() in str(value).lower() for tool in suspicious_tools):
                        metadata["red_flag"] = True
    except Exception:
        pass
    return metadata

def parse_model_scores(results: list, target_pipeline) -> dict:
    id2label = {}
    if hasattr(target_pipeline, "model") and hasattr(target_pipeline.model, "config") and hasattr(target_pipeline.model.config, "id2label"):
        id2label = target_pipeline.model.config.id2label

    fake_score = 0.0
    real_score = 0.0

    for item in results:
        raw_label = str(item.get("label", ""))
        score = float(item.get("score", 0.0))

        resolved_label = raw_label
        if raw_label.startswith("LABEL_") and id2label:
            try:
                label_id = int(raw_label.split("_")[-1])
                resolved_label = str(id2label.get(label_id, raw_label))
            except ValueError:
                pass

        label_lower = resolved_label.lower()
        if any(term in label_lower for term in ["fake", "ai", "synthetic", "generated", "artificial"]):
            fake_score = score
        elif any(term in label_lower for term in ["real", "human", "authentic", "original"]):
            real_score = score

    if fake_score == 0.0 and real_score > 0.0:
        fake_score = 1.0 - real_score
    elif real_score == 0.0 and fake_score > 0.0:
        real_score = 1.0 - fake_score

    return {"fake_score": fake_score, "real_score": real_score}

def calibrate_score(raw_score, temperature=1.75):
    clamped = max(1e-7, min(1.0 - 1e-7, float(raw_score)))
    logit = math.log(clamped / (1.0 - clamped))
    return 1.0 / (1.0 + math.exp(-logit / temperature))

def classify_image_type(image):
    try:
        img_small = image.resize((256, 256), Image.LANCZOS)
        arr = np.array(img_small, dtype=np.uint8)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.count_nonzero(edges)) / edges.size

        hist = cv2.calcHist(
            [bgr], [0, 1, 2], None, [8, 8, 8],
            [0, 256, 0, 256, 0, 256],
        )
        h = hist.flatten().astype(np.float64)
        h = h / (h.sum() + 1e-10)
        h = h[h > 0]
        color_entropy = float(-np.sum(h * np.log2(h)))

        mean_saturation = float(np.mean(hsv[:, :, 1])) / 255.0

        gf = gray.astype(np.float32)
        local_mean = cv2.blur(gf, (8, 8))
        local_sq_mean = cv2.blur(gf * gf, (8, 8))
        local_var = np.maximum(local_sq_mean - local_mean ** 2, 0)
        uniform_fraction = float(np.mean(np.sqrt(local_var) < 5.0))

        if uniform_fraction > 0.70 and color_entropy < 3.0:
            return "document" if edge_density > 0.15 else "screenshot"
        if mean_saturation < 0.08 and color_entropy < 3.5:
            return "document"
        if edge_density > 0.12 and uniform_fraction > 0.40:
            return "meme"
        if mean_saturation < 0.15 and color_entropy > 5.0:
            return "illustration"
        return "photo"
    except Exception:
        return "photo"

_ROUTING_TABLE = {
    "photo":        {"ela": True,  "fft": True,  "face_detection": True,  "face_model": True},
    "screenshot":   {"ela": False, "fft": False, "face_detection": False, "face_model": False},
    "illustration": {"ela": False, "fft": True,  "face_detection": False, "face_model": False},
    "document":     {"ela": False, "fft": False, "face_detection": False, "face_model": False},
    "meme":         {"ela": True,  "fft": True,  "face_detection": False, "face_model": False},
}

def get_analysis_routing(image_type):
    return _ROUTING_TABLE.get(image_type, _ROUTING_TABLE["photo"])

def detect_faces_mediapipe(image):
    _NO_FACE = {"face_detected": False, "face_count": 0, "best_confidence": 0.0}
    if face_detector is None:
        return _NO_FACE
    try:
        rgb = np.array(image, dtype=np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = face_detector.detect(mp_image)
        if result.detections:
            best = max(
                d.categories[0].score for d in result.detections
            )
            return {
                "face_detected": True,
                "face_count": len(result.detections),
                "best_confidence": round(float(best), 4),
            }
        return _NO_FACE
    except Exception as e:
        logger.warning("Face detection failed: %s", e)
        return _NO_FACE

def compute_ela_metrics(image):
    try:
        original_img = preprocess_image(image)
        np_original = np.array(original_img, dtype=np.float32)

        quality_levels = [75, 85, 95]
        quality_results = {}

        for q in quality_levels:
            buf = BytesIO()
            original_img.save(buf, "JPEG", quality=q)
            buf.seek(0)
            compressed = Image.open(buf).convert("RGB")
            np_compressed = np.array(compressed, dtype=np.float32)

            diff = np.abs(np_original - np_compressed)
            channel_avg = np.mean(diff, axis=2)

            quality_results[q] = {
                "p95": float(np.percentile(channel_avg, 95)),
                "p99": float(np.percentile(channel_avg, 99)),
                "mean": float(np.mean(channel_avg)),
                "std":  float(np.std(channel_avg)),
            }

        best_q = min(quality_results, key=lambda q: quality_results[q]["p95"])
        best   = quality_results[best_q]
        min_p95 = best["p95"]

        anomaly_score = 1.0 / (1.0 + math.exp(-(min_p95 - 8.0) / 4.0))

        return {
            "mean_diff":          round(best["mean"], 2),
            "std_diff":           round(best["std"], 2),
            "p95_diff":           round(min_p95, 2),
            "p99_diff":           round(best["p99"], 2),
            "best_match_quality": best_q,
            "anomaly_score":      round(anomaly_score, 4),
        }
    except Exception as e:
        logger.warning("ELA computation failed: %s", e)
        return {
            "mean_diff": 0.0, "std_diff": 0.0,
            "p95_diff": 0.0, "p99_diff": 0.0,
            "best_match_quality": 0, "anomaly_score": 0.0,
        }

def compute_fft_metrics(image):
    try:
        gray = image.convert("L")
        img_array = np.array(gray, dtype=np.float32)
        rows, cols = img_array.shape

        f_shift   = np.fft.fftshift(np.fft.fft2(img_array))
        magnitude = np.abs(f_shift) + 1e-10

        crow, ccol = rows // 2, cols // 2
        Y, X = np.ogrid[-crow:rows - crow, -ccol:cols - ccol]
        radius_map = np.sqrt(X.astype(np.float64) ** 2
                             + Y.astype(np.float64) ** 2)
        max_radius = np.sqrt(float(crow ** 2 + ccol ** 2)) + 1e-10
        norm_radius = radius_map / max_radius

        n_bins = 64
        bin_edges = np.linspace(0, 1, n_bins + 1)
        radial_profile = np.zeros(n_bins, dtype=np.float64)
        for i in range(n_bins):
            mask = (norm_radius >= bin_edges[i]) & (norm_radius < bin_edges[i + 1])
            if np.any(mask):
                radial_profile[i] = np.mean(magnitude[mask])

        suppression_w = max(1, int(0.02 * n_bins))
        cleaned = radial_profile.copy()
        for k in range(1, 5):
            centre = int((k / 8.0) * n_bins)
            lo = max(1, centre - suppression_w)
            hi = min(n_bins - 1, centre + suppression_w + 1)
            if lo > 0 and hi < n_bins:
                cleaned[lo:hi] = np.linspace(
                    cleaned[lo - 1], cleaned[hi], hi - lo,
                )

        kernel  = np.array([0.25, 0.5, 0.25])
        cleaned = np.convolve(cleaned, kernel, mode="same")

        p = cleaned / (cleaned.sum() + 1e-10)
        p = p[p > 1e-15]
        spectral_entropy = float(-np.sum(p * np.log2(p)))
        max_entropy  = np.log2(n_bins)
        norm_entropy = spectral_entropy / max_entropy if max_entropy > 0 else 0.0

        hf_mask   = norm_radius > 0.3
        hf_energy = float(np.sum(magnitude[hf_mask]))
        total_energy = float(np.sum(magnitude))
        hf_ratio  = hf_energy / (total_energy + 1e-10)

        anomaly_score = 1.0 / (1.0 + math.exp((norm_entropy - 0.72) / 0.08))

        return {
            "hf_ratio":         round(hf_ratio, 4),
            "spectral_entropy": round(norm_entropy, 4),
            "total_energy":     round(total_energy, 2),
            "anomaly_score":    round(anomaly_score, 4),
        }
    except Exception as e:
        logger.warning("FFT computation failed: %s", e)
        return {
            "hf_ratio": 0.0, "spectral_entropy": 0.0,
            "total_energy": 0.0, "anomaly_score": 0.0,
        }

def compute_ensemble(
    general_scores,
    face_scores,
    face_detected,
    ela_metrics,
    fft_metrics,
    metadata,
    image_type,
):
    g = calibrate_score(general_scores["fake_score"])
    signals = ["general_ai"]
    reasons = []

    if face_detected and face_scores is not None:
        f = calibrate_score(face_scores["fake_score"])
        neural = 0.55 * g + 0.45 * f
        signals.append("face_deepfake")
        reasons.append(
            f"General AI detector ({g:.2f}) + face deepfake detector ({f:.2f}) "
            f"-> neural score {neural:.2f}"
        )
    else:
        f = None
        neural = g
        why = (
            f"image type: {image_type}"
            if image_type != "photo"
            else "no face detected"
        )
        reasons.append(f"General AI detector: {g:.2f} (face model skipped — {why})")

    f_scores = []
    if ela_metrics and ela_metrics.get("anomaly_score", 0) > 0:
        f_scores.append(ela_metrics["anomaly_score"])
        signals.append("ela")
    if fft_metrics and fft_metrics.get("anomaly_score", 0) > 0:
        f_scores.append(fft_metrics["anomaly_score"])
        signals.append("fft")

    if f_scores:
        f_avg = sum(f_scores) / len(f_scores)
        if f_avg > 0.5:
            corr = 1.0 + 0.15 * (f_avg - 0.5)
            reasons.append(f"Forensic signals corroborate AI detection ({f_avg:.2f} avg)")
        else:
            corr = 1.0 - 0.10 * (0.5 - f_avg)
            reasons.append(f"Forensic signals lean authentic ({f_avg:.2f} avg)")
    else:
        corr = 1.0
        reasons.append(f"Forensic analysis not applicable for {image_type}")

    if metadata.get("red_flag"):
        corr += 0.08
        signals.append("metadata_flag")
        reasons.append(
            f"AI-associated software tag: {metadata.get('software_tag', 'unknown')}"
        )

    final = max(0.0, min(1.0, neural * corr))

    is_fake   = (final >= 0.55) and (neural >= 0.45)
    prediction = "FAKE" if is_fake else "REAL"

    if is_fake:
        raw_c = (final - 0.55) / 0.45
        confidence = round(50.0 + raw_c * 49.0, 2)
    else:
        capped = min(final, 0.55)
        raw_c  = (0.55 - capped) / 0.55
        confidence = round(50.0 + raw_c * 49.0, 2)

    return {
        "prediction":             prediction,
        "confidence":             confidence,
        "composite_fake_score":   round(final, 4),
        "neural_score":           round(neural, 4),
        "general_score_calibrated": round(g, 4),
        "face_score_calibrated":  round(f, 4) if f is not None else None,
        "corroboration":          round(corr, 4),
        "signals_used":           signals,
        "explanation":            ". ".join(reasons) + ".",
    }
