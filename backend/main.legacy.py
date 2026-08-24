from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
from urllib.parse import urlparse
import requests
from io import BytesIO
from PIL import Image, ExifTags, ImageChops, ImageEnhance
import base64
import re
import numpy as np
import cv2
import math
import logging
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python import BaseOptions as MpBaseOptions

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("truthlens")
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# MediaPipe BlazeFace — initialised once at module level for efficiency
# ---------------------------------------------------------------------------
import os as _os
_BLAZE_MODEL_PATH = _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)),
    "blaze_face_short_range.tflite",
)
_face_detector = None
if _os.path.isfile(_BLAZE_MODEL_PATH):
    try:
        _face_detector = mp_vision.FaceDetector.create_from_options(
            mp_vision.FaceDetectorOptions(
                base_options=MpBaseOptions(model_asset_path=_BLAZE_MODEL_PATH),
                min_detection_confidence=0.5,
            )
        )
        print("BlazeFace face detector loaded.")
    except Exception as _e:
        logger.warning("Failed to load BlazeFace model: %s", _e)
else:
    logger.warning("BlazeFace model not found at %s — face gating disabled.", _BLAZE_MODEL_PATH)

app = FastAPI(title="TruthLens API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading Fake News Engine...")
fake_news_model = pipeline("text-classification", model="dhruvpal/fake-news-bert")

print("Loading Tone/Bias Engine...")
bias_model = pipeline("text-classification", model="valurank/distilroberta-bias")

print("Loading Deepfake Vision Engine...")
deepfake_model = pipeline("image-classification", model="umm-maybe/AI-image-detector")
face_model = pipeline("image-classification", model="dima806/deepfake_vs_real_image_detection")

print("All engines online!")

HISTORIC_BIAS_DB = {
    "cnn.com": "Left-Leaning",
    "nytimes.com": "Left-Leaning",
    "foxnews.com": "Right-Leaning",
    "breitbart.com": "Right",
    "wsj.com": "Center-Right",
    "reuters.com": "Center / Neutral",
    "apnews.com": "Center / Neutral",
    "bbc.com": "Center / Neutral",
    "aljazeera.com": "Center-Right"
}

class ArticleData(BaseModel):
    text: str
    url: str

class ExplainData(BaseModel):
    text: str

class ImageData(BaseModel):
    image_url: str

def extract_domain(url: str):
    domain = urlparse(url).netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain

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


# ===================================================================
#  REDESIGNED FORENSIC + ENSEMBLE PIPELINE  (v2)
# ===================================================================


# ---------------------------------------------------------------------------
# Temperature-scaling calibration
# ---------------------------------------------------------------------------
def calibrate_score(raw_score, temperature=1.75):
    """
    Apply temperature scaling in logit space to calibrate overconfident
    softmax outputs from HuggingFace classifiers.

    T > 1 compresses extreme probabilities toward 0.5.  T = 1.75 maps
    a raw 0.92 → ~0.79 and a raw 0.55 → ~0.53 while preserving rank
    ordering — a principled default when no calibration dataset is
    available (Guo et al., 2017).
    """
    clamped = max(1e-7, min(1.0 - 1e-7, float(raw_score)))
    logit = math.log(clamped / (1.0 - clamped))
    return 1.0 / (1.0 + math.exp(-logit / temperature))


# ---------------------------------------------------------------------------
# Lightweight image-type router
# ---------------------------------------------------------------------------
def classify_image_type(image):
    """
    Classify image as photo / screenshot / illustration / document / meme.

    Uses five lightweight OpenCV + NumPy features computed on a 256x256
    downscale (~2 ms).  Defaults to 'photo' (full forensic pipeline) for
    any ambiguous image — conservative by design.

    Features:
        edge_density      – Canny edge pixels / total pixels
        color_entropy      – Shannon entropy of 8x8x8 colour histogram
        mean_saturation    – mean HSV S-channel (0–1)
        uniform_fraction   – fraction of pixels with local std < 5
    """
    try:
        img_small = image.resize((256, 256), Image.LANCZOS)
        arr = np.array(img_small, dtype=np.uint8)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # 1. Edge density (Canny)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.count_nonzero(edges)) / edges.size

        # 2. Color entropy (8x8x8 histogram → Shannon entropy in bits)
        hist = cv2.calcHist(
            [bgr], [0, 1, 2], None, [8, 8, 8],
            [0, 256, 0, 256, 0, 256],
        )
        h = hist.flatten().astype(np.float64)
        h = h / (h.sum() + 1e-10)
        h = h[h > 0]
        color_entropy = float(-np.sum(h * np.log2(h)))

        # 3. Mean saturation (0–1)
        mean_saturation = float(np.mean(hsv[:, :, 1])) / 255.0

        # 4. Uniform-region fraction (local sigma < 5 in 8x8 neighbourhood)
        gf = gray.astype(np.float32)
        local_mean = cv2.blur(gf, (8, 8))
        local_sq_mean = cv2.blur(gf * gf, (8, 8))
        local_var = np.maximum(local_sq_mean - local_mean ** 2, 0)
        uniform_fraction = float(np.mean(np.sqrt(local_var) < 5.0))

        # Decision tree  ───────────────────────────────────────────────
        if uniform_fraction > 0.70 and color_entropy < 3.0:
            return "document" if edge_density > 0.15 else "screenshot"
        if mean_saturation < 0.08 and color_entropy < 3.5:
            return "document"
        if edge_density > 0.12 and uniform_fraction > 0.40:
            return "meme"
        if mean_saturation < 0.15 and color_entropy > 5.0:
            return "illustration"
        return "photo"  # default — runs full pipeline
    except Exception:
        return "photo"


# ---------------------------------------------------------------------------
# Analysis routing table
# ---------------------------------------------------------------------------
# Mapping from image type → which analyses are meaningful.
# See implementation_plan.md §2.4 for per-type rationale.
_ROUTING_TABLE = {
    #                    ELA    FFT    FaceDet  FaceModel
    "photo":        {"ela": True,  "fft": True,  "face_detection": True,  "face_model": True},
    "screenshot":   {"ela": False, "fft": False, "face_detection": False, "face_model": False},
    "illustration": {"ela": False, "fft": True,  "face_detection": False, "face_model": False},
    "document":     {"ela": False, "fft": False, "face_detection": False, "face_model": False},
    "meme":         {"ela": True,  "fft": True,  "face_detection": False, "face_model": False},
}


def get_analysis_routing(image_type):
    """Return the set of analyses to run for a given image type."""
    return _ROUTING_TABLE.get(image_type, _ROUTING_TABLE["photo"])


# ---------------------------------------------------------------------------
# Face detection  — MediaPipe BlazeFace
# ---------------------------------------------------------------------------
def detect_faces_mediapipe(image):
    """
    Detect human faces using MediaPipe BlazeFace (short-range model).

    Uses the module-level ``_face_detector`` (Tasks API, initialised
    once at startup).  If the model failed to load, gracefully returns
    no-face so the pipeline continues with the general detector only.
    """
    _NO_FACE = {"face_detected": False, "face_count": 0, "best_confidence": 0.0}
    if _face_detector is None:
        return _NO_FACE
    try:
        rgb = np.array(image, dtype=np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = _face_detector.detect(mp_image)
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


# ---------------------------------------------------------------------------
# ELA v2 — Multi-quality ladder with percentile statistics
# ---------------------------------------------------------------------------
def compute_ela_metrics(image):
    """
    Error Level Analysis using a multi-quality JPEG re-save ladder.

    Instead of guessing the input JPEG quality (fragile for PNG / WebP
    and non-standard encoders), re-saves at three quality levels
    (75, 85, 95) and picks the *minimum* 95th-percentile difference.
    This implicitly selects the quality closest to the original,
    where legitimate recompression produces the smallest delta.

    AI-generated or spliced regions show anomalous ELA at *all*
    quality levels, so the minimum remains elevated.

    Returns a calibrated anomaly_score in [0, 1] via a sigmoid mapping:
        P95 <  4 → anomaly < 0.27  (normal JPEG recompression)
        P95 ~  8 → anomaly ~ 0.50  (borderline)
        P95 > 12 → anomaly > 0.73  (suspicious)
    """
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

            # Per-pixel absolute diff averaged across RGB channels
            diff = np.abs(np_original - np_compressed)
            channel_avg = np.mean(diff, axis=2)

            quality_results[q] = {
                "p95": float(np.percentile(channel_avg, 95)),
                "p99": float(np.percentile(channel_avg, 99)),
                "mean": float(np.mean(channel_avg)),
                "std":  float(np.std(channel_avg)),
            }

        # Select quality with lowest P95 (closest match to original)
        best_q = min(quality_results, key=lambda q: quality_results[q]["p95"])
        best   = quality_results[best_q]
        min_p95 = best["p95"]

        # Sigmoid calibration to [0, 1]
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


# ---------------------------------------------------------------------------
# FFT v2 — Radial profile with JPEG suppression + spectral entropy
# ---------------------------------------------------------------------------
def compute_fft_metrics(image):
    """
    Frequency-domain analysis using a resolution-normalised radial
    profile with JPEG blocking-frequency suppression.

    Steps
    -----
    1. 2D FFT of the greyscale image → magnitude spectrum
    2. Mean magnitude in 64 radial-frequency bins (normalised to [0, 1]
       so results are resolution-independent)
    3. Suppress bins near JPEG 8x8-block frequencies (k/8, k = 1..4)
       via linear interpolation, then smooth with a 3-tap triangle kernel
    4. Compute spectral entropy of the cleaned profile

    AI-generated images tend to have smoother frequency rolloff (lower
    entropy) than real photographs.

    Returns calibrated anomaly_score in [0, 1].
    """
    try:
        gray = image.convert("L")
        img_array = np.array(gray, dtype=np.float32)
        rows, cols = img_array.shape

        # 2D FFT with DC at centre
        f_shift   = np.fft.fftshift(np.fft.fft2(img_array))
        magnitude = np.abs(f_shift) + 1e-10

        # Radial distance map normalised to [0, 1]
        crow, ccol = rows // 2, cols // 2
        Y, X = np.ogrid[-crow:rows - crow, -ccol:cols - ccol]
        radius_map = np.sqrt(X.astype(np.float64) ** 2
                             + Y.astype(np.float64) ** 2)
        max_radius = np.sqrt(float(crow ** 2 + ccol ** 2)) + 1e-10
        norm_radius = radius_map / max_radius

        # Radial profile (64 bins)
        n_bins = 64
        bin_edges = np.linspace(0, 1, n_bins + 1)
        radial_profile = np.zeros(n_bins, dtype=np.float64)
        for i in range(n_bins):
            mask = (norm_radius >= bin_edges[i]) & (norm_radius < bin_edges[i + 1])
            if np.any(mask):
                radial_profile[i] = np.mean(magnitude[mask])

        # Suppress JPEG blocking frequencies at k/8 for k = 1..4
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

        # Mild smoothing (3-tap triangle kernel)
        kernel  = np.array([0.25, 0.5, 0.25])
        cleaned = np.convolve(cleaned, kernel, mode="same")

        # Spectral entropy (normalised to [0, 1])
        p = cleaned / (cleaned.sum() + 1e-10)
        p = p[p > 1e-15]
        spectral_entropy = float(-np.sum(p * np.log2(p)))
        max_entropy  = np.log2(n_bins)
        norm_entropy = spectral_entropy / max_entropy if max_entropy > 0 else 0.0

        # High-frequency energy ratio (boundary at 0.3 normalised freq)
        hf_mask   = norm_radius > 0.3
        hf_energy = float(np.sum(magnitude[hf_mask]))
        total_energy = float(np.sum(magnitude))
        hf_ratio  = hf_energy / (total_energy + 1e-10)

        # Anomaly: low entropy → smoother spectrum → more likely AI
        # Sigmoid centred at 0.72 (normalised entropy), scale 0.08
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


# ---------------------------------------------------------------------------
# Confidence-aware ensemble with corroboration gating
# ---------------------------------------------------------------------------
def compute_ensemble(
    general_scores,
    face_scores,
    face_detected,
    ela_metrics,
    fft_metrics,
    metadata,
    image_type,
):
    """
    Fuse temperature-calibrated neural scores with forensic corroboration.

    Decision requires BOTH gates to pass:
        • final_score  >= 0.55   (overall evidence above noise)
        • neural_score >= 0.45   (at least one neural detector flags AI)

    The dual gate prevents forensic noise alone from triggering
    false positives and prevents a single uncertain detector from
    dominating the ensemble.

    Threshold / weight rationale (see implementation_plan.md §2.7):
        0.55  final threshold   – asymmetric toward REAL to minimise FP
        0.45  neural floor      – forensics alone cannot override a REAL neural prediction
        0.55 / 0.45  g/f split  – general model is more broadly trained; slight preference
        0.15  forensic boost    – mild corroboration, cannot flip a clear REAL
        0.10  forensic dampen   – asymmetric: we trust forensic disagreement less
        0.08  metadata bonus    – weak signal; alone cannot trigger FAKE
    """
    # ── 1. Calibrate neural outputs ──────────────────────────────────
    g = calibrate_score(general_scores["fake_score"])
    signals = ["general_ai"]
    reasons = []

    # ── 2. Neural base score ─────────────────────────────────────────
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

    # ── 3. Forensic corroboration factor ─────────────────────────────
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

    # ── 4. Metadata bonus (weak signal, never sole evidence) ─────────
    if metadata.get("red_flag"):
        corr += 0.08
        signals.append("metadata_flag")
        reasons.append(
            f"AI-associated software tag: {metadata.get('software_tag', 'unknown')}"
        )

    # ── 5. Final fused score ─────────────────────────────────────────
    final = max(0.0, min(1.0, neural * corr))

    # ── 6. Dual-gate decision ────────────────────────────────────────
    is_fake   = (final >= 0.55) and (neural >= 0.45)
    prediction = "FAKE" if is_fake else "REAL"

    # ── 7. Confidence: distance from threshold → 50–99 % ────────────
    if is_fake:
        # final ∈ [0.55, 1.0] → confidence ∈ [50, 99]
        raw_c = (final - 0.55) / 0.45
        confidence = round(50.0 + raw_c * 49.0, 2)
    else:
        # final ∈ [0.0, 0.55] → confidence ∈ [50, 99]  (inverted)
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


# ===================================================================
#  ENDPOINTS
# ===================================================================

@app.post("/analyze")
async def analyze_text(data: ArticleData):
    truncated_text = data.text[:2000] 
    domain = extract_domain(data.url)
    historic_bias = HISTORIC_BIAS_DB.get(domain, "Unknown / Not in Database")
    
    fake_news_result = fake_news_model(truncated_text)[0]
    tone_result = bias_model(truncated_text)[0]
    
    return {
        "status": "success",
        "source": domain,
        "historic_bias": historic_bias,
        "ai_analysis": {
            "fake_news": fake_news_result,
            "tone_bias": tone_result
        }
    }

@app.post("/explain")
async def explain_bias(data: ExplainData):
    text_to_process = data.text[:2000] 
    sentences = re.split(r'(?<=[.!?]) +', text_to_process)
    flagged_sentences = []

    for sentence in sentences:
        if len(sentence) < 15:
            continue
            
        result = bias_model(sentence)[0]
        if result['label'].upper() == 'BIASED' and result['score'] > 0.75:
            flagged_sentences.append(sentence)

    return {
        "status": "success",
        "flagged_sentences": flagged_sentences
    }

@app.post("/analyze-image")
async def analyze_image(data: ImageData):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(data.image_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        raw_image = Image.open(BytesIO(response.content))
        metadata = extract_metadata(raw_image)
        
        original_img = preprocess_image(raw_image)
        
        temp_io = BytesIO()
        original_img.save(temp_io, 'JPEG', quality=90)
        temp_io.seek(0)
        compressed_img = Image.open(temp_io)
        
        ela_image = ImageChops.difference(original_img, compressed_img)
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        scale = 255.0 / max_diff
        
        ela_enhanced = ImageEnhance.Brightness(ela_image).enhance(scale)
        
        buffered = BytesIO()
        ela_enhanced.save(buffered, format="JPEG")
        ela_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return {
            "status": "success",
            "tier_1_metadata": metadata,
            "tier_2_ela": {
                "heatmap_base64": f"data:image/jpeg;base64,{ela_base64}"
            }
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/deep-scan-image")
async def deep_scan_image(data: ImageData):
    """
    Production image authenticity scanner.

    Pipeline:
        1. Download & validate image
        2. Classify image type (photo / screenshot / illustration / document / meme)
        3. Route to appropriate forensic analyses
        4. Face-gate the deepfake model (MediaPipe BlazeFace)
        5. Temperature-calibrate all neural scores
        6. Fuse via confidence-aware ensemble with corroboration gating
        7. Apply dual-threshold decision (final >= 0.55 AND neural >= 0.45)

    Returns the same top-level contract as before:
        { status, prediction, confidence, details }
    """
    try:
        # ── Download ─────────────────────────────────────────────────
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(data.image_url, headers=headers, timeout=15)
        response.raise_for_status()

        image_bytes = response.content
        if len(image_bytes) > 20 * 1024 * 1024:
            return {"status": "error", "message": "Image too large (max 20 MB)"}

        raw_image = Image.open(BytesIO(image_bytes))

        # Down-sample very large images to prevent OOM
        w, h = raw_image.size
        if w * h > 10_000_000:
            raw_image.thumbnail((3000, 3000), Image.LANCZOS)
            logger.info("Downsampled %dx%d image to %s", w, h, raw_image.size)

        metadata  = extract_metadata(raw_image)
        processed = preprocess_image(raw_image)

        # ── Image-type routing ───────────────────────────────────────
        image_type = classify_image_type(processed)
        routing    = get_analysis_routing(image_type)
        logger.info("Image type: %s | routing: %s", image_type, routing)

        # ── Run enabled forensic analyses ────────────────────────────
        ela_metrics = compute_ela_metrics(raw_image) if routing["ela"] else None
        fft_metrics = compute_fft_metrics(processed) if routing["fft"] else None

        # ── Face detection → conditional face model ──────────────────
        face_info  = {"face_detected": False, "face_count": 0, "best_confidence": 0.0}
        face_scores = None
        if routing["face_detection"]:
            face_info = detect_faces_mediapipe(processed)
            if face_info["face_detected"]:
                face_results = face_model(processed)
                face_scores  = parse_model_scores(face_results, face_model)

        # ── General AI detector (always runs) ────────────────────────
        general_results = deepfake_model(processed)
        general_scores  = parse_model_scores(general_results, deepfake_model)

        # ── Confidence-aware ensemble ────────────────────────────────
        result = compute_ensemble(
            general_scores=general_scores,
            face_scores=face_scores,
            face_detected=face_info["face_detected"],
            ela_metrics=ela_metrics,
            fft_metrics=fft_metrics,
            metadata=metadata,
            image_type=image_type,
        )

        return {
            "status": "success",
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "details": {
                "composite_fake_score":  result["composite_fake_score"],
                "calibrated_threshold":  0.55,
                "exif_present":          metadata["exif_present"],
                "ela_metrics": ela_metrics or {
                    "skipped": True,
                    "reason": f"Not applicable for {image_type}",
                },
                "fft_metrics": fft_metrics or {
                    "skipped": True,
                    "reason": f"Not applicable for {image_type}",
                },
                "image_type":            image_type,
                "face_detected":         face_info["face_detected"],
                "face_count":            face_info["face_count"],
                "general_detector_score": result["general_score_calibrated"],
                "face_detector_score":   result["face_score_calibrated"],
                "neural_score":          result["neural_score"],
                "forensic_corroboration": result["corroboration"],
                "signals_used":          result["signals_used"],
                "explanation":           result["explanation"],
            },
        }

    except requests.exceptions.RequestException as e:
        logger.error("Image download failed: %s", e)
        return {"status": "error", "message": f"Failed to download image: {e}"}
    except Exception as e:
        logger.error("Deep scan failed: %s", e, exc_info=True)
        return {"status": "error", "message": str(e)}