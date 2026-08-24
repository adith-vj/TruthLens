import re
import base64
import logging
from urllib.parse import urlparse
from io import BytesIO
from fastapi import APIRouter
import requests
from PIL import Image, ImageChops, ImageEnhance

# Pydantic schemas
from app.models.legacy import ArticleData, ExplainData, ImageData

# Legacy ML models
from app.services.legacy_models import (
    fake_news_model,
    bias_model,
    HISTORIC_BIAS_DB,
    deepfake_model,
    face_model,
)

# Legacy image functions
from app.services.legacy_image import (
    extract_metadata,
    preprocess_image,
    classify_image_type,
    get_analysis_routing,
    compute_ela_metrics,
    compute_fft_metrics,
    detect_faces_mediapipe,
    parse_model_scores,
    compute_ensemble,
)

logger = logging.getLogger(__name__)

router = APIRouter()

def extract_domain(url: str):
    domain = urlparse(url).netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


# NOTE: Converted from `async def` to `def` so FastAPI correctly runs
# this CPU-blocking synchronous code in a threadpool to avoid freezing 
# the rest of the application (like /api/verify). The ML behavior is unchanged.
@router.post("/analyze")
def analyze_text(data: ArticleData):
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


@router.post("/explain")
def explain_bias(data: ExplainData):
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


@router.post("/analyze-image")
def analyze_image(data: ImageData):
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


@router.post("/deep-scan-image")
def deep_scan_image(data: ImageData):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(data.image_url, headers=headers, timeout=15)
        response.raise_for_status()

        image_bytes = response.content
        if len(image_bytes) > 20 * 1024 * 1024:
            return {"status": "error", "message": "Image too large (max 20 MB)"}

        raw_image = Image.open(BytesIO(image_bytes))

        w, h = raw_image.size
        if w * h > 10_000_000:
            raw_image.thumbnail((3000, 3000), Image.LANCZOS)
            logger.info("Downsampled %dx%d image to %s", w, h, raw_image.size)

        metadata  = extract_metadata(raw_image)
        processed = preprocess_image(raw_image)

        image_type = classify_image_type(processed)
        routing    = get_analysis_routing(image_type)
        logger.info("Image type: %s | routing: %s", image_type, routing)

        ela_metrics = compute_ela_metrics(raw_image) if routing["ela"] else None
        fft_metrics = compute_fft_metrics(processed) if routing["fft"] else None

        face_info  = {"face_detected": False, "face_count": 0, "best_confidence": 0.0}
        face_scores = None
        if routing["face_detection"]:
            face_info = detect_faces_mediapipe(processed)
            if face_info["face_detected"]:
                face_results = face_model(processed)
                face_scores  = parse_model_scores(face_results, face_model)

        general_results = deepfake_model(processed)
        general_scores  = parse_model_scores(general_results, deepfake_model)

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
