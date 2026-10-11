import os
import logging
from transformers import pipeline
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python import BaseOptions as MpBaseOptions

logger = logging.getLogger(__name__)

_fake_news_model = None
_bias_model = None
_deepfake_model = None
_face_model = None
_face_detector = None
_face_detector_initialized = False

# MediaPipe BlazeFace
_BLAZE_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "blaze_face_short_range.tflite",
)

def get_fake_news_model():
    global _fake_news_model
    if _fake_news_model is None:
        logger.info("Loading Fake News Engine...")
        _fake_news_model = pipeline("text-classification", model="dhruvpal/fake-news-bert")
    return _fake_news_model

def get_bias_model():
    global _bias_model
    if _bias_model is None:
        logger.info("Loading Tone/Bias Engine...")
        _bias_model = pipeline("text-classification", model="valurank/distilroberta-bias")
    return _bias_model

def get_deepfake_model():
    global _deepfake_model
    if _deepfake_model is None:
        logger.info("Loading Deepfake Vision Engine...")
        _deepfake_model = pipeline("image-classification", model="umm-maybe/AI-image-detector")
    return _deepfake_model

def get_face_model():
    global _face_model
    if _face_model is None:
        logger.info("Loading Face Deepfake Engine...")
        _face_model = pipeline("image-classification", model="dima806/deepfake_vs_real_image_detection")
    return _face_model

def get_face_detector():
    global _face_detector, _face_detector_initialized
    if not _face_detector_initialized:
        _face_detector_initialized = True
        if os.path.isfile(_BLAZE_MODEL_PATH):
            try:
                _face_detector = mp_vision.FaceDetector.create_from_options(
                    mp_vision.FaceDetectorOptions(
                        base_options=MpBaseOptions(model_asset_path=_BLAZE_MODEL_PATH),
                        min_detection_confidence=0.5,
                    )
                )
                logger.info("BlazeFace face detector loaded.")
            except Exception as _e:
                logger.warning("Failed to load BlazeFace model: %s", _e)
        else:
            logger.warning("BlazeFace model not found at %s — face gating disabled.", _BLAZE_MODEL_PATH)
    return _face_detector

def __getattr__(name: str):
    if name == "fake_news_model":
        return get_fake_news_model()
    if name == "bias_model":
        return get_bias_model()
    if name == "deepfake_model":
        return get_deepfake_model()
    if name == "face_model":
        return get_face_model()
    if name == "face_detector":
        return get_face_detector()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

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
