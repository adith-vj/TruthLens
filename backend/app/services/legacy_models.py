import os
import logging
from transformers import pipeline
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python import BaseOptions as MpBaseOptions

logger = logging.getLogger(__name__)

print("Loading Fake News Engine...")
fake_news_model = pipeline("text-classification", model="dhruvpal/fake-news-bert")

print("Loading Tone/Bias Engine...")
bias_model = pipeline("text-classification", model="valurank/distilroberta-bias")

print("Loading Deepfake Vision Engine...")
deepfake_model = pipeline("image-classification", model="umm-maybe/AI-image-detector")
face_model = pipeline("image-classification", model="dima806/deepfake_vs_real_image_detection")

print("All engines online!")

# MediaPipe BlazeFace
_BLAZE_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "blaze_face_short_range.tflite",
)
face_detector = None
if os.path.isfile(_BLAZE_MODEL_PATH):
    try:
        face_detector = mp_vision.FaceDetector.create_from_options(
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
