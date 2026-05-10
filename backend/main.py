from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
from urllib.parse import urlparse
from transformers_interpret import SequenceClassificationExplainer
import requests
from io import BytesIO
from PIL import Image, ExifTags
from PIL import Image, ExifTags, ImageChops, ImageEnhance
import base64
from io import BytesIO
import re 

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

print("Loading Deepfake Vision Engine... (This will take a minute to download weights)")
deepfake_model = pipeline("image-classification", model="dima806/deepfake_vs_real_image_detection")

print("All engines online!")

HISTORIC_BIAS_DB = {
    "cnn.com": "Left-Leaning",
    "nytimes.com": "Left-Leaning",
    "foxnews.com": "Right-Leaning",
    "breitbart.com": "Right",
    "wsj.com": "Center-Right",
    "reuters.com": "Center / Neutral",
    "apnews.com": "Center / Neutral",
    "bbc.com": "Center / Neutral"
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
    
    # Slice the paragraph into sentences based on punctuation (. ! ?) followed by a space
    sentences = re.split(r'(?<=[.!?]) +', text_to_process)
    
    flagged_sentences = []

    for sentence in sentences:
        # Skip tiny fragments or empty strings
        if len(sentence) < 15:
            continue
            
        # Run the standard, blazing-fast pipeline on the single sentence
        result = bias_model(sentence)[0]
        print(f"TruthLens scanned: '{sentence[:30]}...' -> {result}")
        
        # distilroberta-bias outputs labels like 'BIASED' or 'NEUTRAL'
        # We only flag it if the AI is highly confident (e.g., > 75%)
        if result['label'].upper() == 'BIASED' and result['score'] > 0.75:
            flagged_sentences.append(sentence)

    return {
        "status": "success",
        "flagged_sentences": flagged_sentences
    }

@app.post("/analyze-image")
async def analyze_image(data: ImageData):
    try:
        # 1. Download the image 
        # (We use a User-Agent header so news sites don't block us thinking we're a malicious scraper)
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(data.image_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # 2. Load the image into memory
        image = Image.open(BytesIO(response.content))
        
        # 3. Baseline Metadata extraction
        metadata = {
            "format": image.format,
            "size": f"{image.size[0]}x{image.size[1]}",
            "exif_present": False,
            "software_tag": "None detected",
            "red_flag": False
        }
        
        # 4. Deep dive into EXIF tags
        exif_data = image.getexif()
        if exif_data:
            metadata["exif_present"] = True
            for tag_id, value in exif_data.items():
                tag_name = ExifTags.TAGS.get(tag_id, tag_id)
                
                # The "Software" tag is where Photoshop or AI generators usually leave their mark
                if tag_name == "Software":
                    metadata["software_tag"] = str(value)
                    
                    # Flag known manipulation tools
                    suspicious_tools = ["Photoshop", "Midjourney", "DALL", "Stable Diffusion"]
                    if any(tool.lower() in str(value).lower() for tool in suspicious_tools):
                        metadata["red_flag"] = True
        # --- TIER 2: ERROR LEVEL ANALYSIS (ELA) ---
        # Convert to RGB to ensure uniform channels
        original_img = image.convert("RGB")
        
        # Resave at a known quality level (90%)
        temp_io = BytesIO()
        original_img.save(temp_io, 'JPEG', quality=90)
        temp_io.seek(0)
        compressed_img = Image.open(temp_io)
        
        # Mathematically subtract the compressed image from the original
        ela_image = ImageChops.difference(original_img, compressed_img)
        
        # The difference is usually very dark, so we enhance the brightness to see the artifacts
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        scale = 255.0 / max_diff
        
        ela_enhanced = ImageEnhance.Brightness(ela_image).enhance(scale)
        
        # Convert the ELA heat map to a Base64 string so the Chrome Extension can display it
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
        # Catch errors if the image is protected or corrupted
        return {"status": "error", "message": str(e)}
@app.post("/deep-scan-image")
async def deep_scan_image(data: ImageData):
    try:
        # Download the image again (or you could cache it, but this keeps it stateless and clean)
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(data.image_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        image = Image.open(BytesIO(response.content)).convert("RGB")
        
        # Run the heavy Vision model
        results = deepfake_model(image)
        
        # The model returns a list of dicts like: [{'label': 'fake', 'score': 0.98}, {'label': 'real', 'score': 0.02}]
        top_prediction = results[0]
        
        return {
            "status": "success",
            "prediction": top_prediction['label'].upper(), # 'FAKE' or 'REAL'
            "confidence": round(top_prediction['score'] * 100, 2)
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}