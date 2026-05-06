from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
from urllib.parse import urlparse

# Initialize the FastAPI app
app = FastAPI(title="TruthLens API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load BOTH models into memory
print("Loading Fake News Engine...")
fake_news_model = pipeline("text-classification", model="dhruvpal/fake-news-bert")

print("Loading Tone/Bias Engine... (This will download weights the first time)")
bias_model = pipeline("text-classification", model="valurank/distilroberta-bias")

print("All engines online!")

# The Macro Context: Historic Domain Bias Dictionary
# We can easily move this to a Prisma/MongoDB database later
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

# Updated expected payload
class ArticleData(BaseModel):
    text: str
    url: str

# Helper function to clean URLs (turns 'https://www.cnn.com/article' into 'cnn.com')
def extract_domain(url: str):
    domain = urlparse(url).netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain

@app.post("/analyze")
async def analyze_text(data: ArticleData):
    # Truncate text for the models
    truncated_text = data.text[:2000] 
    
    # 1. Macro Context: Lookup the historic domain bias
    domain = extract_domain(data.url)
    historic_bias = HISTORIC_BIAS_DB.get(domain, "Unknown / Not in Database")
    
    # 2. Micro Context: Run the text through both AI models
    fake_news_result = fake_news_model(truncated_text)[0]
    tone_result = bias_model(truncated_text)[0]
    
    # Send a unified response back to the extension
    return {
        "status": "success",
        "source": domain,
        "historic_bias": historic_bias,
        "ai_analysis": {
            "fake_news": fake_news_result,
            "tone_bias": tone_result
        }
    }