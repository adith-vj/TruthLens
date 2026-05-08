from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
from urllib.parse import urlparse
# --- NEW IMPORT ---
from transformers_interpret import SequenceClassificationExplainer

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

# --- UPDATED SMART EXPLAINER ENDPOINT ---
@app.post("/explain")
async def explain_bias(data: ExplainData):
    # Truncate text (Explaining takes more compute than just classifying)
    text_to_process = data.text[:1000] 

    # Initialize the explainer with your loaded bias model
    explainer = SequenceClassificationExplainer(
        bias_model.model,
        bias_model.tokenizer
    )

    # Get attributions for the predicted class
    word_attributions = explainer(text_to_process)

    tokens_data = []
    current_word = ""
    current_weight = 0.0

    # Stitching subwords back into full words for the Chrome Extension
    for subword, weight in word_attributions:
        # Skip special structural tokens
        if subword in ['<s>', '</s>', '<pad>', '<cls>', '<sep>']:
            continue

        # RoBERTa uses 'Ġ' to denote a space / start of a new word
        if subword.startswith('Ġ'):
            if current_word:
                # Raw weights are usually small decimals (e.g. 0.15). 
                # We multiply by 2.5 to scale them up so the frontend CSS highlights pop.
                normalized_weight = min(abs(current_weight) * 2.5, 1.0)
                tokens_data.append({"word": current_word, "weight": normalized_weight})

            # Start a new word
            current_word = subword.replace('Ġ', '')
            current_weight = weight
        else:
            # It's a continuation of the previous word (a subword chunk)
            current_word += subword.replace('Ġ', '')
            current_weight += weight 

    # Catch the very last word in the loop
    if current_word:
        normalized_weight = min(abs(current_weight) * 2.5, 1.0)
        tokens_data.append({"word": current_word, "weight": normalized_weight})

    return {
        "status": "success",
        "tokens": tokens_data
    }