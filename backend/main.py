from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline

# Initialize the FastAPI app
app = FastAPI(title="TruthLens API")

# Set up CORS to allow the Chrome Extension to communicate with this local API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Note: In production, we will restrict this to your specific extension ID
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the fast, pre-trained fake news model into memory on startup
print("Loading model... (this will download the weights the very first time)")
classifier = pipeline("text-classification", model="dhruvpal/fake-news-bert")
print("Model loaded successfully!")

# Define the data structure we expect to receive from the Chrome extension
class ArticleData(BaseModel):
    text: str

# Create the endpoint that the extension will hit
@app.post("/analyze")
async def analyze_text(data: ArticleData):
    # Transformer models typically max out at 512 tokens. 
    # We truncate the raw text to about 2000 characters to prevent the model from throwing an error.
    truncated_text = data.text[:2000] 
    
    # Run the text through the BERT model
    prediction = classifier(truncated_text)
    
    # Return the label and confidence score
    return {
        "status": "success",
        "result": prediction[0] 
    }