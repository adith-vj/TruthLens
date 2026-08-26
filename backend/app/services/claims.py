import asyncio
import json
import logging
from typing import List, Dict, Any
from difflib import SequenceMatcher
import httpx

from app.core.config import settings
from app.services.classifier import classify_claim
from app.models.video import CandidateClaim
from app.services.youtube_transcript import get_transcript

logger = logging.getLogger(__name__)

GEMINI_GENERATE_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

EXTRACT_PROMPT = """You are a factual claim extraction AI. Extract externally checkable assertions from the provided transcript chunk. 
A good claim is a specific assertion about the world, an event, a metric, or a finding that can be fact-checked. 
Do not extract pure opinions, questions, predictions, or filler. 

Format the output strictly as a JSON array of objects. Do not invent timestamps. Use the bounding start_time and end_time of the segments from which the claim was extracted.

Output JSON Schema:
[
  {{
    "text": "The extracted claim text, polished for clarity if needed but remaining true to the transcript.",
    "start_time": 10.5,
    "end_time": 15.2,
    "has_numbers": true,
    "has_entities": true,
    "is_specific": true
  }}
]

Transcript Chunk:
{chunk_text}
"""

def chunk_transcript(segments: List[Dict[str, Any]], max_duration=60.0, overlap=10.0) -> List[List[Dict[str, Any]]]:
    chunks = []
    current_chunk = []
    current_start = None
    
    for seg in segments:
        if not current_chunk:
            current_start = seg["start_time"]
            current_chunk.append(seg)
            continue
            
        duration = seg["end_time"] - current_start
        if duration > max_duration:
            chunks.append(current_chunk)
            
            # Find the overlap start point
            overlap_target = seg["end_time"] - overlap
            
            new_chunk = []
            new_start = None
            for c_seg in current_chunk:
                if c_seg["end_time"] >= overlap_target:
                    if new_start is None:
                        new_start = c_seg["start_time"]
                    new_chunk.append(c_seg)
                    
            if not new_chunk:
                new_start = seg["start_time"]
                
            current_chunk = new_chunk
            current_chunk.append(seg)
            current_start = new_start
        else:
            current_chunk.append(seg)
            
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks

async def extract_claims_from_chunk(chunk: List[Dict[str, Any]], sem: asyncio.Semaphore) -> List[Dict[str, Any]]:
    chunk_text = "\n".join([f"[{s['start_time']} - {s['end_time']}] {s['text']}" for s in chunk])
    
    payload = {
        "contents": [{"parts": [{"text": EXTRACT_PROMPT.format(chunk_text=chunk_text)}]}],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json"
        },
    }
    
    api_key = settings.GEMINI_API_KEY.get_secret_value()
    if not api_key:
        logger.warning("GEMINI_API_KEY not set. Cannot extract claims.")
        return []
        
    async with sem:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
                response = await client.post(
                    GEMINI_GENERATE_URL,
                    json=payload,
                    params={"key": api_key},
                )
                response.raise_for_status()
                data = response.json()
                
                raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(raw_text)
        except Exception as e:
            logger.error(f"Failed to extract claims from chunk: {e}")
            return []

async def process_video_claims(video_id: str, top_n: int = 12) -> List[CandidateClaim]:
    # 1. Fetch transcript (Phase 5.3)
    # get_transcript is synchronous
    transcript_data = get_transcript(video_id)
    segments = transcript_data["segments"]
    
    if not segments:
        return []
        
    # 2. Chunk transcript
    chunks = chunk_transcript(segments)
    
    # 3. Extract claims concurrently with limits
    sem = asyncio.Semaphore(5) # limit concurrency
    tasks = [extract_claims_from_chunk(chunk, sem) for chunk in chunks]
    extracted_results = await asyncio.gather(*tasks)
    
    raw_claims = []
    for res in extracted_results:
        raw_claims.extend(res)
        
    if not raw_claims:
        return []
        
    # 4. Classify and Score
    # Process classification sequentially to avoid hammering Gemini rate limits, 
    # but we can do it in small batches if needed. 
    # Because _classify_by_rules is fast, we just gather with a semaphore.
    classify_sem = asyncio.Semaphore(5)
    
    async def process_claim(raw: Dict[str, Any]) -> CandidateClaim:
        text = raw.get("text", "").strip()
        start_time = float(raw.get("start_time", 0))
        end_time = float(raw.get("end_time", 0))
        
        async with classify_sem:
            claim_type_enum = await classify_claim(text)
            
        claim_type = claim_type_enum.value
        
        # Checkability score based on structural flags
        has_num = bool(raw.get("has_numbers", False))
        has_ent = bool(raw.get("has_entities", False))
        is_spec = bool(raw.get("is_specific", False))
        
        checkability = 0.0
        if has_num: checkability += 0.35
        if has_ent: checkability += 0.35
        if is_spec: checkability += 0.30
        
        # Base claim score
        claim_score = checkability
        
        if claim_type == "factual_claim":
            claim_score *= 1.0
        elif claim_type == "ambiguous":
            claim_score *= 0.5
        else: # opinion, advertisement
            claim_score *= 0.1
            
        return CandidateClaim(
            text=text,
            start_time=start_time,
            end_time=end_time,
            claim_type=claim_type,
            checkability_score=checkability,
            claim_score=claim_score
        )
        
    processed_claims = await asyncio.gather(*(process_claim(c) for c in raw_claims if c.get("text")))
    
    # 5. Deduplicate
    unique_claims = []
    for claim in processed_claims:
        is_dup = False
        for u_claim in unique_claims:
            ratio = SequenceMatcher(None, claim.text.lower(), u_claim.text.lower()).ratio()
            if ratio > 0.8:
                is_dup = True
                # Keep strongest wording
                if len(claim.text) > len(u_claim.text):
                    u_claim.text = claim.text
                # Preserve timestamps (earliest start, latest end)
                u_claim.start_time = min(u_claim.start_time, claim.start_time)
                u_claim.end_time = max(u_claim.end_time, claim.end_time)
                u_claim.claim_score = max(u_claim.claim_score, claim.claim_score)
                break
        if not is_dup:
            unique_claims.append(claim)
            
    # 6. Rank by claim_score descending
    unique_claims.sort(key=lambda x: x.claim_score, reverse=True)
    
    # 7. Top-N
    return unique_claims[:top_n]
