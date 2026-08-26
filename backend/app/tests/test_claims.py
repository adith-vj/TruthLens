import pytest
import asyncio
from unittest.mock import patch, MagicMock
from app.services.claims import chunk_transcript, process_video_claims
from app.models.video import CandidateClaim
from app.services.classifier import ClaimType
import app.services.claims as claims_svc

def test_transcript_chunking():
    # 200 segments of 1 second each
    segments = [{"text": f"Word {i}", "start_time": float(i), "end_time": float(i+1)} for i in range(200)]
    
    chunks = chunk_transcript(segments, max_duration=60.0, overlap=10.0)
    
    # 200 seconds total. Chunk max 60s, overlap 10s.
    # Chunk 1: 0 - 61s (61 segments)
    # Chunk 2: overlap starts at 61-10 = 51s. 51s - 112s (61 segments)
    # Chunk 3: overlap starts at 112-10 = 102s. 102s - 163s
    # Chunk 4: overlap starts at 163-10 = 153s. 153s - 200s
    assert len(chunks) == 4
    
    # Check overlap explicitly
    chunk_1 = chunks[0]
    chunk_2 = chunks[1]
    
    # End of chunk 1 is ~61
    assert chunk_1[-1]["end_time"] >= 60.0
    # Start of chunk 2 should be around 51
    assert chunk_2[0]["start_time"] <= chunk_1[-1]["end_time"] - 9.0

@pytest.mark.asyncio
async def test_duplicate_claim_detection():
    # Setup mock to return multiple claims that are very similar
    mock_get_transcript = MagicMock(return_value={"segments": [{"text": "Mock", "start_time": 0.0, "end_time": 10.0}]})
    
    async def mock_extract(*args, **kwargs):
        return [
            {"text": "The company launched the product in 2024.", "start_time": 10.5, "end_time": 15.2, "has_numbers": True},
            {"text": "Company launched product in 2024.", "start_time": 15.2, "end_time": 18.0, "has_numbers": True},
            {"text": "Totally different claim here.", "start_time": 20.0, "end_time": 22.0}
        ]
        
    async def mock_classify(text):
        return ClaimType.FACTUAL_CLAIM
        
    with patch("app.services.claims.get_transcript", mock_get_transcript), \
         patch("app.services.claims.extract_claims_from_chunk", mock_extract), \
         patch("app.services.claims.classify_claim", mock_classify):
         
         claims = await process_video_claims("FAKE_VIDEO")
         
         # The two similar claims should be merged. Total = 2.
         assert len(claims) == 2
         
         merged_claim = next(c for c in claims if "2024" in c.text)
         # Should keep the strongest (longest) wording
         assert merged_claim.text == "The company launched the product in 2024."
         # Should combine start and end times (min start, max end)
         assert merged_claim.start_time == 10.5
         assert merged_claim.end_time == 18.0

@pytest.mark.asyncio
async def test_deterministic_claim_scoring():
    mock_get_transcript = MagicMock(return_value={"segments": [{"text": "Mock", "start_time": 0.0, "end_time": 10.0}]})
    
    async def mock_extract(*args, **kwargs):
        return [
            # High checkability
            {"text": "A", "start_time": 0, "end_time": 1, "has_numbers": True, "has_entities": True, "is_specific": True},
            # Low checkability
            {"text": "B", "start_time": 1, "end_time": 2, "has_numbers": False, "has_entities": False, "is_specific": False},
        ]
        
    # Mock classify to return AMBIGUOUS for B to test score penalty
    async def mock_classify(text):
        if text == "B": return ClaimType.AMBIGUOUS
        return ClaimType.FACTUAL_CLAIM
        
    with patch("app.services.claims.get_transcript", mock_get_transcript), \
         patch("app.services.claims.extract_claims_from_chunk", mock_extract), \
         patch("app.services.claims.classify_claim", mock_classify):
         
         claims = await process_video_claims("FAKE_VIDEO")
         
         assert len(claims) == 2
         claim_a = next(c for c in claims if c.text == "A")
         claim_b = next(c for c in claims if c.text == "B")
         
         assert claim_a.checkability_score > claim_b.checkability_score
         # Check penalty for ambiguous
         assert claim_a.claim_score == 1.0 # (0.35+0.35+0.3) * 1.0
         assert claim_b.claim_score == 0.0 # 0.0 * 0.5 = 0.0
