import asyncio
import sys

from app.api.video import extract_video_claims
from app.models.video import VideoClaimsRequest
import app.services.claims as claims_svc

def mock_get_transcript(video_id):
    return {
        "segments": [
            {"text": "The company launched the product in 2024.", "start_time": 10.5, "end_time": 15.2},
            {"text": "It was very successful.", "start_time": 15.2, "end_time": 18.0},
            {"text": "Revenues increased by 30%.", "start_time": 18.0, "end_time": 22.0},
            {"text": "We think it's great.", "start_time": 22.0, "end_time": 25.0}
        ]
    }

claims_svc.get_transcript = mock_get_transcript

async def main():
    try:
        req = VideoClaimsRequest(video_id="FAKE")
        result = await extract_video_claims(req)
        print(f"Success! Total Candidates: {result.total_candidates}")
        for i, c in enumerate(result.selected_claims):
            print(f"{i+1}. [{c.start_time:.1f}-{c.end_time:.1f}] {c.text} (Type: {c.claim_type}, Score: {c.claim_score:.2f})")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
