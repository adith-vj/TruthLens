import asyncio
import sys

from app.api.video import extract_video_claims
from app.models.video import VideoClaimsRequest

async def main():
    try:
        req = VideoClaimsRequest(video_id="EV9WuIheQl0")
        result = await extract_video_claims(req)
        print(f"Success! Total Candidates: {result.total_candidates}")
        for i, c in enumerate(result.selected_claims):
            print(f"{i+1}. [{c.start_time:.1f}-{c.end_time:.1f}] {c.text} (Type: {c.claim_type}, Score: {c.claim_score:.2f})")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
