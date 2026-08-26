import asyncio
import sys

from app.api.video import fetch_transcript
from app.models.video import VideoTranscriptRequest

async def main():
    try:
        req = VideoTranscriptRequest(video_id="EV9WuIheQl0")
        result = await fetch_transcript(req)
        print(f"Success! Language: {result.language}, Auto-generated: {result.is_auto_generated}")
        print(f"Segment Count: {len(result.segments)}")
        if result.segments:
            print(f"First Segment: {result.segments[0]}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
