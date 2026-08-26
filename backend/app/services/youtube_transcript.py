import logging
import re
from typing import List, Dict, Any, Optional
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import JSONFormatter

logger = logging.getLogger(__name__)

class TranscriptNormalizer:
    @staticmethod
    def normalize(raw_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Ported from frontend TranscriptNormalizer.
        Handles fragmented segments, max boundaries, punctuation flushes, and overlapping duplicate text.
        """
        if not raw_segments:
            return []

        normalized = []
        current_merge = None

        MAX_MERGE_DURATION_SEC = 15.0 
        MAX_MERGE_LENGTH_CHARS = 200
        MAX_GAP_SEC = 2.0

        for seg in raw_segments:
            text = seg.get("text", "").strip()
            if not text:
                continue

            if not current_merge:
                current_merge = {
                    "text": text,
                    "start_time": seg["start_time"],
                    "end_time": seg["end_time"]
                }
                continue

            gap = seg["start_time"] - current_merge["end_time"]
            duration = seg["end_time"] - current_merge["start_time"]
            length = len(current_merge["text"]) + len(text)

            has_punctuation_ending = current_merge["text"].strip().endswith(('.', '!', '?'))
            is_too_long = duration > MAX_MERGE_DURATION_SEC or length > MAX_MERGE_LENGTH_CHARS
            is_gap_too_large = gap > MAX_GAP_SEC

            if has_punctuation_ending or is_too_long or is_gap_too_large:
                # Flush current
                normalized.append(current_merge)
                current_merge = {
                    "text": text,
                    "start_time": seg["start_time"],
                    "end_time": seg["end_time"]
                }
            else:
                # Handle duplicate overlaps often seen in live/roll-up captions
                if current_merge["text"].endswith(text):
                    current_merge["end_time"] = max(current_merge["end_time"], seg["end_time"])
                elif text.startswith(current_merge["text"]):
                    current_merge["text"] = text
                    current_merge["end_time"] = max(current_merge["end_time"], seg["end_time"])
                else:
                    current_merge["text"] += " " + text
                    current_merge["end_time"] = max(current_merge["end_time"], seg["end_time"])

        if current_merge:
            normalized.append(current_merge)

        # Cleanup whitespaces
        for seg in normalized:
            seg["text"] = re.sub(r'\s+', ' ', seg["text"]).strip()

        return normalized

def get_transcript(video_id: str) -> Dict[str, Any]:
    """
    Retrieve transcript for a YouTube video, prioritizing manual English over auto English.
    Normalizes the transcript using TruthLens logic.
    """
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        
        selected_transcript = None
        is_auto_generated = False
        
        # 1. Prefer manually created English
        try:
            selected_transcript = transcript_list.find_manually_created_transcript(['en'])
        except Exception:
            pass
            
        # 2. Fallback to auto-generated English
        if not selected_transcript:
            try:
                selected_transcript = transcript_list.find_generated_transcript(['en'])
                is_auto_generated = True
            except Exception:
                pass
                
        # 3. Fallback to ANY manual transcript translated to English
        if not selected_transcript:
            for t in transcript_list:
                if t.is_translatable:
                    try:
                        selected_transcript = t.translate('en')
                        is_auto_generated = t.is_generated
                        break
                    except Exception:
                        pass
                        
        # 4. Grab anything
        if not selected_transcript:
            for t in transcript_list:
                selected_transcript = t
                is_auto_generated = t.is_generated
                break
                
        if not selected_transcript:
            raise ValueError("No transcript available for this video.")

        raw_snippets = selected_transcript.fetch()
        
        # Convert to TruthLens snippet format
        mapped_segments = []
        for snip in raw_snippets:
            mapped_segments.append({
                "text": snip.text,
                "start_time": snip.start,
                "end_time": snip.start + snip.duration
            })
            
        normalized_segments = TranscriptNormalizer.normalize(mapped_segments)
        
        if not normalized_segments:
            raise ValueError("Transcript is empty after normalization.")

        return {
            "video_id": video_id,
            "platform": "youtube",
            "source": "youtube_transcript_api",
            "language": selected_transcript.language,
            "language_code": selected_transcript.language_code,
            "is_auto_generated": is_auto_generated,
            "segments": normalized_segments
        }
    except Exception as e:
        logger.error(f"Error fetching transcript for {video_id}: {str(e)}")
        raise
