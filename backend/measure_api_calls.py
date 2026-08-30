"""
measure_api_calls.py - Measures actual API calls made during video claim verification.

Usage (from backend/ directory):
    python measure_api_calls.py [video_id]

Runs the BEFORE pipeline (old verify_claim via /api/verify route) and the
AFTER pipeline (new verify_video_claim with Gemini-first) against the same
claims set, with instrumented wrappers counting every external call.

Requires .env with GOOGLE_FACTCHECK_API_KEY, GEMINI_API_KEY, TAVILY_API_KEY.
Uses the test video Rpq1P7TiBD0 (cockroach farming/insects) by default, or any video_id argument.
"""

import asyncio
import sys
import os
from dataclasses import dataclass, field
from typing import Any

# Ensure backend/ is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@dataclass
class CallCounts:
    factcheck: int = 0
    tavily: int = 0
    gemini_first_pass: int = 0
    gemini_evidence: int = 0

    @property
    def total_gemini(self):
        return self.gemini_first_pass + self.gemini_evidence

    def __str__(self):
        return (
            f"  Google Fact Check: {self.factcheck}\n"
            f"  Tavily:            {self.tavily}\n"
            f"  Gemini first-pass: {self.gemini_first_pass}\n"
            f"  Gemini evidence:   {self.gemini_evidence}\n"
            f"  Total Gemini:      {self.total_gemini}"
        )


async def run_before(claims, counts: CallCounts):
    """Run BEFORE pipeline: old /api/verify path (Tavily-first)."""
    from unittest.mock import patch, AsyncMock
    from app.services.factcheck import verify_claim_factcheck as real_fc
    from app.services.search import search_evidence as real_se
    from app.services.llm import verify_with_llm as real_llm

    async def counting_fc(text):
        counts.factcheck += 1
        return await real_fc(text)

    async def counting_se(text):
        counts.tavily += 1
        return await real_se(text)

    async def counting_llm(text, ev):
        counts.gemini_evidence += 1
        return await real_llm(text, ev)

    from app.api.verify import verify_claim
    from app.models.verification import VerifyRequest

    with patch("app.api.verify.verify_claim_factcheck", side_effect=counting_fc), \
         patch("app.api.verify.search_evidence", side_effect=counting_se), \
         patch("app.api.verify.verify_with_llm", side_effect=counting_llm):
        for i, claim in enumerate(claims):
            print(f"  [{i+1}/{len(claims)}] Verifying: {claim.text[:60]}...")
            req = VerifyRequest(text=claim.text)
            try:
                result = await verify_claim(req)
                print(f"         -> {result.verdict} ({result.confidence_score:.2f})")
            except Exception as e:
                print(f"         -> ERROR: {e}")


async def run_after(claims, counts: CallCounts, video_id: str):
    """Run AFTER pipeline: new verify_video_claim (Gemini-first)."""
    from app.services.factcheck import verify_claim_factcheck as real_fc
    from app.services.search import search_evidence as real_se
    from app.services.llm import verify_with_llm as real_llm
    from app.services.video_verify import _gemini_first_pass as real_gfp
    from app.services.video_verify import verify_video_claim, clear_verify_cache

    clear_verify_cache()

    async def counting_fc(text):
        counts.factcheck += 1
        return await real_fc(text)

    async def counting_se(text):
        counts.tavily += 1
        return await real_se(text)

    async def counting_llm(text, ev):
        counts.gemini_evidence += 1
        return await real_llm(text, ev)

    async def counting_gfp(claim, context):
        counts.gemini_first_pass += 1
        return await real_gfp(claim, context)

    from unittest.mock import patch

    with patch("app.services.video_verify.verify_claim_factcheck", side_effect=counting_fc), \
         patch("app.services.video_verify.search_evidence", side_effect=counting_se), \
         patch("app.services.video_verify.verify_with_llm", side_effect=counting_llm), \
         patch("app.services.video_verify._gemini_first_pass", side_effect=counting_gfp):
        for i, claim in enumerate(claims):
            print(f"  [{i+1}/{len(claims)}] Verifying: {claim.text[:60]}...")
            try:
                result = await verify_video_claim(claim, video_id=video_id)
                print(
                    f"         -> {result.verdict} ({result.confidence_score:.2f}) "
                    f"[FC={result.metrics.google_factcheck_calls} "
                    f"G1={result.metrics.gemini_first_pass_calls} "
                    f"Tv={result.metrics.tavily_calls} "
                    f"Gev={result.metrics.gemini_evidence_calls}]"
                )
            except Exception as e:
                print(f"         -> ERROR: {e}")


async def main():
    video_id = sys.argv[1] if len(sys.argv) > 1 else "Rpq1P7TiBD0"

    print(f"\n=== TruthLens Phase 5.5 API Call Measurement ===")
    print(f"Video ID: {video_id}\n")

    # Step 1: Get or fetch Phase 5.4 claims
    from app.services.claims import process_video_claims

    print("Fetching Phase 5.4 claims (this makes real YouTube + Gemini requests)...")
    try:
        claims = await process_video_claims(video_id, top_n=3)
        print(f"Got {len(claims)} claims.\n")
        for i, c in enumerate(claims):
            print(f"  [{i+1}] score={c.claim_score:.3f} check={c.checkability_score:.3f} {c.text[:80]}")
        print()
    except Exception as e:
        print(f"ERROR fetching claims: {e}")
        print("Make sure YOUTUBE/GEMINI API keys are configured in .env")
        return

    if not claims:
        print("No claims extracted. Exiting.")
        return

    # Step 2: BEFORE measurement
    print("--- BEFORE (old pipeline: Fact Check -> Tavily -> Gemini) ---")
    before = CallCounts()
    await run_before(claims, before)
    print(f"\nBEFORE totals (over {len(claims)} claims):\n{before}")

    # Step 3: AFTER measurement
    print(f"\n--- AFTER (new pipeline: Fact Check -> Gemini first-pass -> conditional Tavily) ---")
    after = CallCounts()
    await run_after(claims, after, video_id)
    print(f"\nAFTER totals (over {len(claims)} claims):\n{after}")

    # Step 4: Summary
    print("\n=== Summary ===")
    print(f"Claims analyzed: {len(claims)}")
    print(f"Tavily calls:    {before.tavily} -> {after.tavily}  ({before.tavily - after.tavily:+d})")
    print(f"Gemini total:    {before.total_gemini} -> {after.total_gemini}  ({before.total_gemini - after.total_gemini:+d})")
    print(f"  first-pass:    0 -> {after.gemini_first_pass}")
    print(f"  evidence:      {before.gemini_evidence} -> {after.gemini_evidence}")
    print()
    if before.tavily > 0:
        reduction = (before.tavily - after.tavily) / before.tavily * 100
        print(f"Estimated Tavily reduction: {reduction:.0f}%")
    print()


if __name__ == "__main__":
    asyncio.run(main())
