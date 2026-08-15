"""
services/classifier.py — Claim type classification interface.

SCAFFOLDING PHASE: This file defines the data contract for the future claim
classifier service. No implementation logic is present. This module is NOT
imported or invoked by any route handler in the current scaffold.

--- Future implementation task (Phase 3) ---

The classifier determines whether a piece of text represents a verifiable
factual claim before sending it through the verification pipeline.

Examples:
    "The Earth has one moon."          → FACTUAL_CLAIM   (proceed to verify)
    "I think this movie is boring."    → OPINION         (return unverifiable)
    "Buy this product now!"            → ADVERTISEMENT   (return unverifiable)
    "What is the population of India?" → AMBIGUOUS       (proceed, low confidence)

Implementation approaches to evaluate in Phase 3:
    1. Zero-shot classification via Gemini (e.g., "Is this a factual claim?")
    2. Dedicated HuggingFace classifier fine-tuned on claim detection
    3. Rule-based heuristics as a lightweight fallback

The function must be async-compatible for use inside FastAPI route handlers.
"""

from __future__ import annotations

from enum import Enum


class ClaimType(str, Enum):
    """
    The category of an input text as determined by the claim classifier.

    Used by the verification pipeline to decide whether to proceed with
    fact-checking or to return an 'unverifiable' response immediately.

    Values:
        FACTUAL_CLAIM:  Text makes a specific, verifiable assertion of fact.
                        Proceed to Google Fact Check and/or LLM verification.
        OPINION:        Text expresses a subjective view or preference.
                        Return verdict='unverifiable' without querying sources.
        ADVERTISEMENT:  Text is a call to action, promotional, or commercial.
                        Return verdict='unverifiable' without querying sources.
        AMBIGUOUS:      Text cannot be clearly classified. Proceed with lower
                        confidence weighting.
    """

    FACTUAL_CLAIM = "factual_claim"
    OPINION = "opinion"
    ADVERTISEMENT = "advertisement"
    AMBIGUOUS = "ambiguous"


# ---------------------------------------------------------------------------
# Future function signature — DO NOT implement until Phase 3
# ---------------------------------------------------------------------------
#
# async def classify_claim(text: str) -> ClaimType:
#     """
#     Classify the input text as a factual claim, opinion, advertisement,
#     or ambiguous.
#
#     Args:
#         text: The raw user-highlighted text to classify.
#
#     Returns:
#         A ClaimType value indicating how the pipeline should proceed.
#
#     Raises:
#         ServiceUnavailableError: If the underlying classifier is unreachable.
#     """
#     ...
