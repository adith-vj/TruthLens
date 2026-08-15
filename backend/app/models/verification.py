"""
models/verification.py — Pydantic v2 request and response schemas.

These models define the public API contract for POST /api/verify.
All validation is declared in the schema, not in the route handler,
so it is enforced consistently at both input and output boundaries.

Schema rules:
  - VerifyRequest.text   must be non-empty and non-whitespace-only
  - VerifyResponse.verdict            must be one of the four Literal values
  - VerifyResponse.confidence_score   must be in [0.0, 1.0]
  - VerifyResponse.sources            may be an empty list
  - SourceItem.url                    must be a valid URL string
"""

from __future__ import annotations

from typing import Literal

from pydantic import AnyUrl, BaseModel, Field, field_validator


class SourceItem(BaseModel):
    """
    A single evidence source returned with a verification result.

    Fields:
        title:     Human-readable title of the source article or page.
        url:       Validated URL of the source. Must be a real URL returned
                   by an actual search or fact-check API — never fabricated.
        publisher: Name of the publishing organization.
    """

    title: str
    url: AnyUrl
    publisher: str


# The exhaustive set of allowed verdict strings.
# Using Literal instead of Enum produces a cleaner JSON schema and catches
# invalid values at the Pydantic model layer rather than in route logic.
VerdictType = Literal["true", "false", "misleading", "unverifiable"]


class VerifyRequest(BaseModel):
    """
    Request body for POST /api/verify.

    Fields:
        text: The user-highlighted claim to be verified. Must be a non-empty,
              non-whitespace-only string. Maximum length is enforced at the
              route handler level using settings.MAX_CLAIM_LENGTH so that
              the limit is configurable without changing schema code.
    """

    text: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        """Reject empty strings and whitespace-only strings."""
        if not v.strip():
            raise ValueError("text must not be empty or whitespace-only")
        return v


class VerifyResponse(BaseModel):
    """
    Response schema for POST /api/verify.

    This is the strict public contract. No additional fields may be added
    merely to indicate scaffolding or placeholder state — that information
    belongs in documentation and tests, not in the API surface.

    Fields:
        verdict:          One of "true", "false", "misleading", "unverifiable".
        confidence_score: A float in [0.0, 1.0] representing certainty.
                          0.0 = no confidence; 1.0 = maximum confidence.
        sources:          List of evidence sources. May be empty when no
                          external sources were consulted or found.
    """

    verdict: VerdictType
    confidence_score: float = Field(ge=0.0, le=1.0)
    sources: list[SourceItem] = Field(default_factory=list)
