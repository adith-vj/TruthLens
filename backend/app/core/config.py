"""
core/config.py — Application configuration via Pydantic Settings.

All secrets and external configuration are loaded from environment variables
(or a .env file in the working directory). SecretStr is used for API keys
so they are never serialized into logs, repr output, or JSON responses.

Usage:
    from app.core.config import settings

    key = settings.GOOGLE_FACTCHECK_API_KEY.get_secret_value()
"""

from __future__ import annotations

import json

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    All fields have safe defaults so the server starts cleanly without a
    .env file (useful for CI and initial scaffolding). API key fields default
    to empty strings — services that require them must validate presence before
    use and raise a descriptive ConfigurationError, not an unhandled exception.
    """

    model_config = SettingsConfigDict(
        # Resolve .env relative to the process working directory (backend/).
        # Run uvicorn from backend/ to ensure correct resolution.
        env_file=".env",
        env_file_encoding="utf-8",
        # Ignore extra fields present in .env that are not declared here.
        extra="ignore",
    )

    # --- External API keys ---
    # SecretStr ensures these values are masked in __repr__ and never
    # serialized into JSON or log output.
    GOOGLE_FACTCHECK_API_KEY: SecretStr = SecretStr("")
    GEMINI_API_KEY: SecretStr = SecretStr("")

    # --- CORS ---
    # In .env files, write as a JSON array string: ALLOWED_ORIGINS=["*"]
    # or ALLOWED_ORIGINS=["chrome-extension://abc","http://localhost"]
    ALLOWED_ORIGINS: list[str] = ["*"]

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_allowed_origins(cls, v: object) -> list[str]:
        """
        Accept ALLOWED_ORIGINS as either:
          - A list (already parsed by pydantic-settings)     → pass through
          - A JSON array string '["*"]'                      → parse and return
          - A bare wildcard string '*'                       → wrap in list
        """
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            stripped = v.strip()
            if stripped.startswith("["):
                return json.loads(stripped)
            return [stripped]
        return [str(v)]

    # --- Input validation ---
    MAX_CLAIM_LENGTH: int = 2000

    # --- Google Fact Check client ---
    # Timeout (seconds) for each HTTP request to the Google Fact Check API.
    # Set low enough to avoid blocking FastAPI workers on slow upstream responses.
    FACTCHECK_TIMEOUT_SECONDS: float = 5.0
    # Maximum number of claims to request per API call (pageSize parameter).
    # The first matching claim is used; more than 5 is rarely beneficial.
    FACTCHECK_MAX_RESULTS: int = 5

    # --- Gemini classifier client ---
    # Timeout (seconds) for each HTTP request to the Gemini API.
    # Increased to 10.0s to accommodate implicit chain-of-thought generation.
    CLASSIFIER_TIMEOUT_SECONDS: float = 10.0

    # --- Groq fallback client ---
    # Used for video claim extraction when Gemini quota is exhausted or request fails.
    GROQ_API_KEY: SecretStr = SecretStr("")
    # Model for Groq fallback claim extraction. Llama-3.3-70b-versatile is excellent for structured tasks.
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    # Target max input tokens for Groq to avoid 413s on restricted tiers (e.g. openai/gpt-oss-120b limit is 8000)
    GROQ_MAX_INPUT_TOKENS: int = 5000

    # --- Tavily web search client (Phase 4) ---
    # API key for the Tavily AI search API.  Obtain a free key (1,000 credits/month,
    # no credit card required) at https://app.tavily.com
    # search_depth is always "basic" (1 credit/request); "advanced" costs 2 credits.
    TAVILY_API_KEY: SecretStr = SecretStr("")
    # Timeout (seconds) for each Tavily search request.
    # Web search is slower than a classification call; 8 s is a safe default.
    SEARCH_TIMEOUT_SECONDS: float = 8.0
    # Maximum number of results to request per Tavily search.
    # Each result = 1 Tavily credit with search_depth="basic".
    SEARCH_MAX_RESULTS: int = 5

    # --- LLM verifier client (Phase 4) ---
    # Uses the same GEMINI_API_KEY as the classifier.
    # Separate timeout because the verifier call is heavier (JSON-mode output).
    LLM_TIMEOUT_SECONDS: float = 15.0

    # --- Video verification (Phase 5.5) ---
    # Gemini first-pass: minimum confidence required to skip Tavily.
    # All four eligibility conditions must be satisfied; this is one of them.
    GEMINI_FIRST_PASS_CONFIDENCE_THRESHOLD: float = 0.80
    # Niche-claim heuristic: claims whose Phase-5.4 checkability_score is at
    # or above this value are considered unusually specific and are always
    # escalated to Tavily even when Gemini returns high confidence.
    # 0.90 is intentionally conservative — ordinary claims score 0.70–0.85.
    NICHE_CLAIM_CHECKABILITY_THRESHOLD: float = 0.90
    # Pipeline version string included in the video verification cache key.
    # Bump this whenever prompt, thresholds, or escalation logic changes so
    # that old cached results are automatically invalidated.
    VIDEO_VERIFICATION_PIPELINE_VERSION: str = "v2"

    # --- Logging ---
    LOG_LEVEL: str = "INFO"

    # --- App metadata ---
    APP_ENV: str = "development"


# Module-level singleton. Import this throughout the application.
# Tests can override individual values via monkeypatch before importing
# modules that depend on settings (see tests/conftest.py).
settings = Settings()
