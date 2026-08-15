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

    # --- Logging ---
    LOG_LEVEL: str = "INFO"

    # --- App metadata ---
    APP_ENV: str = "development"


# Module-level singleton. Import this throughout the application.
# Tests can override individual values via monkeypatch before importing
# modules that depend on settings (see tests/conftest.py).
settings = Settings()
