"""
app/main.py — FastAPI application factory for the TruthLens verification backend.

Entry point for the verification service. The existing image-detection endpoints
live in backend/main.py and are NOT modified or imported here.

To run the development server:
    cd backend/
    uvicorn app.main:app --reload --port 8001

The 'app' object is the ASGI application instance created by create_app().
It is exposed at module level so uvicorn can import it directly.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi import status as http_status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.verify import router as verify_router
from app.core.config import settings
from app.core.logging import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):  # type: ignore[type-arg]
    """
    FastAPI lifespan context manager.

    Startup tasks:
        - Configure structured logging (idempotent).

    Shutdown tasks:
        - (none in scaffold phase; add cleanup here in Phase 5)
    """
    configure_logging()
    logger.info(
        "TruthLens verification backend starting up "
        "(env=%s, log_level=%s, max_claim_length=%d)",
        settings.APP_ENV,
        settings.LOG_LEVEL,
        settings.MAX_CLAIM_LENGTH,
    )
    yield
    logger.info("TruthLens verification backend shutting down")


def create_app() -> FastAPI:
    """
    Application factory — creates and configures the FastAPI instance.

    Using a factory function (rather than a bare module-level app = FastAPI())
    makes the app easy to instantiate with different configurations in tests.

    Returns:
        A fully configured FastAPI application instance.
    """
    application = FastAPI(
        title="TruthLens Verification API",
        description=(
            "Backend API for the TruthLens Chrome Extension. "
            "Provides real-time claim verification via a multi-layer pipeline: "
            "claim classification → Google Fact Check → LLM/search fallback."
        ),
        version="0.1.0",
        # OpenAPI docs paths (disable in production if desired)
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # --- CORS middleware ---
    # ALLOWED_ORIGINS is configurable via the ALLOWED_ORIGINS env variable.
    # For production, set it to the specific Chrome Extension origin:
    #   chrome-extension://<your-extension-id>
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["POST", "GET", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )

    # --- Global exception handler ---
    # Catches any unhandled exception and returns a safe 500 response.
    # Stack traces, API keys, and internal paths are never included.
    @application.exception_handler(Exception)
    async def global_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.error(
            "Unhandled exception on %s %s: %s",
            request.method,
            request.url.path,
            type(exc).__name__,
            exc_info=True,  # Full traceback goes to logs only, not the response
        )
        return JSONResponse(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "internal server error"},
        )

    # --- Routers ---
    application.include_router(verify_router, prefix="/api", tags=["Verification"])
    
    # Import and register legacy router without prefix to preserve old URLs
    from app.api.legacy import router as legacy_router
    application.include_router(legacy_router, tags=["Legacy"])

    # --- Health check endpoint ---
    @application.get(
        "/health",
        summary="Health check",
        description="Returns the current health status and environment of the API.",
        tags=["Health"],
    )
    async def health() -> dict[str, Any]:
        return {"status": "ok", "env": settings.APP_ENV}

    return application


# Module-level app instance — imported by uvicorn.
# uvicorn app.main:app --reload --port 8001
app = create_app()
