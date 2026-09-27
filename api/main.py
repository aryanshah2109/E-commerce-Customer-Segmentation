"""FastAPI application factory for customer and sales analytics."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import Settings, get_settings
from api.routers import health, pipeline, prediction
from api.services.model_registry import ModelRegistry

LOGGER = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create the model registry and attempt a non-fatal initial load."""
    settings: Settings = app.state.settings
    registry = ModelRegistry(settings)
    app.state.model_registry = registry
    try:
        registry.reload_if_stale()
    except Exception as error:
        LOGGER.warning("API started without a loaded model: %s", error)
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the configured FastAPI application."""
    resolved_settings = settings or get_settings()
    app = FastAPI(
        title="Customer Sales Analytics API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_settings.cors_allow_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router, prefix="/api/v1")
    app.include_router(prediction.router, prefix="/api/v1")
    if resolved_settings.enable_pipeline:
        app.include_router(pipeline.router, prefix="/api/v1")
    return app


app = create_app()
