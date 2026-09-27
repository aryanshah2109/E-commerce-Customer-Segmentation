"""Health and model-readiness endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request

from api.schema.common import HealthResponse
from api.services.model_registry import ModelRegistry

router = APIRouter(tags=["health"])


def _registry(request: Request) -> ModelRegistry:
    return request.app.state.model_registry


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Return liveness and current model readiness."""
    model_loaded, trained_at_utc = _registry(request).health()
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_trained_at_utc=trained_at_utc,
    )
