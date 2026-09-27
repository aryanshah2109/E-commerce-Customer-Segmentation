"""Shared API response schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Liveness and model-readiness response."""

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_trained_at_utc: str | None


class ErrorResponse(BaseModel):
    """Stable error response body."""

    detail: str
    error_code: str | None = None
