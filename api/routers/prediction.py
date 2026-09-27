"""Inference endpoints for customer segmentation."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from api.schema.common import ErrorResponse
from api.schema.prediction import PredictRequest, PredictResponse
from api.services.model_registry import ModelNotAvailableError, ModelRegistry
from api.services.prediction_service import predict_clusters

LOGGER = logging.getLogger(__name__)
router = APIRouter(tags=["prediction"])


def _registry(request: Request) -> ModelRegistry:
    return request.app.state.model_registry


@router.post("/predict", response_model=PredictResponse)
def predict(request: Request, payload: PredictRequest) -> PredictResponse:
    """Assign supplied engineered customer rows to existing clusters."""
    registry = _registry(request)
    try:
        registry.reload_if_stale()
        assignments = predict_clusters(payload.customers, registry)
        _, _, _, trained_at_utc = registry.get_bundle()
        return PredictResponse(
            predictions=assignments,
            model_trained_at_utc=trained_at_utc,
        )
    except ModelNotAvailableError as error:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ErrorResponse(
                detail=str(error),
                error_code="MODEL_NOT_AVAILABLE",
            ).model_dump(),
        )
    except Exception as error:
        LOGGER.exception("Prediction request failed: %s", error)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                detail="Prediction failed.",
                error_code="PREDICTION_FAILED",
            ).model_dump(),
        )

