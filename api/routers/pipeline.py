"""Full retraining endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Body, HTTPException, Request, status

from api.config import Settings
from api.schema.pipeline import PipelineRunRequest, PipelineRunResponse
from api.services.model_registry import ModelRegistry
from api.services.pipeline_service import run_full_pipeline

LOGGER = logging.getLogger(__name__)
router = APIRouter(tags=["pipeline"])


def _registry(request: Request) -> ModelRegistry:
    return request.app.state.model_registry


def _settings(request: Request) -> Settings:
    return request.app.state.settings


@router.post("/pipeline/run", response_model=PipelineRunResponse)
async def run_pipeline_endpoint(
    request: Request,
    payload: PipelineRunRequest | None = Body(default=None),
) -> PipelineRunResponse:
    """Run all batch stages and return the resulting segmentation metrics."""
    try:
        return await run_full_pipeline(
            payload or PipelineRunRequest(),
            _settings(request),
            _registry(request),
        )
    except TimeoutError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Pipeline execution timed out.",
        ) from error
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A configured pipeline file was not found.",
        ) from error
    except Exception as error:
        LOGGER.exception("Pipeline API request failed: %s", error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Pipeline execution could not be completed.",
        ) from error
