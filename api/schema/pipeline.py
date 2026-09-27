"""Pipeline execution request and response schemas."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class PipelineRunRequest(BaseModel):
    """Optional configuration-path overrides for a full retraining run."""

    data_config_path: str | None = None
    paths_config_path: str | None = None
    preprocessing_config_path: str | None = None


class SegmentationMetrics(BaseModel):
    """Metrics and artifact locations produced by segmentation training."""

    model_name: str
    silhouette_score: float
    davies_bouldin_score: float
    cluster_sizes: dict[str, int]
    promoted_to_best_model: bool
    training_rows: int
    model_path: str
    cluster_profile_path: str


class PipelineRunResponse(BaseModel):
    """Result of a complete batch pipeline execution."""

    status: Literal["success", "failed"]
    failed_stage: str | None
    duration_seconds: float
    log_path: str
    segmentation_metrics: SegmentationMetrics
