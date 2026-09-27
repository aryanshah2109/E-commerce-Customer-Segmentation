"""Service for executing and reporting a complete retraining pipeline."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import anyio

from customer_sales_analytics.pipelines.main_pipeline import run_pipeline

from api.config import Settings, resolve_api_paths
from api.schema.pipeline import (
    PipelineRunRequest,
    PipelineRunResponse,
    SegmentationMetrics,
)
from api.services.model_registry import ModelRegistry

LOGGER = logging.getLogger(__name__)


def _latest_pipeline_log() -> Path | None:
    logs = sorted(Path("logs").glob("pipeline_*.log"), key=lambda path: path.stat().st_mtime)
    return logs[-1] if logs else None


def _failed_stage(log_path: Path | None) -> str | None:
    if log_path is None or not log_path.is_file():
        return None
    content = log_path.read_text(encoding="utf-8", errors="replace")
    failed_match = re.findall(r"Stage failed: ([^(]+)", content)
    if failed_match:
        return failed_match[-1].strip()
    started = re.findall(r"Starting stage: (.+)", content)
    return started[-1].strip() if started else None


def _empty_metrics() -> SegmentationMetrics:
    return SegmentationMetrics(
        model_name="not_available",
        silhouette_score=0.0,
        davies_bouldin_score=0.0,
        cluster_sizes={},
        promoted_to_best_model=False,
        training_rows=0,
        model_path="",
        cluster_profile_path="",
    )


def _read_best_metrics(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


async def run_full_pipeline(
    request: PipelineRunRequest,
    settings: Settings,
    model_registry: ModelRegistry,
) -> PipelineRunResponse:
    """Run every batch stage in a worker thread and parse its artifacts."""
    data_config_path = Path(request.data_config_path or settings.data_config_path)
    paths_config_path = Path(request.paths_config_path or settings.paths_config_path)
    preprocessing_config_path = Path(
        request.preprocessing_config_path or settings.preprocessing_config_path
    )
    resolved_paths = resolve_api_paths(paths_config_path)
    before_best_metrics = _read_best_metrics(resolved_paths.best_metrics)
    before_trained_at = before_best_metrics.get("trained_at_utc")
    start = time.perf_counter()

    try:
        with anyio.fail_after(settings.pipeline_timeout_seconds):
            return_code = await anyio.to_thread.run_sync(
                run_pipeline,
                data_config_path,
                paths_config_path,
                preprocessing_config_path,
            )
    except TimeoutError:
        log_path = _latest_pipeline_log()
        return PipelineRunResponse(
            status="failed",
            failed_stage="timeout",
            duration_seconds=time.perf_counter() - start,
            log_path=str(log_path or ""),
            segmentation_metrics=_empty_metrics(),
        )

    log_path = _latest_pipeline_log()
    duration_seconds = time.perf_counter() - start
    if return_code != 0:
        return PipelineRunResponse(
            status="failed",
            failed_stage=_failed_stage(log_path),
            duration_seconds=duration_seconds,
            log_path=str(log_path or ""),
            segmentation_metrics=_empty_metrics(),
        )

    report = json.loads(
        resolved_paths.segmentation_training_report.read_text(encoding="utf-8")
    )
    metrics = report["metrics"]
    after_best_metrics = _read_best_metrics(resolved_paths.best_metrics)
    promoted = (
        report["model_path"]
        == str(resolved_paths.best_model)
        or after_best_metrics.get("trained_at_utc") != before_trained_at
    )
    response_metrics = SegmentationMetrics(
        model_name=str(report["model"]),
        silhouette_score=float(metrics["silhouette_score"]),
        davies_bouldin_score=float(metrics["davies_bouldin_score"]),
        cluster_sizes={
            str(key): int(value)
            for key, value in report["cluster_sizes"].items()
        },
        promoted_to_best_model=promoted,
        training_rows=int(report["training_rows"]),
        model_path=str(report["model_path"]),
        cluster_profile_path=str(report["cluster_profile_path"]),
    )
    model_registry.configure_paths(paths_config_path)
    model_registry.reload_if_stale()
    return PipelineRunResponse(
        status="success",
        failed_stage=None,
        duration_seconds=duration_seconds,
        log_path=str(log_path or ""),
        segmentation_metrics=response_metrics,
    )
