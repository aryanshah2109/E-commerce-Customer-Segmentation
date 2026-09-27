"""Manual execution of the existing full batch pipeline."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from customer_sales_analytics.pipelines.main_pipeline import run_pipeline
from streamlit_app.config import PROJECT_ROOT


def _latest_log() -> Path | None:
    logs = sorted(
        (PROJECT_ROOT / "logs").glob("pipeline_*.log"),
        key=lambda path: path.stat().st_mtime,
    )
    return logs[-1] if logs else None


def run_full_pipeline() -> dict[str, Any]:
    """Run the existing orchestrator and parse its generated report."""
    start = time.perf_counter()
    result_code = run_pipeline(
        PROJECT_ROOT / "configs" / "data_config.yaml",
        PROJECT_ROOT / "configs" / "paths.yaml",
        PROJECT_ROOT / "configs" / "preprocessing_config.yaml",
    )
    duration = time.perf_counter() - start
    log_path = _latest_log()
    report_path = PROJECT_ROOT / "artifacts" / "reports" / (
        "segmentation_training_report.json"
    )
    if result_code != 0 or not report_path.is_file():
        return {
            "status": "failed",
            "duration_seconds": duration,
            "log_path": str(log_path or ""),
            "failed_stage": "pipeline stage failed",
        }

    report = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = report["metrics"]
    return {
        "status": "success",
        "duration_seconds": duration,
        "log_path": str(log_path or ""),
        "failed_stage": None,
        "model_name": report["model"],
        "silhouette_score": metrics["silhouette_score"],
        "davies_bouldin_score": metrics["davies_bouldin_score"],
        "training_rows": report["training_rows"],
        "cluster_sizes": report["cluster_sizes"],
        "model_path": report["model_path"],
        "cluster_profile_path": report["cluster_profile_path"],
    }
