"""Dashboard configuration derived from the project's YAML files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from customer_sales_analytics.config.io import load_yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DashboardPaths:
    """Resolved input, artifact, and model paths for the dashboard."""

    raw_data: Path
    customer_features: Path
    sales_features: Path
    reports: Path
    eda: Path
    model_comparison: Path
    best_model: Path
    best_metrics: Path
    cluster_profile: Path


def _absolute(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def get_paths() -> DashboardPaths:
    """Load configured paths without duplicating project locations."""
    paths_config = load_yaml(PROJECT_ROOT / "configs" / "paths.yaml")
    data = paths_config["data"]
    artifacts = paths_config["artifacts"]
    models = paths_config["models"]
    best_model_directory = _absolute(Path(models["best_model"]))
    return DashboardPaths(
        raw_data=_absolute(Path(data["raw"])),
        customer_features=_absolute(Path(data["customer_features"])),
        sales_features=_absolute(Path(data["sales_features"])),
        reports=_absolute(Path(artifacts["reports"])),
        eda=_absolute(Path(artifacts["eda"])),
        model_comparison=_absolute(Path(artifacts["model_comparison"])),
        best_model=best_model_directory / "best_model.joblib",
        best_metrics=best_model_directory / "best_metrics.json",
        cluster_profile=best_model_directory / "cluster_profile.csv",
    )
