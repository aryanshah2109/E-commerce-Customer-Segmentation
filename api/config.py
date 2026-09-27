"""Configuration and path resolution for the analytics API."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from customer_sales_analytics.config.io import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Runtime settings loaded from defaults and ``APP_`` environment values."""

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        extra="ignore",
    )

    data_config_path: Path = PROJECT_ROOT / "configs" / "data_config.yaml"
    paths_config_path: Path = PROJECT_ROOT / "configs" / "paths.yaml"
    preprocessing_config_path: Path = (
        PROJECT_ROOT / "configs" / "preprocessing_config.yaml"
    )
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "INFO"
    cors_allow_origins: list[str] = Field(default_factory=lambda: ["*"])
    pipeline_timeout_seconds: float = Field(default=3600.0, gt=0)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-cached API settings."""
    return Settings()


@dataclass(frozen=True)
class ApiPaths:
    """Absolute paths consumed by API services."""

    best_model: Path
    best_metrics: Path
    cluster_profile: Path
    segmentation_training_report: Path
    segmentation_model_comparison_report: Path


def resolve_api_paths(
    paths_config_path: Path,
    project_root: Path = PROJECT_ROOT,
) -> ApiPaths:
    """Resolve model and report paths from the project's YAML configuration.

    Args:
        paths_config_path: Path to ``configs/paths.yaml``.
        project_root: Base directory for relative configured paths.

    Returns:
        Absolute paths required by the API.
    """
    paths_config = load_yaml(paths_config_path)
    best_model_directory = Path(paths_config["models"]["best_model"])
    reports_directory = Path(paths_config["artifacts"]["reports"])

    def absolute(configured_path: Path) -> Path:
        if configured_path.is_absolute():
            return configured_path
        return (project_root / configured_path).resolve()

    best_model_directory = absolute(best_model_directory)
    reports_directory = absolute(reports_directory)
    return ApiPaths(
        best_model=best_model_directory / "best_model.joblib",
        best_metrics=best_model_directory / "best_metrics.json",
        cluster_profile=best_model_directory / "cluster_profile.csv",
        segmentation_training_report=(
            reports_directory / "segmentation_training_report.json"
        ),
        segmentation_model_comparison_report=(
            reports_directory / "segmentation_model_comparison_report.json"
        ),
    )
