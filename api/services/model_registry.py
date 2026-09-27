"""Cached access to the promoted segmentation model artifacts."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Any

import joblib
import pandas as pd

from api.config import ApiPaths, Settings, resolve_api_paths

LOGGER = logging.getLogger(__name__)


class ModelNotAvailableError(RuntimeError):
    """Raised when the promoted model has not been created yet."""


class ModelRegistry:
    """Load and cache the promoted preprocessing/model/profile bundle."""

    def __init__(self, settings: Settings) -> None:
        """Initialize a registry without requiring a model to exist."""
        self.settings = settings
        self.paths = resolve_api_paths(settings.paths_config_path)
        self._bundle: tuple[Any, Any, pd.DataFrame, str | None] | None = None
        self._model_mtime_ns: int | None = None
        self._lock = RLock()

    def configure_paths(self, paths_config_path: Path) -> None:
        """Use paths from an optional pipeline-run override."""
        with self._lock:
            new_paths = resolve_api_paths(paths_config_path)
            if new_paths != self.paths:
                self.paths = new_paths
                self._bundle = None
                self._model_mtime_ns = None

    def _load(self) -> tuple[Any, Any, pd.DataFrame, str | None]:
        if not self.paths.best_model.is_file():
            raise ModelNotAvailableError(
                f"Promoted model is not available: {self.paths.best_model}"
            )
        if not self.paths.cluster_profile.is_file():
            raise ModelNotAvailableError(
                f"Cluster profile is not available: {self.paths.cluster_profile}"
            )

        try:
            bundle = joblib.load(self.paths.best_model)
            preprocessing = bundle["preprocessing"]
            model = bundle["model"]
            cluster_profile = pd.read_csv(self.paths.cluster_profile)
            trained_at_utc = None
            if self.paths.best_metrics.is_file():
                metrics = json.loads(
                    self.paths.best_metrics.read_text(encoding="utf-8")
                )
                trained_at_utc = metrics.get("trained_at_utc")
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ModelNotAvailableError(
                f"Promoted model artifacts could not be loaded: {error}"
            ) from error

        self._model_mtime_ns = self.paths.best_model.stat().st_mtime_ns
        self._bundle = (preprocessing, model, cluster_profile, trained_at_utc)
        LOGGER.info("Loaded promoted segmentation model from %s", self.paths.best_model)
        return self._bundle

    def get_bundle(self) -> tuple[Any, Any, pd.DataFrame, str | None]:
        """Return the loaded preprocessing, estimator, profile, and timestamp."""
        with self._lock:
            if self._bundle is None:
                return self._load()
            return self._bundle

    def reload_if_stale(self) -> tuple[Any, Any, pd.DataFrame, str | None]:
        """Reload artifacts when the promoted model file's mtime changes."""
        with self._lock:
            if not self.paths.best_model.is_file():
                self._bundle = None
                self._model_mtime_ns = None
                raise ModelNotAvailableError(
                    f"Promoted model is not available: {self.paths.best_model}"
                )
            current_mtime_ns = self.paths.best_model.stat().st_mtime_ns
            if self._bundle is None or current_mtime_ns != self._model_mtime_ns:
                return self._load()
            return self._bundle

    def health(self) -> tuple[bool, str | None]:
        """Return readiness state without raising when artifacts are absent."""
        try:
            bundle = self.reload_if_stale()
        except Exception as error:
            LOGGER.warning("Model readiness check failed: %s", error)
            return False, None
        return True, bundle[3]


@lru_cache(maxsize=1)
def get_model_registry() -> ModelRegistry:
    """Return the process-cached default model registry."""
    from api.config import get_settings

    return ModelRegistry(get_settings())
