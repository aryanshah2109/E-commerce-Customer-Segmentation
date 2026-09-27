"""Small YAML configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a mapping from a YAML configuration file.

    Args:
        path: YAML file path.

    Returns:
        Parsed configuration mapping.

    Raises:
        ValueError: If YAML does not contain a mapping.
    """
    with path.open(encoding="utf-8") as file_handle:
        configuration = yaml.safe_load(file_handle) or {}
    if not isinstance(configuration, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return configuration
