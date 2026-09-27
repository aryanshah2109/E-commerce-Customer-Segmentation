"""Persist EDA leakage candidates as configuration for later modeling phases."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


def write_modeling_exclusions(
    eda_summary_path: Path,
    output_path: Path,
) -> list[str]:
    """Copy possible leakage columns from EDA into preprocessing config.

    Args:
        eda_summary_path: Path to the consolidated EDA JSON report.
        output_path: YAML configuration path to update.

    Returns:
        Leakage column names exactly as listed in the EDA report.

    Raises:
        FileNotFoundError: If the EDA report is missing.
        KeyError: If the report lacks the leakage-column section.
        ValueError: If the YAML configuration is not a mapping.
    """
    with eda_summary_path.open(encoding="utf-8") as file_handle:
        eda_summary = json.load(file_handle)

    leakage_columns = eda_summary["possible_leakage_columns"]

    if output_path.is_file():
        with output_path.open(encoding="utf-8") as file_handle:
            configuration = yaml.safe_load(file_handle) or {}
    else:
        configuration = {}

    if not isinstance(configuration, dict):
        raise ValueError("Preprocessing configuration must be a mapping")

    configuration["modeling_exclude_columns"] = leakage_columns
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_handle:
        yaml.safe_dump(configuration, file_handle, sort_keys=False)

    return leakage_columns
