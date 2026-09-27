"""Rule-based handling for semantically not-applicable missing values."""

from __future__ import annotations

import logging

import pandas as pd

LOGGER = logging.getLogger(__name__)


def fill_not_applicable_values(
    data_frame: pd.DataFrame,
    fill_map: dict[str, str],
) -> pd.DataFrame:
    """Fill configured missing values with explicit not-applicable labels.

    Args:
        data_frame: Dataset to copy and update.
        fill_map: Mapping from column name to configured fill value.

    Returns:
        A copied DataFrame with configured nulls filled.

    Raises:
        KeyError: If a configured column is absent from the DataFrame.
    """
    cleaned_df = data_frame.copy()

    for column, fill_value in fill_map.items():
        if column not in cleaned_df.columns:
            raise KeyError(f"Configured fill column is missing: {column}")

        missing_before = int(cleaned_df[column].isna().sum())
        cleaned_df[column] = cleaned_df[column].fillna(fill_value)
        LOGGER.info(
            "Filled %d missing value(s) in %s with %r",
            missing_before,
            column,
            fill_value,
        )

    return cleaned_df


def check_remaining_missing_values(
    data_frame: pd.DataFrame,
    threshold: float,
) -> list[str]:
    """Return columns whose remaining null rate exceeds a threshold.

    Args:
        data_frame: Dataset after configured fills.
        threshold: Maximum permitted null-rate fraction.

    Returns:
        Column names whose null rate is above ``threshold``.

    Raises:
        ValueError: If threshold is outside [0, 1].
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    null_rates = data_frame.isna().mean()
    remaining_columns = []
    for column, null_rate in null_rates.items():
        if null_rate > threshold:
            remaining_columns.append(column)

    if remaining_columns:
        LOGGER.warning(
            "Columns remain above the %.2f null-rate threshold: %s",
            threshold,
            remaining_columns,
        )
    else:
        LOGGER.info("No columns remain above the %.2f null-rate threshold", threshold)

    return remaining_columns
