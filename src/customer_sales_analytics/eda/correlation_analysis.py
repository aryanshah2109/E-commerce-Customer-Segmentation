"""Checklist step 6: inspect Pearson correlations between numeric features."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CorrelationResults:
    """Pearson matrix and pairs above the configured absolute threshold."""

    pearson: pd.DataFrame
    flagged_pairs: list[dict[str, object]]


def compute_correlation_analysis(
    data_frame: pd.DataFrame,
    threshold: float,
) -> CorrelationResults:
    """Compute a Pearson matrix and notable numeric feature pairs.

    Args:
        data_frame: Input sales data.
        threshold: Absolute Pearson correlation threshold to flag.

    Returns:
        Pearson correlation results for numeric columns.

    Raises:
        ValueError: If threshold is outside [0, 1].
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    numeric_df = data_frame.select_dtypes(include="number")
    pearson = numeric_df.corr(method="pearson")
    flagged_pairs: list[dict[str, object]] = []
    columns = list(pearson.columns)

    for left_index, left_column in enumerate(columns):
        remaining_columns = columns[left_index + 1 :]

        for right_column in remaining_columns:
            correlation = pearson.loc[left_column, right_column]

            if pd.notna(correlation) and abs(float(correlation)) >= threshold:
                pair = {
                    "left": left_column,
                    "right": right_column,
                    "pearson": float(correlation),
                }
                flagged_pairs.append(pair)

    return CorrelationResults(
        pearson=pearson,
        flagged_pairs=flagged_pairs,
    )
