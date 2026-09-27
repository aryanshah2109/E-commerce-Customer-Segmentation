"""Checklist step 2: inspect categorical columns and their frequencies."""

from __future__ import annotations

import pandas as pd


def compute_categorical_analysis(
    data_frame: pd.DataFrame,
    rare_threshold: float,
    top_n: int,
) -> dict[str, object]:
    """Compute cardinality and common values for categorical columns.

    ``rare_threshold`` remains for compatibility with the original caller.
    Rare-category detection is intentionally not part of this simple EDA pass.

    Args:
        data_frame: Input sales data.
        rare_threshold: Compatibility argument, not used in this pass.
        top_n: Number of frequent values to return for each column.

    Returns:
        Cardinality and top values for each categorical column.

    Raises:
        ValueError: If an argument is invalid.
    """
    if not 0 <= rare_threshold <= 1:
        raise ValueError("rare_threshold must be between 0 and 1")

    if top_n <= 0:
        raise ValueError("top_n must be positive")

    categorical_columns = data_frame.select_dtypes(
        include=["object", "category", "bool"]
    ).columns
    result: dict[str, object] = {}

    for column in categorical_columns:
        value_counts = data_frame[column].value_counts(dropna=False)
        column_result = {
            "cardinality": int(value_counts.size),
            "top": value_counts.head(top_n).to_dict(),
        }
        result[column] = column_result

    return result
