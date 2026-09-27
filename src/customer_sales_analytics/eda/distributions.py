"""Checklist steps 2, 4, and 5: types, distributions, and outliers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DistributionResults:
    """Numeric summaries, category counts, outliers, and variance flags."""

    numeric_summary: pd.DataFrame
    categorical_counts: dict[str, pd.Series]
    outlier_counts: dict[str, int]
    near_zero_variance_columns: list[str]
    numeric_categorical_candidates: list[str]


def compute_distributions(
    data_frame: pd.DataFrame,
    low_cardinality_max: int,
) -> DistributionResults:
    """Compute univariate summaries and simple IQR outlier counts.

    Numeric codes that actually represent categories should be reviewed by the
    team after seeing the numeric output.  Text columns are treated as
    categories for value counts.

    Args:
        data_frame: Input sales data.
        low_cardinality_max: Maximum category count for value-count tables.

    Returns:
        Numeric summary, categorical counts, IQR outlier counts, and near-zero
        variance columns.

    Raises:
        ValueError: If low_cardinality_max is not positive.
    """
    if low_cardinality_max <= 0:
        raise ValueError("low_cardinality_max must be positive")

    numeric_df = data_frame.select_dtypes(include="number")
    numeric_summary = numeric_df.describe().T

    categorical_counts: dict[str, pd.Series] = {}
    categorical_df = data_frame.select_dtypes(
        include=["object", "category", "bool"]
    )
    for column in categorical_df.columns:
        cardinality = data_frame[column].nunique(dropna=False)
        if cardinality <= low_cardinality_max:
            value_counts = data_frame[column].value_counts(dropna=False)
            categorical_counts[column] = value_counts

    outlier_counts: dict[str, int] = {}
    for column in numeric_df.columns:
        series = numeric_df[column].dropna()
        if series.empty:
            outlier_counts[column] = 0
            continue

        first_quartile = series.quantile(0.25)
        third_quartile = series.quantile(0.75)
        interquartile_range = third_quartile - first_quartile
        lower_bound = first_quartile - 1.5 * interquartile_range
        upper_bound = third_quartile + 1.5 * interquartile_range
        is_outlier = (series < lower_bound) | (series > upper_bound)
        outlier_counts[column] = int(is_outlier.sum())

    near_zero_variance_columns = []
    for column in numeric_df.columns:
        unique_value_count = numeric_df[column].nunique(dropna=True)
        if unique_value_count <= 1:
            near_zero_variance_columns.append(column)

    numeric_categorical_candidates = []
    for column in numeric_df.columns:
        unique_value_count = numeric_df[column].nunique(dropna=True)
        if unique_value_count <= low_cardinality_max:
            numeric_categorical_candidates.append(column)

    return DistributionResults(
        numeric_summary=numeric_summary,
        categorical_counts=categorical_counts,
        outlier_counts=outlier_counts,
        near_zero_variance_columns=near_zero_variance_columns,
        numeric_categorical_candidates=numeric_categorical_candidates,
    )
