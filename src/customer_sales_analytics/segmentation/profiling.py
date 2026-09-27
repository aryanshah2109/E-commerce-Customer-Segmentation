"""Cluster profiling on the original customer feature representation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_cluster_profile(
    transformed_features_df: pd.DataFrame,
    original_features_df: pd.DataFrame,
    cluster_labels: np.ndarray,
) -> pd.DataFrame:
    """Summarize original customer features for each assigned cluster.

    Args:
        transformed_features_df: Transformed features used by the clustering
            estimator. It is used to validate the assignment row count.
        original_features_df: Untransformed customer feature table.
        cluster_labels: Cluster assignment for each customer row.

    Returns:
        One row per cluster containing cluster size, numeric means, and
        categorical modes.

    Raises:
        ValueError: If feature rows and cluster assignments do not align.
    """
    if len(transformed_features_df) != len(original_features_df):
        raise ValueError("Transformed and original features must have equal rows")
    if len(original_features_df) != len(cluster_labels):
        raise ValueError("Feature rows and cluster labels must have equal length")

    profile_df = original_features_df.copy()
    profile_df["cluster"] = cluster_labels
    numeric_columns = profile_df.select_dtypes(include="number").columns.tolist()
    categorical_columns = profile_df.select_dtypes(exclude="number").columns.tolist()

    profile_rows = []
    for cluster_label, cluster_df in profile_df.groupby("cluster", sort=True):
        profile_row: dict[str, object] = {
            "cluster": cluster_label,
            "cluster_size": len(cluster_df),
        }
        for column in numeric_columns:
            profile_row[column] = float(cluster_df[column].mean())
        for column in categorical_columns:
            modes = cluster_df[column].mode(dropna=False)
            profile_row[column] = modes.sort_values().iloc[0]
        profile_rows.append(profile_row)

    return pd.DataFrame(profile_rows)
