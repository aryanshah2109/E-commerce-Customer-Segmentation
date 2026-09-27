"""Evaluation and promotion helpers for the production segmentation model."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from sklearn.metrics import davies_bouldin_score, silhouette_score


def evaluate_clustering_model(
    transformed_features: np.ndarray,
    cluster_labels: np.ndarray,
) -> dict[str, float]:
    """Calculate unsupervised quality metrics for fitted cluster labels.

    Args:
        transformed_features: Features used by the clustering estimator.
        cluster_labels: Cluster assignment for each feature row.

    Returns:
        Silhouette and Davies-Bouldin scores.

    Raises:
        ValueError: If labels are not aligned or contain fewer than two
            clusters.
    """
    if len(transformed_features) != len(cluster_labels):
        raise ValueError(
            "Transformed features and cluster labels must have equal length"
        )

    cluster_count = len(np.unique(cluster_labels))
    if cluster_count < 2:
        raise ValueError("At least two clusters are required for evaluation")

    return {
        "silhouette_score": float(
            silhouette_score(transformed_features, cluster_labels)
        ),
        "davies_bouldin_score": float(
            davies_bouldin_score(transformed_features, cluster_labels)
        ),
    }


def is_candidate_better(
    candidate_metrics: Mapping[str, float],
    stored_metrics: Mapping[str, float] | None,
) -> bool:
    """Determine whether a candidate should replace the stored best model.

    Silhouette is the primary metric and higher is better. Davies-Bouldin is
    used as a lower-is-better tie-breaker.

    Args:
        candidate_metrics: Metrics for the newly trained model.
        stored_metrics: Metrics for the currently promoted model, if present.

    Returns:
        True when the candidate should be promoted.
    """
    if stored_metrics is None:
        return True

    candidate_silhouette = float(candidate_metrics["silhouette_score"])
    stored_silhouette = float(stored_metrics["silhouette_score"])
    if candidate_silhouette != stored_silhouette:
        return candidate_silhouette > stored_silhouette

    return float(candidate_metrics["davies_bouldin_score"]) < float(
        stored_metrics["davies_bouldin_score"]
    )
