"""Validation-only comparison of customer clustering candidates."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.cluster import HDBSCAN
from sklearn.metrics import davies_bouldin_score, silhouette_score

LOGGER = logging.getLogger(__name__)


def evaluate_clustering_candidates(
    candidates: dict[str, object],
    preprocessing_pipeline: ColumnTransformer,
    train_features: pd.DataFrame,
    validation_features: pd.DataFrame,
) -> pd.DataFrame:
    """Fit clustering candidates on train and score assignments on validation.

    The preprocessing pipeline is fitted only on ``train_features`` and then
    reused unchanged to transform ``validation_features``.

    Args:
        candidates: Unfitted clustering estimators keyed by model name.
        preprocessing_pipeline: ColumnTransformer for customer features.
        train_features: Customer training feature table.
        validation_features: Customer validation feature table.

    Returns:
        Validation metrics for every candidate.
    """
    transformed_train = preprocessing_pipeline.fit_transform(train_features)
    transformed_validation = preprocessing_pipeline.transform(validation_features)
    comparison_rows = []
    for model_name, candidate in candidates.items():
        candidate.fit(transformed_train)
        validation_labels = candidate.predict(transformed_validation)
        silhouette = silhouette_score(transformed_validation, validation_labels)
        davies_bouldin = davies_bouldin_score(
            transformed_validation,
            validation_labels,
        )
        cluster_count = getattr(
            candidate,
            "n_clusters",
            None,
        )
        if cluster_count is None:
            cluster_count = getattr(candidate, "n_components")
        comparison_rows.append(
            {
                "model_name": model_name,
                "n_clusters_or_components": cluster_count,
                "silhouette_score": silhouette,
                "davies_bouldin_score": davies_bouldin,
                "scoring_scope": "validation",
            }
        )
        LOGGER.info(
            "Segmentation candidate %s: silhouette=%.6f, davies_bouldin=%.6f",
            model_name,
            silhouette,
            davies_bouldin,
        )

    return pd.DataFrame(comparison_rows)


def evaluate_agglomerative_candidates(
    candidates: dict[str, object],
    preprocessing_pipeline: ColumnTransformer,
    train_features: pd.DataFrame,
    connectivity: object | None = None,
) -> pd.DataFrame:
    """Score agglomerative candidates on the data used to fit them.

    AgglomerativeClustering has no ``predict`` method, so these scores are
    explicitly train-fit metrics and are not comparable to validation scores.

    Args:
        candidates: Unfitted agglomerative estimators keyed by model name.
        preprocessing_pipeline: Numeric preprocessing transformer.
        train_features: Customer training feature table.
        connectivity: Optional sparse neighbor graph used to make Ward
            linkage tractable without removing training rows.

    Returns:
        Train-fit silhouette and Davies-Bouldin metrics for each candidate.
    """
    transformed_train = preprocessing_pipeline.fit_transform(train_features)
    comparison_rows = []
    for model_name, candidate in candidates.items():
        if connectivity is not None:
            candidate.set_params(connectivity=connectivity)
        candidate.fit(transformed_train)
        labels = candidate.labels_
        comparison_rows.append(
            {
                "model_name": model_name,
                "n_clusters_or_components": candidate.n_clusters,
                "silhouette_score": silhouette_score(
                    transformed_train,
                    labels,
                ),
                "davies_bouldin_score": davies_bouldin_score(
                    transformed_train,
                    labels,
                ),
                "scoring_scope": "train_fit",
            }
        )
        LOGGER.info(
            "Agglomerative candidate %s: train-fit silhouette=%.6f, "
            "davies_bouldin=%.6f",
            model_name,
            comparison_rows[-1]["silhouette_score"],
            comparison_rows[-1]["davies_bouldin_score"],
        )

    return pd.DataFrame(comparison_rows)


def evaluate_hdbscan_candidates(
    candidates: dict[str, HDBSCAN],
    preprocessing_pipeline: ColumnTransformer,
    train_features: pd.DataFrame,
) -> pd.DataFrame:
    """Score HDBSCAN candidates on their fit data after excluding noise.

    HDBSCAN does not expose a general ``predict`` method in scikit-learn, so
    these metrics are fit-scope diagnostics. Noise labels (``-1``) are
    excluded from both clustering metrics, and the retained row count is
    reported for interpretation.

    Args:
        candidates: Unfitted HDBSCAN estimators keyed by model name.
        preprocessing_pipeline: Numeric preprocessing transformer.
        train_features: Customer feature table used for fitting.

    Returns:
        Fit-scope HDBSCAN metrics for every candidate.

    Raises:
        ValueError: If a candidate produces fewer than two non-noise clusters.
    """
    transformed_train = preprocessing_pipeline.fit_transform(train_features)
    comparison_rows = []
    for model_name, candidate in candidates.items():
        candidate.fit(transformed_train)
        labels = candidate.labels_
        retained_mask = labels != -1
        retained_features = transformed_train[retained_mask]
        retained_labels = labels[retained_mask]
        cluster_count = len(set(retained_labels))
        if cluster_count < 2:
            raise ValueError(
                f"HDBSCAN candidate {model_name} produced fewer than two "
                "non-noise clusters"
            )
        comparison_rows.append(
            {
                "model_name": model_name,
                "n_clusters_or_components": cluster_count,
                "silhouette_score": silhouette_score(
                    retained_features,
                    retained_labels,
                ),
                "davies_bouldin_score": davies_bouldin_score(
                    retained_features,
                    retained_labels,
                ),
                "scoring_scope": "train_fit_non_noise",
                "retained_rows": int(retained_mask.sum()),
                "noise_rows": int((~retained_mask).sum()),
            }
        )
        LOGGER.info(
            "HDBSCAN candidate %s: silhouette=%.6f, davies_bouldin=%.6f, "
            "noise_rows=%d",
            model_name,
            comparison_rows[-1]["silhouette_score"],
            comparison_rows[-1]["davies_bouldin_score"],
            comparison_rows[-1]["noise_rows"],
        )

    return pd.DataFrame(comparison_rows)
