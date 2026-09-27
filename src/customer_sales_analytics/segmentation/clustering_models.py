"""Candidate clustering model construction."""

from __future__ import annotations

from sklearn.cluster import AgglomerativeClustering, HDBSCAN, KMeans
from sklearn.mixture import GaussianMixture


def build_baseline_model(n_clusters: int) -> KMeans:
    """Build the configured plain KMeans segmentation baseline.

    Args:
        n_clusters: Number of baseline clusters.

    Returns:
        An unfitted KMeans estimator.
    """
    return KMeans(n_clusters=n_clusters, n_init="auto")


def build_kmeans_candidates(
    cluster_counts: list[int],
    seed: int,
) -> dict[str, KMeans]:
    """Build seeded KMeans candidates for each configured cluster count.

    Args:
        cluster_counts: Candidate cluster counts.
        seed: Random seed for reproducibility.

    Returns:
        Candidate KMeans estimators keyed by model label.
    """
    candidates = {}
    for cluster_count in cluster_counts:
        label = f"kmeans_{cluster_count}"
        candidates[label] = KMeans(
            n_clusters=cluster_count,
            random_state=seed,
            n_init="auto",
        )

    return candidates


def build_gaussian_mixture_candidates(
    component_counts: list[int],
    seed: int,
) -> dict[str, GaussianMixture]:
    """Build seeded Gaussian mixture candidates per component count.

    Args:
        component_counts: Candidate Gaussian mixture component counts.
        seed: Random seed for reproducibility.

    Returns:
        Candidate GaussianMixture estimators keyed by model label.
    """
    candidates = {}
    for component_count in component_counts:
        label = f"gaussian_mixture_{component_count}"
        candidates[label] = GaussianMixture(
            n_components=component_count,
            # Diagonal covariance is more stable in one-hot-expanded spaces.
            covariance_type="diag",
            random_state=seed,
        )

    return candidates


def build_agglomerative_candidates(
    cluster_counts: list[int],
) -> dict[str, AgglomerativeClustering]:
    """Build Ward-linkage agglomerative clustering candidates.

    Args:
        cluster_counts: Candidate cluster counts.

    Returns:
        Candidate AgglomerativeClustering estimators keyed by model label.
    """
    candidates = {}
    for cluster_count in cluster_counts:
        label = f"agglomerative_{cluster_count}"
        candidates[label] = AgglomerativeClustering(
            n_clusters=cluster_count,
            linkage="ward",
        )

    return candidates


def build_hdbscan_candidates(
    min_cluster_size_options: list[int],
) -> dict[str, HDBSCAN]:
    """Build HDBSCAN candidates with configured minimum cluster sizes.

    HDBSCAN does not require a fixed cluster count and may label observations
    as noise with ``-1``.  The comparison layer excludes noise observations
    before calculating silhouette and Davies-Bouldin scores and records the
    retained sample count.

    Args:
        min_cluster_size_options: Positive minimum cluster sizes.

    Returns:
        Unfitted HDBSCAN estimators keyed by model label.

    Raises:
        ValueError: If an option is not positive.
    """
    if any(option <= 0 for option in min_cluster_size_options):
        raise ValueError("HDBSCAN min_cluster_size options must be positive")

    candidates = {}
    for min_cluster_size in min_cluster_size_options:
        label = f"hdbscan_min_cluster_size_{min_cluster_size}"
        candidates[label] = HDBSCAN(
            min_cluster_size=min_cluster_size,
        )

    return candidates
