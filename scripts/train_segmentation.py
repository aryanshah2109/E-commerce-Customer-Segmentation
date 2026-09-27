"""Train the selected production customer segmentation model."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from customer_sales_analytics.config.io import load_yaml
from customer_sales_analytics.config.logging import configure_logging
from customer_sales_analytics.segmentation.clustering_models import (
    build_baseline_model,
)
from customer_sales_analytics.segmentation.model_selection import (
    evaluate_clustering_model,
    is_candidate_better,
)
from customer_sales_analytics.segmentation.preprocessing import (
    build_customer_pca_pipeline,
)
from customer_sales_analytics.segmentation.profiling import build_cluster_profile

LOGGER = logging.getLogger(__name__)


def _load_stored_metrics(metrics_path: Path) -> dict[str, float] | None:
    """Load existing best-model metrics when they are available."""
    if not metrics_path.is_file():
        return None

    try:
        stored_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return {
            "silhouette_score": float(stored_metrics["silhouette_score"]),
            "davies_bouldin_score": float(
                stored_metrics["davies_bouldin_score"]
            ),
        }
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        LOGGER.warning(
            "Ignoring unreadable stored best metrics at %s: %s",
            metrics_path,
            error,
        )
        return None


def main() -> int:
    """Train and save PCA-2 plus KMeans-2 on all customer features."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data_config.yaml"),
    )
    parser.add_argument(
        "--paths-config",
        type=Path,
        default=Path("configs/paths.yaml"),
    )
    args = parser.parse_args()
    configure_logging("segmentation_training")
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

    try:
        data_config = load_yaml(args.data_config)
        paths_config = load_yaml(args.paths_config)
        seed = int(data_config["seed"])
        customer_features_path = Path(
            paths_config["data"]["customer_features"]
        )
        model_directory = Path(
            paths_config["models"]["segmentation_candidates"]
        )
        best_model_directory = Path(
            paths_config["models"].get(
                "best_model",
                "models/segmentation/best_model",
            )
        )
        report_directory = Path(paths_config["artifacts"]["reports"])
        customer_features = pd.read_csv(customer_features_path)

        LOGGER.info("Baseline model: two-cluster KMeans")
        LOGGER.info("Selected model: PCA-2 plus KMeans-2")
        LOGGER.info(
            "Training on %d customer records with seed %d",
            len(customer_features),
            seed,
        )

        preprocessing_pipeline = build_customer_pca_pipeline(2)
        model = build_baseline_model(2)
        model.set_params(random_state=seed)
        transformed_features = preprocessing_pipeline.fit_transform(
            customer_features
        )
        model.fit(transformed_features)
        labels = model.labels_
        metrics = evaluate_clustering_model(
            transformed_features,
            labels,
        )
        LOGGER.info(
            "Evaluation metrics on full data: silhouette=%.6f, "
            "davies_bouldin=%.6f",
            metrics["silhouette_score"],
            metrics["davies_bouldin_score"],
        )

        cluster_sizes = {
            str(cluster): int(size)
            for cluster, size in (
                pd.Series(labels).value_counts().sort_index().items()
            )
        }
        training_metadata = {
            "model": "PCA-2 + KMeans-2",
            "preprocessing": "numeric scaling + PCA-2",
            "training_rows": len(customer_features),
            "feature_count": customer_features.shape[1],
            "feature_columns": customer_features.columns.tolist(),
            "random_seed": seed,
            "hyperparameters": model.get_params(),
            "cluster_sizes": cluster_sizes,
        }
        model_bundle = {
            "preprocessing": preprocessing_pipeline,
            "model": model,
            "metadata": training_metadata,
        }

        model_directory.mkdir(parents=True, exist_ok=True)
        model_path = model_directory / "selected_segmentation_bundle.joblib"
        joblib.dump(model_bundle, model_path)

        profile = build_cluster_profile(
            pd.DataFrame(transformed_features),
            customer_features,
            labels,
        )
        profile_path = report_directory / "segmentation_cluster_profile.csv"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile.to_csv(profile_path, index=False)
        report_path = report_directory / "segmentation_training_report.json"
        report = {
            "model": "PCA-2 + KMeans-2",
            "training_rows": len(customer_features),
            "feature_count": customer_features.shape[1],
            "random_seed": seed,
            "metrics": metrics,
            "cluster_sizes": cluster_sizes,
            "model_path": str(model_path),
            "cluster_profile_path": str(profile_path),
        }
        report_path.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        best_metrics_path = best_model_directory / "best_metrics.json"
        stored_metrics = _load_stored_metrics(best_metrics_path)
        should_promote = is_candidate_better(metrics, stored_metrics)
        if should_promote:
            best_model_directory.mkdir(parents=True, exist_ok=True)
            best_model_path = best_model_directory / "best_model.joblib"
            best_profile_path = best_model_directory / "cluster_profile.csv"
            best_metrics = {
                **training_metadata,
                **metrics,
                "trained_at_utc": datetime.now(timezone.utc).isoformat(),
                "model_path": str(best_model_path),
                "cluster_profile_path": str(best_profile_path),
            }
            joblib.dump(model_bundle, best_model_path)
            profile.to_csv(best_profile_path, index=False)
            best_metrics_path.write_text(
                json.dumps(best_metrics, indent=2),
                encoding="utf-8",
            )
            LOGGER.info(
                "Promoted new best model: silhouette=%.6f, davies_bouldin=%.6f",
                metrics["silhouette_score"],
                metrics["davies_bouldin_score"],
            )
        else:
            LOGGER.info(
                "Kept stored best model: silhouette=%.6f, davies_bouldin=%.6f",
                stored_metrics["silhouette_score"],
                stored_metrics["davies_bouldin_score"],
            )

        LOGGER.info("Saved trained model to %s", model_path)
        LOGGER.info("Saved cluster profile to %s", profile_path)
        LOGGER.info("Segmentation training completed successfully")
        return 0
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        LOGGER.error("Segmentation training failed: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
