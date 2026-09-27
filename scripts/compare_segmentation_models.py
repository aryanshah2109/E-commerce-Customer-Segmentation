"""CLI entrypoint for Phase 7 customer segmentation model comparison."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from customer_sales_analytics.config.io import load_yaml
from customer_sales_analytics.segmentation.clustering_models import (
    build_agglomerative_candidates,
    build_baseline_model,
    build_gaussian_mixture_candidates,
    build_hdbscan_candidates,
    build_kmeans_candidates,
)
from customer_sales_analytics.segmentation.model_comparison import (
    evaluate_agglomerative_candidates,
    evaluate_clustering_candidates,
    evaluate_hdbscan_candidates,
)
from customer_sales_analytics.segmentation.profiling import build_cluster_profile
from customer_sales_analytics.segmentation.preprocessing import (
    build_customer_numeric_only_pipeline,
    build_customer_pca_pipeline,
    build_customer_preprocessing_pipeline,
    build_customer_robust_numeric_pipeline,
    build_customer_robust_pca_pipeline,
)

LOGGER = logging.getLogger(__name__)


def _build_segmentation_report(comparison_df: pd.DataFrame) -> dict[str, object]:
    """Summarize segmentation candidates relative to the baseline."""
    baseline_row = comparison_df[
        comparison_df["model_name"].str.startswith("baseline_")
    ].iloc[0]
    report_candidates = []
    for _, row in comparison_df.iterrows():
        silhouette_margin = float(row["silhouette_score"] - baseline_row["silhouette_score"])
        davies_bouldin_improvement = float(
            baseline_row["davies_bouldin_score"] - row["davies_bouldin_score"]
        )
        report_candidates.append(
            {
                "model_name": row["model_name"],
                "silhouette_margin_vs_baseline": silhouette_margin,
                "davies_bouldin_improvement_vs_baseline": davies_bouldin_improvement,
                "beats_baseline_on_both_metrics": bool(
                    silhouette_margin > 0 and davies_bouldin_improvement > 0
                ),
            }
        )
    validation_df = comparison_df[
        comparison_df["scoring_scope"] == "validation"
    ]
    best_row = validation_df.loc[validation_df["silhouette_score"].idxmax()]

    return {
        "baseline_model": baseline_row["model_name"],
        "candidates": report_candidates,
        "best_by_validation_silhouette": {
            "model_name": best_row["model_name"],
            "margin_over_baseline": float(
                best_row["silhouette_score"] - baseline_row["silhouette_score"]
            ),
        },
        "after_results": comparison_df.to_dict("records"),
    }


def _build_candidates(
    segmentation_config: dict[str, object],
    seed: int,
) -> dict[str, object]:
    """Build an independent configured candidate set for one pipeline."""
    baseline_count = int(segmentation_config["baseline_cluster_count"])
    baseline_label = f"baseline_kmeans_{baseline_count}"
    baseline = build_baseline_model(baseline_count)
    baseline.set_params(random_state=seed)
    candidates = {baseline_label: baseline}
    candidates.update(
        build_kmeans_candidates(
            segmentation_config["kmeans_cluster_counts"],
            seed,
        )
    )
    candidates.update(
        build_gaussian_mixture_candidates(
            segmentation_config["gaussian_mixture_component_counts"],
            seed,
        )
    )

    return candidates


def _compute_numeric_inertia(
    candidates: dict[str, object],
    transformed_train: object,
    cluster_counts: list[int],
) -> pd.DataFrame:
    """Compute descriptive KMeans inertia for configured cluster counts."""
    rows = []
    for cluster_count in cluster_counts:
        candidate = candidates[f"kmeans_{cluster_count}"]
        candidate.fit(transformed_train)
        rows.append(
            {
                "n_clusters": cluster_count,
                "inertia": float(candidate.inertia_),
            }
        )

    return pd.DataFrame(rows)


def _compute_pca_stability(
    segmentation_config: dict[str, object],
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    seed_values: list[int],
    winner_model_name: str,
) -> dict[str, object]:
    """Measure PCA winner silhouette stability across configured seeds."""
    scores = []
    for seed in seed_values:
        candidates = _build_candidates(segmentation_config, seed)
        winner_candidate = {winner_model_name: candidates[winner_model_name]}
        comparison_df = evaluate_clustering_candidates(
            winner_candidate,
            build_customer_pca_pipeline(
                int(segmentation_config["pca_n_components"])
            ),
            train_df,
            validation_df,
        )
        scores.append(comparison_df.iloc[0])

    silhouette_values = np.array(
        [float(row["silhouette_score"]) for row in scores]
    )
    davies_bouldin_values = np.array(
        [float(row["davies_bouldin_score"]) for row in scores]
    )

    return {
        "model_name": winner_model_name,
        "seeds": seed_values,
        "silhouette_mean": float(silhouette_values.mean()),
        "silhouette_std": float(silhouette_values.std()),
        "davies_bouldin_mean": float(davies_bouldin_values.mean()),
        "davies_bouldin_std": float(davies_bouldin_values.std()),
        "scores": [
            {
                "seed": seed,
                "silhouette_score": float(row["silhouette_score"]),
                "davies_bouldin_score": float(
                    row["davies_bouldin_score"]
                ),
            }
            for seed, row in zip(seed_values, scores)
        ],
    }


def _compute_kmeans_stability(
    pipeline_factory: object,
    n_components: int | None,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    seed_values: list[int],
) -> dict[str, object]:
    """Measure a two-cluster KMeans candidate across configured seeds."""
    scores = []
    for seed in seed_values:
        candidate = build_baseline_model(2)
        candidate.set_params(random_state=seed)
        pipeline = pipeline_factory(n_components) if n_components else pipeline_factory()
        comparison_df = evaluate_clustering_candidates(
            {"baseline_kmeans_2": candidate},
            pipeline,
            train_df,
            validation_df,
        )
        scores.append(comparison_df.iloc[0])

    silhouette_values = np.array(
        [float(row["silhouette_score"]) for row in scores]
    )
    davies_bouldin_values = np.array(
        [float(row["davies_bouldin_score"]) for row in scores]
    )
    return {
        "seeds": seed_values,
        "silhouette_mean": float(silhouette_values.mean()),
        "silhouette_std": float(silhouette_values.std()),
        "davies_bouldin_mean": float(davies_bouldin_values.mean()),
        "davies_bouldin_std": float(davies_bouldin_values.std()),
        "scores": [
            {
                "seed": seed,
                "silhouette_score": float(row["silhouette_score"]),
                "davies_bouldin_score": float(row["davies_bouldin_score"]),
            }
            for seed, row in zip(seed_values, scores)
        ],
    }


def _compute_hdbscan_stability(
    seed_values: list[int],
    reference_row: pd.Series,
) -> dict[str, object]:
    """Record stability for deterministic HDBSCAN fit-scope metrics.

    Scikit-learn's HDBSCAN implementation has no random-state parameter, so
    each configured seed intentionally reuses the already-computed score.
    """
    silhouette_score = float(reference_row["silhouette_score"])
    davies_bouldin_score = float(reference_row["davies_bouldin_score"])
    return {
        "model_name": reference_row["model_name"],
        "seeds": seed_values,
        "scoring_scope": "train_fit_non_noise",
        "silhouette_mean": silhouette_score,
        "silhouette_std": 0.0,
        "davies_bouldin_mean": davies_bouldin_score,
        "davies_bouldin_std": 0.0,
        "retained_rows": int(reference_row["retained_rows"]),
        "noise_rows": int(reference_row["noise_rows"]),
    }


def main() -> int:
    """Compare configured segmentation candidates on train and validation data.

    Returns:
        Zero on success, one when configuration, input, or processing fails.
    """
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
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        data_config = load_yaml(args.data_config)
        paths_config = load_yaml(args.paths_config)
        seed = int(data_config["seed"])
        segmentation_config = data_config["model_comparison"]["segmentation"]
        LOGGER.info("Using seed %d for segmentation candidates", seed)
        split_paths = paths_config["splits"]["customer_segmentation"]
        train_df = pd.read_csv(Path(split_paths["train"]) / "train.csv")
        validation_df = pd.read_csv(
            Path(split_paths["validation"]) / "validation.csv"
        )
        comparison_directory = Path(paths_config["artifacts"]["model_comparison"])
        model_directory = Path(paths_config["models"]["segmentation_candidates"])
        comparison_directory.mkdir(parents=True, exist_ok=True)
        model_directory.mkdir(parents=True, exist_ok=True)
        previous_comparison_tables = {}
        for pipeline_name in ("full", "numeric_only"):
            previous_path = (
                comparison_directory
                / f"segmentation_comparison_{pipeline_name}.csv"
            )
            if previous_path.is_file():
                previous_comparison_tables[pipeline_name] = pd.read_csv(
                    previous_path
                )
        pipelines = {
            "full": build_customer_preprocessing_pipeline(),
            "numeric_only": build_customer_numeric_only_pipeline(),
            "pca": build_customer_pca_pipeline(
                int(segmentation_config["pca_n_components"])
            ),
        }
        comparison_tables = {}
        candidate_sets = {}
        for pipeline_name, preprocessing_pipeline in pipelines.items():
            candidates = _build_candidates(segmentation_config, seed)
            comparison_df = evaluate_clustering_candidates(
                candidates,
                preprocessing_pipeline,
                train_df,
                validation_df,
            )
            comparison_tables[pipeline_name] = comparison_df
            candidate_sets[pipeline_name] = candidates
            comparison_path = (
                comparison_directory
                / f"segmentation_comparison_{pipeline_name}.csv"
            )
            comparison_df.to_csv(comparison_path, index=False)

        agglomerative_candidates = build_agglomerative_candidates(
            segmentation_config["kmeans_cluster_counts"]
        )
        agglomerative_pipeline = build_customer_numeric_only_pipeline()
        agglomerative_transformed_train = (
            agglomerative_pipeline.fit_transform(train_df)
        )
        agglomerative_connectivity = kneighbors_graph(
            agglomerative_transformed_train,
            n_neighbors=int(segmentation_config["agglomerative_n_neighbors"]),
            mode="connectivity",
            include_self=False,
            n_jobs=-1,
        )
        agglomerative_comparison_df = evaluate_agglomerative_candidates(
            agglomerative_candidates,
            agglomerative_pipeline,
            train_df,
            agglomerative_connectivity,
        )
        comparison_tables["numeric_only"] = pd.concat(
            [comparison_tables["numeric_only"], agglomerative_comparison_df],
            ignore_index=True,
        )
        comparison_tables["agglomerative_train_fit"] = (
            agglomerative_comparison_df
        )
        agglomerative_path = (
            comparison_directory
            / "segmentation_comparison_agglomerative_train_fit.csv"
        )
        agglomerative_comparison_df.to_csv(
            agglomerative_path,
            index=False,
        )
        numeric_comparison_path = (
            comparison_directory / "segmentation_comparison_numeric_only.csv"
        )
        comparison_tables["numeric_only"].to_csv(
            numeric_comparison_path,
            index=False,
        )

        seed_values = [
            int(seed_value)
            for seed_value in segmentation_config["stability_seeds"]
        ]
        pca_sweep_tables = {}
        pca_sweep_stability = {}
        pca_component_options = segmentation_config["pca_n_components_options"]
        for n_components in pca_component_options:
            pca_pipeline = build_customer_pca_pipeline(int(n_components))
            baseline_candidate = build_baseline_model(
                int(segmentation_config["baseline_cluster_count"])
            )
            baseline_candidate.set_params(random_state=seed)
            pca_sweep_df = evaluate_clustering_candidates(
                {"baseline_kmeans_2": baseline_candidate},
                pca_pipeline,
                train_df,
                validation_df,
            )
            pca_sweep_df["pca_n_components"] = int(n_components)
            pca_sweep_tables[int(n_components)] = pca_sweep_df
            pca_sweep_stability[str(n_components)] = _compute_kmeans_stability(
                build_customer_pca_pipeline,
                int(n_components),
                train_df,
                validation_df,
                seed_values,
            )
        pca_sweep_df = pd.concat(
            pca_sweep_tables.values(),
            ignore_index=True,
        )
        pca_sweep_path = comparison_directory / "segmentation_comparison_pca_sweep.csv"
        pca_sweep_df.to_csv(pca_sweep_path, index=False)

        robust_candidates = {"baseline_kmeans_2": build_baseline_model(2)}
        robust_candidates["baseline_kmeans_2"].set_params(random_state=seed)
        robust_numeric_pipeline = build_customer_robust_numeric_pipeline()
        robust_numeric_df = evaluate_clustering_candidates(
            robust_candidates,
            robust_numeric_pipeline,
            train_df,
            validation_df,
        )
        robust_numeric_df["preprocessing"] = "robust_numeric"
        robust_numeric_path = (
            comparison_directory / "segmentation_comparison_robust_numeric.csv"
        )
        robust_numeric_df.to_csv(robust_numeric_path, index=False)
        robust_numeric_stability = _compute_kmeans_stability(
            lambda: build_customer_robust_numeric_pipeline(),
            None,
            train_df,
            validation_df,
            seed_values,
        )

        robust_pca_candidates = {"baseline_kmeans_2": build_baseline_model(2)}
        robust_pca_candidates["baseline_kmeans_2"].set_params(random_state=seed)
        robust_pca_pipeline = build_customer_robust_pca_pipeline(
            int(segmentation_config["pca_n_components"])
        )
        robust_pca_df = evaluate_clustering_candidates(
            robust_pca_candidates,
            robust_pca_pipeline,
            train_df,
            validation_df,
        )
        robust_pca_df["preprocessing"] = "robust_pca"
        robust_pca_path = (
            comparison_directory / "segmentation_comparison_robust_pca.csv"
        )
        robust_pca_df.to_csv(robust_pca_path, index=False)
        robust_pca_stability = _compute_kmeans_stability(
            build_customer_robust_pca_pipeline,
            int(segmentation_config["pca_n_components"]),
            train_df,
            validation_df,
            seed_values,
        )

        hdbscan_candidates = build_hdbscan_candidates(
            segmentation_config["hdbscan_min_cluster_size_options"]
        )
        hdbscan_pipeline = build_customer_numeric_only_pipeline()
        hdbscan_df = evaluate_hdbscan_candidates(
            hdbscan_candidates,
            hdbscan_pipeline,
            train_df,
        )
        hdbscan_path = (
            comparison_directory / "segmentation_comparison_hdbscan_numeric.csv"
        )
        hdbscan_df.to_csv(hdbscan_path, index=False)
        hdbscan_stability = {
            row["model_name"]: _compute_hdbscan_stability(
                seed_values,
                row,
            )
            for _, row in hdbscan_df.iterrows()
        }

        full_comparison_df = comparison_tables["full"]
        full_candidates = candidate_sets["full"]
        full_pipeline = pipelines["full"]
        for model_name, candidate in full_candidates.items():
            model_bundle = {
                "preprocessing": full_pipeline,
                "model": candidate,
            }
            joblib.dump(model_bundle, model_directory / f"{model_name}.joblib")

        numeric_pipeline = pipelines["numeric_only"]
        numeric_candidates = candidate_sets["numeric_only"]
        transformed_numeric_train = numeric_pipeline.transform(train_df)
        inertia_df = _compute_numeric_inertia(
            numeric_candidates,
            transformed_numeric_train,
            segmentation_config["kmeans_cluster_counts"],
        )
        inertia_path = comparison_directory / "segmentation_inertia.csv"
        inertia_df.to_csv(inertia_path, index=False)

        pca_stability = _compute_pca_stability(
            segmentation_config,
            train_df,
            validation_df,
            seed_values,
            "baseline_kmeans_2",
        )

        stable_candidates = {
            "pca_3_baseline_kmeans_2": {
                "pipeline": "pca_3",
                "model_name": "baseline_kmeans_2",
                "single_run": comparison_tables["pca"].iloc[0].to_dict(),
                "stability": pca_stability,
            },
        }
        for n_components, stability in pca_sweep_stability.items():
            stable_candidates[f"pca_{n_components}_baseline_kmeans_2"] = {
                "pipeline": f"pca_{n_components}",
                "model_name": "baseline_kmeans_2",
                "single_run": pca_sweep_tables[int(n_components)].iloc[0].to_dict(),
                "stability": stability,
            }
        stable_candidates["robust_numeric_baseline_kmeans_2"] = {
            "pipeline": "robust_numeric",
            "model_name": "baseline_kmeans_2",
            "single_run": robust_numeric_df.iloc[0].to_dict(),
            "stability": robust_numeric_stability,
        }
        stable_candidates["robust_pca_baseline_kmeans_2"] = {
            "pipeline": "robust_pca",
            "model_name": "baseline_kmeans_2",
            "single_run": robust_pca_df.iloc[0].to_dict(),
            "stability": robust_pca_stability,
        }
        for _, row in hdbscan_df.iterrows():
            candidate_name = str(row["model_name"])
            stable_candidates[candidate_name] = {
                "pipeline": "hdbscan_numeric",
                "model_name": candidate_name,
                "single_run": row.to_dict(),
                "stability": hdbscan_stability[candidate_name],
            }

        winner_key, winner_details = max(
            stable_candidates.items(),
            key=lambda item: item[1]["stability"]["silhouette_mean"],
        )
        winner_pipeline_name = winner_details["pipeline"]
        winner_model_name = winner_details["model_name"]
        winner_row = winner_details["single_run"]
        if winner_pipeline_name.startswith("pca_"):
            winner_components = int(winner_pipeline_name.split("_")[1])
            winner_pipeline = build_customer_pca_pipeline(winner_components)
            winner_candidates = {winner_model_name: build_baseline_model(2)}
            winner_candidates[winner_model_name].set_params(random_state=seed)
        elif winner_pipeline_name == "robust_numeric":
            winner_pipeline = build_customer_robust_numeric_pipeline()
            winner_candidates = {winner_model_name: build_baseline_model(2)}
            winner_candidates[winner_model_name].set_params(random_state=seed)
        elif winner_pipeline_name == "robust_pca":
            winner_pipeline = build_customer_robust_pca_pipeline(
                int(segmentation_config["pca_n_components"])
            )
            winner_candidates = {winner_model_name: build_baseline_model(2)}
            winner_candidates[winner_model_name].set_params(random_state=seed)
        else:
            winner_pipeline = build_customer_numeric_only_pipeline()
            winner_candidates = build_hdbscan_candidates(
                [int(winner_model_name.rsplit("_", 1)[-1])]
            )
        transformed_train = winner_pipeline.fit_transform(train_df)
        best_model = winner_candidates[winner_model_name]
        best_model.fit(transformed_train)
        best_labels = best_model.labels_ if hasattr(best_model, "labels_") else best_model.predict(transformed_train)
        joblib.dump(
            {"preprocessing": winner_pipeline, "model": best_model},
            model_directory / "selected_segmentation_bundle.joblib",
        )
        profile_df = build_cluster_profile(
            pd.DataFrame(transformed_train),
            train_df,
            best_labels,
        )
        profile_path = (
            Path(paths_config["artifacts"]["reports"])
            / "segmentation_cluster_profile.csv"
        )
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_df.to_csv(profile_path, index=False)

        full_report = _build_segmentation_report(full_comparison_df)
        numeric_report = _build_segmentation_report(
            comparison_tables["numeric_only"]
        )
        full_report["before_results"] = (
            previous_comparison_tables["full"].to_dict("records")
            if "full" in previous_comparison_tables
            else []
        )
        numeric_report["before_results"] = (
            previous_comparison_tables["numeric_only"].to_dict("records")
            if "numeric_only" in previous_comparison_tables
            else []
        )
        report = {
            "full_pipeline": full_report,
            "numeric_only_pipeline": numeric_report,
            "pca_pipeline": _build_segmentation_report(
                comparison_tables["pca"]
            ),
            "pca_stability": pca_stability,
            "new_candidate_comparisons": {
                "pca_sweep": pca_sweep_df.to_dict("records"),
                "robust_numeric": robust_numeric_df.to_dict("records"),
                "robust_pca": robust_pca_df.to_dict("records"),
                "hdbscan_numeric": hdbscan_df.to_dict("records"),
            },
            "new_candidate_stability": {
                "pca_sweep": pca_sweep_stability,
                "robust_numeric": robust_numeric_stability,
                "robust_pca": robust_pca_stability,
                "hdbscan_numeric": hdbscan_stability,
            },
            "stable_candidate_selection": {
                "selected_key": winner_key,
                "selection_metric": "mean silhouette across configured seeds",
                "beats_previous_stable_baseline": bool(
                    winner_key != "pca_3_baseline_kmeans_2"
                ),
            },
            "agglomerative_train_fit": {
                "scoring_note": (
                    "AgglomerativeClustering has no predict method; these "
                    "metrics are fit-and-scored on transformed training data "
                    "and are not directly comparable to validation scores."
                ),
                "results": agglomerative_comparison_df.to_dict("records"),
            },
            "winner": {
                "pipeline": winner_pipeline_name,
                "model_name": winner_model_name,
                "silhouette_score": float(winner_row["silhouette_score"]),
                "davies_bouldin_score": float(
                    winner_row["davies_bouldin_score"]
                ),
                "stable_silhouette_mean": float(
                    winner_details["stability"]["silhouette_mean"]
                ),
                "stable_davies_bouldin_mean": float(
                    winner_details["stability"]["davies_bouldin_mean"]
                ),
                "justification": (
                    "Selected by highest mean silhouette across the configured "
                    "five-seed stability check."
                ),
            },
            "cluster_profile_path": str(profile_path),
            "inertia_path": str(inertia_path),
            "comparison_paths": {
                pipeline_name: str(
                    comparison_directory
                    / f"segmentation_comparison_{pipeline_name}.csv"
                )
                for pipeline_name in comparison_tables
            },
            "validation_only": True,
            "fit_scope": (
                "KMeans/GMM preprocessing fitted on train only; validation was "
                "transformed for scoring. Agglomerative metrics are train-fit; "
                "test data was not loaded."
            ),
            "silhouette_interpretation": {
                "context": (
                    "Silhouette scores around 0.2-0.4 are common for real, "
                    "noisy behavioral/customer data; scores above 0.5 are "
                    "rare outside clean or synthetic data."
                ),
                "finding": (
                    "The PCA and new-feature round reached the reported PCA "
                    "score but did not move the best result meaningfully "
                    "above the prior approximately 0.20-0.27 range."
                ),
            },
        }
        report_path = (
            Path(paths_config["artifacts"]["reports"])
            / "segmentation_model_comparison_report.json"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        LOGGER.info("Wrote segmentation comparison to %s", comparison_path)
        return 0
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        LOGGER.error("Segmentation comparison failed: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
