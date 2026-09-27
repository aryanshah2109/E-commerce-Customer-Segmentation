"""Technical model evaluation page."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from streamlit_app.components.cards import render_kpis
from streamlit_app.components.common import page_header, section_title, unavailable
from streamlit_app.config import DashboardPaths
from streamlit_app.services.artifact_loader import optional_json
from streamlit_app.utils.helpers import number


def _candidate_rows(report: dict) -> pd.DataFrame:
    rows = []
    for row in report.get("pca_pipeline", {}).get("after_results", []):
        rows.append({
            "model": row.get("model_name"),
            "silhouette": row.get("silhouette_score"),
            "davies_bouldin": row.get("davies_bouldin_score"),
            "scope": row.get("scoring_scope"),
            "representation": "PCA",
        })
    for row in report.get("new_candidate_comparisons", {}).get("pca2_cluster_sweep", []):
        rows.append({
            "model": row.get("model_name"),
            "silhouette": row.get("silhouette_score"),
            "davies_bouldin": row.get("davies_bouldin_score"),
            "scope": row.get("scoring_scope"),
            "representation": "PCA-2 sweep",
        })
    return pd.DataFrame(rows).drop_duplicates(subset=["model", "representation"])


def render(paths: DashboardPaths) -> None:
    """Render model metrics, stability, and candidate comparison."""
    page_header(
        "MODEL GOVERNANCE",
        "Model performance",
        "Unsupervised evaluation of the promoted customer segmentation model, with stability and comparison evidence.",
    )
    training = optional_json(paths.reports / "segmentation_training_report.json")
    comparison = optional_json(paths.reports / "segmentation_model_comparison_report.json")
    if training is None:
        unavailable("Segmentation training artifacts are unavailable.")
        return
    metrics = training.get("metrics", {})
    render_kpis(
        [
            ("Model", training.get("model", "—"), None),
            ("Silhouette", f"{metrics.get('silhouette_score', 0):.4f}", "higher is better"),
            ("Davies–Bouldin", f"{metrics.get('davies_bouldin_score', 0):.4f}", "lower is better"),
            ("Training rows", number(training.get("training_rows")), None),
            ("Seed", str(training.get("random_seed", "—")), None),
        ]
    )

    st.markdown("<div class='callout'><strong>How to read these metrics</strong><span>Silhouette rewards compact, separated clusters. Davies–Bouldin rewards low within-cluster spread relative to between-cluster separation. There is no accuracy metric because this is unsupervised learning.</span></div>", unsafe_allow_html=True)
    if comparison:
        stability = comparison.get("pca_stability", {})
        section_title("Stability across seeds", "The selection rule considers mean silhouette minus standard deviation and rejects unstable candidates.")
        stability_frame = pd.DataFrame(stability.get("scores", []))
        if not stability_frame.empty:
            st.plotly_chart(px.line(stability_frame, x="seed", y="silhouette_score", markers=True, title="PCA-2 winner silhouette by seed"), width="stretch")
        render_kpis(
            [
                ("Seeds", ", ".join(map(str, stability.get("seeds", []))), None),
                ("Mean silhouette", f"{stability.get('silhouette_mean', 0):.4f}", None),
                ("Silhouette std", f"{stability.get('silhouette_std', 0):.6f}", None),
                (
                    "Selection score",
                    f"{stability.get('selection_score', stability.get('silhouette_mean', 0) - stability.get('silhouette_std', 0)):.4f}",
                    None,
                ),
            ]
        )

        candidate_frame = _candidate_rows(comparison)
        if not candidate_frame.empty:
            section_title("Candidate comparison")
            st.dataframe(candidate_frame.sort_values("silhouette", ascending=False), width="stretch", hide_index=True)
            st.plotly_chart(px.scatter(candidate_frame, x="silhouette", y="davies_bouldin", color="representation", hover_name="model", title="Candidate quality comparison"), width="stretch")

        with st.expander("Selection rationale"):
            st.write(comparison.get("stable_candidate_selection", {}))
            st.write(comparison.get("winner", {}))
            st.caption(comparison.get("fit_scope", ""))
    else:
        unavailable("Model comparison report is unavailable.")
