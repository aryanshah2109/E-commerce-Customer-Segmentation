"""Manual full-pipeline execution page."""

from __future__ import annotations

import streamlit as st

from streamlit_app.components.common import page_header, section_title
from streamlit_app.config import DashboardPaths
from streamlit_app.services.pipeline import run_full_pipeline
from streamlit_app.utils.helpers import number


def render(paths: DashboardPaths) -> None:
    """Render manual pipeline controls and latest run status."""
    page_header(
        "OPERATIONS",
        "Pipeline control",
        "Manually run the existing batch workflow and refresh its generated artifacts. The dashboard never retrains automatically.",
    )
    section_title("Full pipeline", "Validation → EDA → preprocessing → features → splits → segmentation → evaluation → promotion")
    st.markdown("<div class='pipeline-strip'><span>Data preparation</span><b>→</b><span>EDA</span><b>→</b><span>Preprocessing</span><b>→</b><span>Features</span><b>→</b><span>Splits</span><b>→</b><span>Promotion</span></div>", unsafe_allow_html=True)
    st.warning("This operation can take several minutes and rewrites generated artifacts. Start it only when a retraining run is intended.")
    if st.button("Run Full Pipeline", type="primary", width="stretch"):
        with st.spinner("Running the existing batch pipeline…"):
            result = run_full_pipeline()
        st.session_state["last_pipeline_result"] = result
        st.rerun()

    result = st.session_state.get("last_pipeline_result")
    if result is None:
        st.info("No pipeline run has been started from this dashboard session.")
        return
    if result.get("status") == "success":
        st.success("Pipeline completed successfully.")
        columns = st.columns(6)
        columns[0].metric("Duration", f"{result['duration_seconds']:.1f}s")
        columns[1].metric("Model", result.get("model_name", "—"))
        columns[2].metric("Silhouette", f"{result.get('silhouette_score', 0):.4f}")
        columns[3].metric("Davies–Bouldin", f"{result.get('davies_bouldin_score', 0):.4f}")
        columns[4].metric("Training rows", number(result.get("training_rows")))
        columns[5].metric("Promotion", "Completed")
        st.caption(f"Log: {result.get('log_path', '—')}")
        st.caption("Cached dashboard data refreshes on the next page interaction or browser reload.")
    else:
        st.error(f"Pipeline failed: {result.get('failed_stage', 'unknown stage')}")
        st.caption(f"Log: {result.get('log_path', '—')}")
