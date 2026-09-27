"""Customer segmentation and customer explorer page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from streamlit_app.components.charts import cluster_bar, scatter_chart
from streamlit_app.components.common import page_header, section_title, unavailable
from streamlit_app.config import DashboardPaths
from streamlit_app.services.artifact_loader import optional_profile
from streamlit_app.services.data_loader import load_optional_csv
from streamlit_app.services.prediction import (
    get_registry,
    predict_feature_table,
)
from streamlit_app.utils.helpers import cluster_color


def render(paths: DashboardPaths) -> None:
    """Render cluster distribution, profiles, and customer explorer."""
    page_header(
        "CUSTOMER INTELLIGENCE",
        "Customer segmentation",
        "Explore the promoted PCA-2 + KMeans-2 representation and the behavioral differences observed between clusters.",
    )
    features = load_optional_csv(paths.customer_features)
    profile = optional_profile(paths.cluster_profile)
    if features is None or profile is None:
        unavailable("Customer features or the promoted cluster profile is unavailable. Run the pipeline first.")
        return
    if not get_registry().health()[0]:
        unavailable("Model not available — run the pipeline first.")
        return

    with st.spinner("Preparing customer cluster view…"):
        assignments, transformed = predict_feature_table(features)
    cluster_sizes = assignments["cluster"].value_counts().sort_index().rename_axis("cluster").reset_index(name="cluster_size")
    cluster_sizes["percentage"] = cluster_sizes["cluster_size"] / len(assignments)
    render_kpis = st.columns(len(cluster_sizes))
    for column, row in zip(render_kpis, cluster_sizes.itertuples()):
        column.metric(f"Cluster {row.cluster}", f"{row.cluster_size:,.0f}", f"{row.percentage:.1%} of customers")

    left, right = st.columns([1, 1.25])
    with left:
        st.plotly_chart(cluster_bar(cluster_sizes, "Customer distribution"), width="stretch")
    with right:
        section_title("Profile comparison", "Means from the saved cluster profile artifact.")
        profile_display = profile.copy()
        profile_display["cluster"] = profile_display["cluster"].astype(int)
        st.dataframe(profile_display, width="stretch", hide_index=True)

    section_title("Behavioral map", "The plotted coordinates are the fitted preprocessing pipeline's transformed representation; no second clustering model is created.")
    coordinates = pd.DataFrame(transformed[:, :2], columns=["component_1", "component_2"])
    coordinates = pd.concat([assignments.reset_index(drop=True), coordinates], axis=1)
    st.plotly_chart(scatter_chart(coordinates.sample(min(12000, len(coordinates)), random_state=42), "component_1", "component_2", "cluster", "Customer representation"), width="stretch")

    section_title("Customer explorer")
    customer_ids = assignments["Customer_ID"].astype(str).tolist()
    selected_id = st.selectbox("Search Customer_ID", customer_ids)
    selected = features[features["Customer_ID"].astype(str) == selected_id].iloc[0]
    selected_cluster = int(assignments.loc[assignments["Customer_ID"] == selected_id, "cluster"].iloc[0])
    left, right = st.columns(2)
    with left:
        st.markdown(f"<div class='cluster-chip' style='--chip-color:{cluster_color(selected_cluster)}'>Cluster {selected_cluster}</div>", unsafe_allow_html=True)
        st.metric("Customer ID", selected_id)
        st.metric("Assigned cluster", str(selected_cluster))
        st.dataframe(selected.astype(str).to_frame("value"), width="stretch")
    with right:
        cluster_profile = profile[profile["cluster"].round().astype(int) == selected_cluster]
        if not cluster_profile.empty:
            st.markdown("**Assigned cluster profile**")
            profile_table = cluster_profile.iloc[0].astype(str).rename("value").reset_index()
            st.dataframe(profile_table, width="stretch", hide_index=True)
