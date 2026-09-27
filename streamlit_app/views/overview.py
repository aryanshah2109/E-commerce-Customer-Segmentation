"""Executive overview page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from streamlit_app.components.cards import render_kpis
from streamlit_app.components.charts import cluster_bar, line_chart
from streamlit_app.components.common import page_header, section_title, unavailable
from streamlit_app.config import DashboardPaths
from streamlit_app.services.artifact_loader import optional_json, optional_profile
from streamlit_app.services.data_loader import load_optional_csv
from streamlit_app.utils.helpers import money, number


def render(paths: DashboardPaths) -> None:
    """Render the executive overview."""
    page_header(
        "EXECUTIVE VIEW",
        "Customer and Sales Analytics",
        "A decision-ready view of descriptive commerce performance and the production customer segmentation model.",
    )
    report = optional_json(paths.reports / "eda_summary.json")
    training = optional_json(paths.reports / "segmentation_training_report.json")
    profile = optional_profile(paths.cluster_profile)
    sales = load_optional_csv(paths.sales_features, ("Order_Date",))
    features = load_optional_csv(paths.customer_features)

    if report is None:
        unavailable("EDA artifacts are not available. Run the batch pipeline to populate the dashboard.")
        return

    kpis = report.get("sales_kpis", {})
    unique_counts = report.get("overview", {}).get("unique_counts", {})
    cluster_sizes = training.get("cluster_sizes", {}) if training else {}
    render_kpis(
        [
            ("Total revenue", money(kpis.get("total_revenue")), None),
            ("Total orders", number(kpis.get("total_orders")), None),
            ("Customers", number(unique_counts.get("Customer_ID")), None),
            ("Products", number(unique_counts.get("Product_ID")), None),
            ("Clusters", number(len(cluster_sizes) or (len(profile) if profile is not None else None)), None),
            ("Average order value", money(kpis.get("average_order_value")), None),
        ]
    )

    st.markdown("<div class='callout'><strong>Analytical scope</strong><span>Sales is descriptive analytics only. Customer segmentation is the project's only machine-learning component.</span></div>", unsafe_allow_html=True)
    left, right = st.columns([1.55, 1])
    with left:
        section_title("Revenue trajectory", "Observed daily revenue aggregated by month; this is not a forecast.")
        if sales is not None and not sales.empty:
            monthly = sales.assign(month=sales["Order_Date"].dt.to_period("M").astype(str)).groupby("month", as_index=False)["total_sales"].sum()
            st.plotly_chart(line_chart(monthly, "month", "total_sales", "Monthly net sales", "Net sales (USD)"), width="stretch")
        else:
            unavailable("Daily sales features are not available.")
    with right:
        section_title("Cluster distribution")
        if cluster_sizes:
            cluster_frame = pd.DataFrame({"cluster": [int(key) for key in cluster_sizes], "cluster_size": list(cluster_sizes.values())})
            st.plotly_chart(cluster_bar(cluster_frame, "Promoted segmentation clusters"), width="stretch")
        else:
            unavailable("The promoted cluster profile is not available.")

    left, right = st.columns([1.1, 1])
    with left:
        section_title("Cluster profile summary", "Observed profile means from the promoted artifact; cluster labels are not business tiers.")
        if profile is not None:
            display = profile.copy()
            display["cluster"] = display["cluster"].astype(int)
            st.dataframe(display, width="stretch", hide_index=True)
        else:
            unavailable("Run the pipeline to generate cluster profiles.")
    with right:
        section_title("Production model")
        if training:
            metrics = training.get("metrics", {})
            render_kpis(
                [
                    ("Model", training.get("model", "—"), None),
                    ("Silhouette", f"{metrics.get('silhouette_score', 0):.4f}", "higher is better"),
                ]
            )
            st.metric("Davies–Bouldin", f"{metrics.get('davies_bouldin_score', 0):.4f}", help="Lower is better.")
            st.caption(f"Trained on {number(training.get('training_rows'))} customer records with seed {training.get('random_seed', '—')}.")
        else:
            unavailable("Model metrics are not available yet.")

    section_title("Pipeline architecture")
    st.markdown("<div class='pipeline-strip'><span>Validate</span><b>→</b><span>EDA</span><b>→</b><span>Preprocess</span><b>→</b><span>Features</span><b>→</b><span>Split</span><b>→</b><span>Segment</span><b>→</b><span>Promote</span></div>", unsafe_allow_html=True)
    if features is not None:
        st.caption(f"Feature table ready: {number(len(features))} customers × {number(features.shape[1] - 1)} engineered features.")
