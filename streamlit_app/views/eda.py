"""Interactive exploratory data analysis page."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from streamlit_app.components.charts import heatmap, line_chart
from streamlit_app.components.common import page_header, section_title, unavailable
from streamlit_app.config import DashboardPaths
from streamlit_app.services.artifact_loader import optional_json
from streamlit_app.services.data_loader import load_optional_csv
from streamlit_app.utils.helpers import number


def render(paths: DashboardPaths) -> None:
    """Render EDA reports and interactive views."""
    page_header(
        "DATA EXPLORATION",
        "Explore the transaction dataset",
        "Quality, distributions, relationships, and operational patterns computed by the existing EDA pipeline.",
    )
    report = optional_json(paths.reports / "eda_summary.json")
    raw = load_optional_csv(paths.raw_data, ("Order_Date",))
    if report is None:
        unavailable("EDA report not found. Run the full batch pipeline first.")
        return

    overview = report.get("overview", {})
    render_kpis = st.columns(4)
    values = [
        ("Rows", number(overview.get("shape", [0, 0])[0])),
        ("Columns", number(overview.get("shape", [0, 0])[1])),
        ("Duplicate rows", number(overview.get("duplicate_rows"))),
        ("Date range", f"{str(overview.get('date_min', '—'))[:10]} → {str(overview.get('date_max', '—'))[:10]}"),
    ]
    for column, (label, value) in zip(render_kpis, values):
        column.metric(label, value)

    tabs = st.tabs(["Data quality", "Distributions", "Relationships", "Operations"])
    with tabs[0]:
        section_title("Missing-value semantics", "High null rates are reviewed in context; optional fields can be not applicable rather than defective.")
        missing = pd.DataFrame(report.get("missing_values", []))
        if not missing.empty:
            missing["null_rate_pct"] = missing["null_rate"] * 100
            st.dataframe(missing, width="stretch", hide_index=True)
            st.info("Return_Reason is naturally unavailable for non-returned orders, and Coupon_Code is naturally unavailable when no coupon was used. These fields are not described as generic data-quality failures.")
        outliers = pd.Series(report.get("outlier_counts", {}), name="outlier_count").sort_values(ascending=False).reset_index()
        outliers.columns = ["column", "outlier_count"]
        if not outliers.empty:
            st.plotly_chart(px.bar(outliers.head(12), x="outlier_count", y="column", orientation="h", title="IQR outlier counts"), width="stretch")
            st.caption("Outliers are documented and retained where high-value transactions are plausible business observations.")
    with tabs[1]:
        if raw is None:
            unavailable("Raw transaction data is not available for interactive distributions.")
        else:
            numeric = raw.select_dtypes(include="number").columns.tolist()
            selected_numeric = st.selectbox("Numeric field", numeric, index=numeric.index("Net_Sales_USD") if "Net_Sales_USD" in numeric else 0)
            sample = raw[[selected_numeric]].dropna().sample(min(50000, raw[selected_numeric].notna().sum()), random_state=42)
            st.plotly_chart(px.histogram(sample, x=selected_numeric, nbins=50, title=f"Distribution of {selected_numeric}"), width="stretch")
            categorical = raw.select_dtypes(include=["object", "bool"]).columns.tolist()
            selected_category = st.selectbox("Categorical field", categorical, index=categorical.index("Product_Category") if "Product_Category" in categorical else 0)
            counts = raw[selected_category].astype(str).value_counts().head(15).reset_index()
            counts.columns = [selected_category, "count"]
            st.plotly_chart(px.bar(counts, x="count", y=selected_category, orientation="h", title=f"Most common {selected_category} values"), width="stretch")
    with tabs[2]:
        correlation = optional_json(paths.reports / "customer_feature_correlation.json")
        if correlation and correlation.get("correlation_matrix"):
            matrix = pd.DataFrame(correlation["correlation_matrix"])
            st.plotly_chart(heatmap(matrix, "Customer feature correlations"), width="stretch")
        elif raw is not None:
            matrix = raw.select_dtypes(include="number").corr()
            st.plotly_chart(heatmap(matrix, "Transaction numeric correlations"), width="stretch")
        else:
            unavailable("Correlation artifacts are not available.")
        flagged = pd.DataFrame(report.get("correlations", []))
        if not flagged.empty:
            section_title("Flagged relationships", "Strong correlations are expected among arithmetically related sales measures.")
            st.dataframe(flagged, width="stretch", hide_index=True)
    with tabs[3]:
        sales = load_optional_csv(paths.sales_features, ("Order_Date",))
        zero = report.get("zero_transaction_analysis", {})
        section_title("Zero-transaction days")
        st.metric("Days with no observed transactions", number(zero.get("count")))
        st.caption("The feature pipeline keeps these calendar dates and marks them with `is_gap_day` rather than dropping them.")
        if sales is not None:
            st.plotly_chart(line_chart(sales, "Order_Date", "total_sales", "Daily net sales", "Net sales (USD)"), width="stretch")
        st.write("Zero-transaction dates", zero.get("zero_transaction_dates", []))
        section_title("Notable findings")
        for finding in report.get("notable_findings", []):
            st.markdown(f"- {finding}")
