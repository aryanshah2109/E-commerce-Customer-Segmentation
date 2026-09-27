"""Descriptive sales analytics page."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from streamlit_app.components.cards import render_kpis
from streamlit_app.components.charts import line_chart
from streamlit_app.components.common import page_header, section_title, unavailable
from streamlit_app.config import DashboardPaths
from streamlit_app.services.data_loader import load_optional_csv
from streamlit_app.utils.helpers import money, number, percent


def render(paths: DashboardPaths) -> None:
    """Render interactive descriptive sales analytics."""
    page_header(
        "DESCRIPTIVE SALES",
        "Sales analytics",
        "Observed revenue, order volume, product mix, and calendar performance. No forecasting model is used here.",
    )
    sales = load_optional_csv(paths.sales_features, ("Order_Date",))
    raw = load_optional_csv(paths.raw_data, ("Order_Date",))
    if sales is None:
        unavailable("Daily sales features are unavailable. Run the batch pipeline first.")
        return

    min_date = sales["Order_Date"].min().date()
    max_date = sales["Order_Date"].max().date()
    selected_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    if isinstance(selected_range, tuple) and len(selected_range) == 2:
        start_date, end_date = selected_range
    else:
        start_date, end_date = min_date, max_date
    filtered_sales = sales[sales["Order_Date"].dt.date.between(start_date, end_date)].copy()

    total_revenue = filtered_sales["total_sales"].sum()
    total_orders = filtered_sales["order_count"].sum()
    average_order = total_revenue / total_orders if total_orders else 0
    monthly = filtered_sales.assign(month=filtered_sales["Order_Date"].dt.to_period("M").astype(str)).groupby("month", as_index=False)["total_sales"].sum()
    growth = monthly["total_sales"].pct_change().iloc[-1] if len(monthly) > 1 else None
    render_kpis(
        [
            ("Revenue", money(total_revenue), None),
            ("Orders", number(total_orders), None),
            ("Average order value", money(average_order), None),
            ("Latest month-over-month", percent(growth) if growth is not None else "—", None),
        ]
    )

    st.plotly_chart(line_chart(filtered_sales, "Order_Date", "total_sales", "Observed daily net sales", "Net sales (USD)"), width="stretch")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.bar(monthly, x="month", y="total_sales", title="Monthly performance", labels={"total_sales": "Net sales (USD)", "month": "Month"}), width="stretch")
    with right:
        weekday = filtered_sales.assign(weekday=filtered_sales["Order_Date"].dt.day_name()).groupby("weekday", as_index=False)["total_sales"].mean()
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekday["weekday"] = pd.Categorical(weekday["weekday"], categories=order, ordered=True)
        weekday = weekday.sort_values("weekday")
        st.plotly_chart(px.bar(weekday, x="weekday", y="total_sales", title="Average daily sales by weekday", labels={"total_sales": "Average net sales (USD)"}), width="stretch")

    section_title("Product and category performance", "Aggregated from the transaction table for the selected date range.")
    if raw is None:
        unavailable("Raw transaction data is unavailable for product/category detail.")
        return
    filtered_raw = raw[raw["Order_Date"].dt.date.between(start_date, end_date)].copy()
    if "Product_Category" in filtered_raw:
        categories = ["All"] + sorted(filtered_raw["Product_Category"].dropna().astype(str).unique().tolist())
        category = st.selectbox("Category filter", categories)
        if category != "All":
            filtered_raw = filtered_raw[filtered_raw["Product_Category"].astype(str) == category]
    product = filtered_raw.groupby(["Product_ID", "Product_Name"], dropna=False).agg(volume=("Quantity", "sum"), revenue=("Net_Sales_USD", "sum")).reset_index().sort_values("revenue", ascending=False).head(10)
    st.dataframe(product, width="stretch", hide_index=True)
    st.plotly_chart(px.bar(product.sort_values("revenue"), x="revenue", y="Product_Name", orientation="h", title="Top products by observed net sales"), width="stretch")

    section_title("Transaction gaps")
    gap_days = filtered_sales[filtered_sales["is_gap_day"]]
    render_kpis([("Zero-transaction days in range", number(len(gap_days)), None)])
    if not gap_days.empty:
        st.dataframe(gap_days[["Order_Date", "is_gap_day", "gap_filled_sales"]], width="stretch", hide_index=True)
