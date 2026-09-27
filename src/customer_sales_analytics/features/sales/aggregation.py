"""Daily descriptive sales feature calculations."""

from __future__ import annotations

import pandas as pd

from customer_sales_analytics.features.sales.gap_days import (
    detect_zero_transaction_dates,
)


def interpolate_gap_days(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Add a gap-smoothed sales series for descriptive visualizations.

    Args:
        daily_df: Daily sales table containing ``total_sales`` and
            ``is_gap_day``.

    Returns:
        A copied table with ``gap_filled_sales`` added.

    Raises:
        KeyError: If required columns are missing.
    """
    required_columns = {"total_sales", "is_gap_day"}
    missing_columns = required_columns.difference(daily_df.columns)
    if missing_columns:
        raise KeyError(f"Missing gap interpolation columns: {sorted(missing_columns)}")

    result_df = daily_df.copy()
    gap_sales = result_df["total_sales"].astype(float).where(
        ~result_df["is_gap_day"].astype(bool)
    )
    result_df["gap_filled_sales"] = gap_sales.interpolate(method="linear")

    return result_df


def compute_daily_sales_features(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sales by day and add gap-day descriptive statistics.

    Args:
        data_frame: Transaction-level sales data.
    Returns:
        One row per observed calendar day with daily sales features.

    """
    working_df = data_frame.copy()
    working_df["Order_Date"] = pd.to_datetime(
        working_df["Order_Date"],
        errors="raise",
    ).dt.normalize()
    daily_df = working_df.groupby("Order_Date", as_index=False).agg(
        total_sales=("Net_Sales_USD", "sum"),
        order_count=("Order_ID", "nunique"),
        average_order_value=("Net_Sales_USD", "mean"),
    )
    first_date = daily_df["Order_Date"].min()
    last_date = daily_df["Order_Date"].max()
    full_date_range = pd.date_range(first_date, last_date, freq="D")
    daily_df = daily_df.set_index("Order_Date").reindex(full_date_range)
    daily_df.index.name = "Order_Date"
    daily_df["total_sales"] = daily_df["total_sales"].fillna(0)
    daily_df["order_count"] = daily_df["order_count"].fillna(0).astype(int)
    daily_df["average_order_value"] = daily_df["average_order_value"].fillna(0)
    daily_df = daily_df.reset_index()
    gap_dates = detect_zero_transaction_dates(working_df)
    daily_df["is_gap_day"] = daily_df["Order_Date"].isin(gap_dates)
    daily_df = interpolate_gap_days(daily_df)

    return daily_df
