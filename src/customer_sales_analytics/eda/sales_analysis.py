"""Descriptive sales statistics for reporting and dashboard views."""

from __future__ import annotations

import pandas as pd

from customer_sales_analytics.features.sales.gap_days import (
    detect_zero_transaction_dates,
)


def analyze_zero_transaction_days(data_frame: pd.DataFrame) -> dict[str, object]:
    """Analyze dates with no transactions in the observed date range.

    Args:
        data_frame: Transaction-level sales data containing ``Order_Date``.

    Returns:
        A serializable report containing exact zero-transaction dates and
        calendar-pattern summaries.

    Raises:
        KeyError: If ``Order_Date`` is missing.
        ValueError: If an order date cannot be converted.
    """
    zero_dates = detect_zero_transaction_dates(data_frame)
    zero_date_series = pd.Series(zero_dates)
    days_in_month = zero_date_series.dt.days_in_month
    day_of_month = zero_date_series.dt.day
    gaps = zero_date_series.diff().dropna().dt.days.astype(int).tolist()

    return {
        "zero_transaction_dates": [
            date.strftime("%Y-%m-%d") for date in zero_dates
        ],
        "count": len(zero_dates),
        "day_of_week": sorted(zero_date_series.dt.day_name().unique().tolist()),
        "day_of_month": sorted(day_of_month.unique().tolist()),
        "days_from_month_end": sorted(
            (days_in_month - day_of_month).unique().tolist()
        ),
        "all_within_final_three_days_of_month": bool(
            ((days_in_month - day_of_month) <= 2).all()
        ),
        "spacing_days": gaps,
        "spacing_is_even": len(set(gaps)) <= 1,
    }


def compute_sales_kpis(data_frame: pd.DataFrame) -> dict[str, float]:
    """Compute the headline sales metrics for the dashboard.

    Args:
        data_frame: Transaction-level sales data.

    Returns:
        Total revenue, unique orders, average order value, and the growth rate
        from the penultimate to the latest calendar month.

    Raises:
        KeyError: If a required sales column is missing.
        ValueError: If an order date cannot be converted.
    """
    required_columns = {"Order_ID", "Order_Date", "Net_Sales_USD"}
    missing_columns = required_columns.difference(data_frame.columns)
    if missing_columns:
        raise KeyError(f"Missing KPI columns: {sorted(missing_columns)}")

    working_df = data_frame.copy()
    working_df["Order_Date"] = pd.to_datetime(
        working_df["Order_Date"],
        errors="raise",
    )
    total_revenue = float(working_df["Net_Sales_USD"].sum())
    total_orders = float(working_df["Order_ID"].nunique())
    average_order_value = total_revenue / total_orders if total_orders else 0.0

    monthly_sales = compute_monthly_trend(working_df)
    if len(monthly_sales) < 2:
        growth_rate = 0.0
    else:
        previous_sales = float(monthly_sales.iloc[-2]["total_sales"])
        latest_sales = float(monthly_sales.iloc[-1]["total_sales"])
        growth_rate = (
            (latest_sales - previous_sales) / previous_sales
            if previous_sales
            else 0.0
        )

    return {
        "total_revenue": total_revenue,
        "total_orders": total_orders,
        "average_order_value": float(average_order_value),
        "period_over_period_growth_rate": float(growth_rate),
    }


def compute_weekday_pattern(data_frame: pd.DataFrame) -> pd.Series:
    """Compute average sales per calendar day for each weekday.

    Args:
        data_frame: Transaction-level sales data.

    Returns:
        A Monday-to-Sunday series of mean daily net sales.
    """
    working_df = data_frame.copy()
    working_df["Order_Date"] = pd.to_datetime(
        working_df["Order_Date"],
        errors="raise",
    ).dt.normalize()
    daily_sales = working_df.groupby("Order_Date")["Net_Sales_USD"].sum()
    weekday_sales = daily_sales.groupby(daily_sales.index.dayofweek).mean()
    weekday_sales = weekday_sales.reindex(range(7), fill_value=0.0)
    weekday_sales.index = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    weekday_sales.name = "average_daily_sales"
    return weekday_sales.astype(float)


def compute_monthly_trend(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly revenue, order count, and month-over-month change.

    Args:
        data_frame: Transaction-level sales data.

    Returns:
        One row per calendar month with chronological descriptive metrics.
    """
    working_df = data_frame.copy()
    working_df["Order_Date"] = pd.to_datetime(
        working_df["Order_Date"],
        errors="raise",
    )
    working_df["month"] = working_df["Order_Date"].dt.to_period("M").dt.to_timestamp()
    monthly_df = working_df.groupby("month", as_index=False).agg(
        total_sales=("Net_Sales_USD", "sum"),
        order_count=("Order_ID", "nunique"),
    )
    monthly_df["month_over_month_change"] = monthly_df["total_sales"].pct_change()
    return monthly_df.sort_values("month").reset_index(drop=True)


def compute_sales_analysis(
    data_frame: pd.DataFrame,
    rolling_window: int,
) -> dict[str, pd.DataFrame]:
    """Compute daily, weekly, and monthly descriptive sales tables.

    The rolling mean and period-over-period change describe historical data;
    they are not a forecasting model.

    Args:
        data_frame: Transaction-level sales data.
        rolling_window: Number of periods used for the rolling mean.

    Returns:
        Daily, weekly, and monthly sales tables.

    Raises:
        ValueError: If rolling_window is not positive.
    """
    if rolling_window <= 0:
        raise ValueError("rolling_window must be positive")

    working_df = data_frame.copy()
    working_df["Order_Date"] = pd.to_datetime(
        working_df["Order_Date"],
        errors="raise",
    )

    sales_series = working_df.set_index("Order_Date")["Net_Sales_USD"]
    result: dict[str, pd.DataFrame] = {}
    periods = {
        "daily": "D",
        "weekly": "W",
        "monthly": "MS",
    }

    for period_name, resample_rule in periods.items():
        period_df = sales_series.resample(resample_rule).agg(
            ["sum", "count"]
        )
        period_df = period_df.rename(
            columns={
                "sum": "sales",
                "count": "orders",
            }
        )
        period_df["rolling_mean"] = period_df["sales"].rolling(
            rolling_window,
            min_periods=1,
        ).mean()
        period_df["period_over_period_change"] = period_df["sales"].pct_change()
        result[period_name] = period_df

    return result
