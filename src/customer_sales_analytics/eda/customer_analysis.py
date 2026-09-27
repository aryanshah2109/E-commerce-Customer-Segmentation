"""Checklist step 6: describe customer frequency, value, and recency."""

from __future__ import annotations

import pandas as pd


def compute_customer_analysis(data_frame: pd.DataFrame) -> dict[str, object]:
    """Compute simple customer-level descriptive signals.

    One input row represents one product line in an order.  These summaries are
    descriptive only; feature engineering and RFM are out of scope.

    Args:
        data_frame: Transaction-level sales data.

    Returns:
        Customer summary table and repeat-purchase rate.
    """
    working_df = data_frame.copy()
    working_df["Order_Date"] = pd.to_datetime(
        working_df["Order_Date"],
        errors="coerce",
    )

    customer_groups = working_df.groupby("Customer_ID", dropna=False)
    customer_df = customer_groups.agg(
        order_frequency=("Order_ID", "nunique"),
        monetary_value=("Net_Sales_USD", "sum"),
        last_order_date=("Order_Date", "max"),
        first_order_date=("Order_Date", "min"),
    )

    last_dataset_date = working_df["Order_Date"].max()
    customer_df["recency_days"] = (
        last_dataset_date - customer_df["last_order_date"]
    ).dt.days

    repeat_customers = customer_df["order_frequency"] > 1
    if len(customer_df) == 0:
        repeat_purchase_rate = 0.0
    else:
        repeat_purchase_rate = float(repeat_customers.mean())

    return {
        "customer_summary": customer_df,
        "repeat_purchase_rate": repeat_purchase_rate,
    }
