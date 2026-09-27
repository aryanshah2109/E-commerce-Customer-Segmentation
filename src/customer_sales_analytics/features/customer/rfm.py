"""RFM-style customer feature calculations."""

from __future__ import annotations

import pandas as pd


def compute_rfm_features(
    data_frame: pd.DataFrame,
    reference_date: pd.Timestamp | None,
) -> pd.DataFrame:
    """Compute one RFM-style feature row for each customer.

    Args:
        data_frame: Preprocessed transaction-level sales data.
        reference_date: Date used for recency, or the dataset maximum date.

    Returns:
        Customer-level RFM features with one row per ``Customer_ID``.

    Raises:
        KeyError: If a required source column is missing.
        ValueError: If an order date cannot be converted.
    """
    working_df = data_frame.copy()
    working_df["Order_Date"] = pd.to_datetime(
        working_df["Order_Date"],
        errors="raise",
    )

    effective_reference_date = reference_date
    if effective_reference_date is None:
        effective_reference_date = working_df["Order_Date"].max()

    customer_df = working_df.groupby("Customer_ID", dropna=False).agg(
        first_order_date=("Order_Date", "min"),
        last_order_date=("Order_Date", "max"),
        frequency=("Order_ID", "nunique"),
        monetary_value=("Net_Sales_USD", "sum"),
        product_diversity=("Product_ID", "nunique"),
    )
    order_values_df = (
        working_df.groupby(["Customer_ID", "Order_ID"], dropna=False)[
            "Net_Sales_USD"
        ]
        .sum()
        .groupby("Customer_ID")
        .std()
        .rename("order_value_std")
    )
    customer_df = customer_df.join(order_values_df)
    customer_df["order_value_std"] = customer_df["order_value_std"].fillna(0.0)
    customer_df["recency_days"] = (
        pd.Timestamp(effective_reference_date) - customer_df["last_order_date"]
    ).dt.days
    customer_df["avg_order_value"] = (
        customer_df["monetary_value"] / customer_df["frequency"]
    )
    tenure_days = (
        pd.Timestamp(effective_reference_date) - customer_df["first_order_date"]
    ).dt.days.clip(lower=1)
    customer_df["purchase_velocity"] = (
        customer_df["frequency"] / tenure_days
    )

    feature_columns = [
        "recency_days",
        "frequency",
        "monetary_value",
        "avg_order_value",
        "order_value_std",
        "product_diversity",
        "purchase_velocity",
    ]
    return customer_df[feature_columns].reset_index()
