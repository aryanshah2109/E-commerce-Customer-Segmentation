"""Behavioral customer feature calculations."""

from __future__ import annotations

import pandas as pd


def _alphabetical_mode(series: pd.Series) -> object:
    """Return the most frequent value, breaking ties alphabetically."""
    counts = series.value_counts(dropna=False)
    highest_count = counts.max()
    candidates = [value for value in counts.index if counts[value] == highest_count]
    candidates.sort(key=lambda value: str(value))
    return candidates[0]


def compute_behavioral_features(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Compute one behavioral feature row for each customer.

    The preferred category and payment method use the most common line-item
    value. Ties are resolved by selecting the alphabetically first value.
    ``Return_Status`` is intentionally used as a descriptive post-outcome
    behavior signal for segmentation, which has no future-outcome target.

    Args:
        data_frame: Preprocessed transaction-level sales data.

    Returns:
        Customer-level behavioral features with one row per ``Customer_ID``.

    Raises:
        KeyError: If a required source column is missing.
    """
    working_df = data_frame.copy()
    customer_groups = working_df.groupby("Customer_ID", dropna=False)
    behavioral_df = customer_groups.agg(
        preferred_category=("Product_Category", _alphabetical_mode),
        preferred_payment_method=("Payment_Method", _alphabetical_mode),
        return_rate=("Return_Status", lambda values: (values == "Returned").mean()),
        coupon_usage_rate=("Coupon_Used", "mean"),
    )

    return behavioral_df.reset_index()
