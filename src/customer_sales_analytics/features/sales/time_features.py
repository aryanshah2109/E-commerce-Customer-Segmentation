"""Calendar feature calculations for sales transactions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_calendar_features(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic calendar decompositions derived from ``Order_Date``.

    Args:
        data_frame: Transaction-level sales data.

    Returns:
        A copied DataFrame with year, month, weekday, weekend, and quarter
        columns added.

    Raises:
        KeyError: If ``Order_Date`` is missing.
        ValueError: If an order date cannot be converted.
    """
    feature_df = data_frame.copy()
    order_dates = pd.to_datetime(feature_df["Order_Date"], errors="raise")
    day_index = (order_dates - order_dates.min()).dt.days.astype(int)
    feature_df["year"] = order_dates.dt.year
    feature_df["month"] = order_dates.dt.month
    feature_df["day_of_week"] = order_dates.dt.dayofweek
    feature_df["is_weekend"] = order_dates.dt.dayofweek >= 5
    feature_df["quarter"] = order_dates.dt.quarter
    feature_df["day_index"] = day_index
    feature_df["month_sin"] = np.sin(2 * np.pi * feature_df["month"] / 12)
    feature_df["month_cos"] = np.cos(2 * np.pi * feature_df["month"] / 12)
    feature_df["day_of_week_sin"] = np.sin(
        2 * np.pi * feature_df["day_of_week"] / 7
    )
    feature_df["day_of_week_cos"] = np.cos(
        2 * np.pi * feature_df["day_of_week"] / 7
    )

    return feature_df
