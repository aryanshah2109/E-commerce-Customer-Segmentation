"""Shared detection of dates with no observed sales transactions."""

from __future__ import annotations

import pandas as pd


def detect_zero_transaction_dates(
    data_frame: pd.DataFrame,
) -> pd.DatetimeIndex:
    """Return calendar dates with no transactions in the observed range.

    Args:
        data_frame: Transaction-level data containing ``Order_Date``.

    Returns:
        Sorted daily dates between the minimum and maximum order dates that
        have no transaction rows.

    Raises:
        KeyError: If ``Order_Date`` is missing.
        ValueError: If an order date cannot be converted.
    """
    order_dates = pd.to_datetime(
        data_frame["Order_Date"],
        errors="raise",
    ).dt.normalize()
    if order_dates.empty:
        return pd.DatetimeIndex([], dtype="datetime64[ns]")

    daily_order_counts = (
        pd.Series(1, index=order_dates)
        .resample("D")
        .sum()
        .reindex(
            pd.date_range(order_dates.min(), order_dates.max(), freq="D"),
            fill_value=0,
        )
    )

    return pd.DatetimeIndex(daily_order_counts.index[daily_order_counts.eq(0)])
