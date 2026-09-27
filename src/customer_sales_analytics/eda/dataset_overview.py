"""Checklist steps 1 and 2: inspect dataset shape and structure."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DatasetOverview:
    """Shape, schema, date range, identifiers, and duplicate-row count."""

    shape: tuple[int, int]
    dtypes: dict[str, str]
    memory_bytes: int
    date_min: str | None
    date_max: str | None
    unique_counts: dict[str, int]
    duplicate_rows: int
    possible_leakage_columns: list[str]
    sanity_checks: list[str]


def compute_dataset_overview(
    data_frame: pd.DataFrame,
) -> DatasetOverview:
    """Compute the basic shape and structure checklist items.

    One row represents one product line in an order.  Several rows can belong
    to the same order.

    Args:
        data_frame: Validated sales data.

    Returns:
        A dataset overview dataclass.
    """
    shape = tuple(data_frame.shape)

    dtypes = {}
    for column, dtype in data_frame.dtypes.items():
        dtypes[column] = str(dtype)

    memory_bytes = int(data_frame.memory_usage(deep=True).sum())

    date_min = None
    date_max = None
    if "Order_Date" in data_frame.columns:
        order_dates = pd.to_datetime(
            data_frame["Order_Date"],
            errors="coerce",
        )
        if order_dates.notna().any():
            date_min = order_dates.min().isoformat()
            date_max = order_dates.max().isoformat()

    unique_counts = {}
    identifier_columns = ["Customer_ID", "Order_ID", "Product_ID"]
    for column in identifier_columns:
        if column in data_frame.columns:
            unique_counts[column] = int(data_frame[column].nunique())

    duplicate_rows = int(data_frame.duplicated().sum())

    possible_leakage_columns = []
    post_outcome_columns = [
        "Return_Status",
        "Return_Reason",
        "Customer_Rating",
        "Review_Sentiment",
        "Order_Profit_USD",
    ]
    for column in post_outcome_columns:
        if column in data_frame.columns:
            possible_leakage_columns.append(column)

    sanity_checks = []
    amount_columns = [
        "Gross_Sales_USD",
        "Net_Sales_USD",
        "Total_Order_Value_USD",
    ]
    for column in amount_columns:
        if column in data_frame.columns:
            negative_count = int((data_frame[column] < 0).sum())
            if negative_count == 0:
                sanity_checks.append(f"{column}: no negative values found")
            else:
                sanity_checks.append(
                    f"{column}: {negative_count} negative value(s) found"
                )

    if "Order_Date" in data_frame.columns:
        current_date = pd.Timestamp.now().normalize()
        future_count = int(
            (pd.to_datetime(data_frame["Order_Date"]) > current_date).sum()
        )
        if future_count == 0:
            sanity_checks.append("Order_Date: no future dates found")
        else:
            sanity_checks.append(
                f"Order_Date: {future_count} future date(s) found"
            )

    return DatasetOverview(
        shape=shape,
        dtypes=dtypes,
        memory_bytes=memory_bytes,
        date_min=date_min,
        date_max=date_max,
        unique_counts=unique_counts,
        duplicate_rows=duplicate_rows,
        possible_leakage_columns=possible_leakage_columns,
        sanity_checks=sanity_checks,
    )
