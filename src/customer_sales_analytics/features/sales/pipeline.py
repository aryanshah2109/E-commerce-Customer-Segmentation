"""Sales feature table assembly."""

from __future__ import annotations

import logging

import pandas as pd

from customer_sales_analytics.features.sales.aggregation import (
    compute_daily_sales_features,
)
from customer_sales_analytics.features.sales.time_features import (
    add_calendar_features,
)

LOGGER = logging.getLogger(__name__)


def build_sales_feature_table(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Build the combined daily sales feature table.

    Args:
        data_frame: Preprocessed transaction-level sales data.
    Returns:
        One row per observed calendar day with calendar and sales features.
    """
    calendar_df = add_calendar_features(data_frame)
    daily_df = compute_daily_sales_features(calendar_df)
    calendar_by_day = add_calendar_features(daily_df[["Order_Date"]])
    calendar_by_day = calendar_by_day[
        [
            "Order_Date",
            "year",
            "month",
            "day_of_week",
            "is_weekend",
            "quarter",
            "day_index",
            "month_sin",
            "month_cos",
            "day_of_week_sin",
            "day_of_week_cos",
        ]
    ]
    feature_df = calendar_by_day.merge(
        daily_df,
        on="Order_Date",
        how="inner",
        validate="one_to_one",
    )
    feature_df = feature_df.sort_values("Order_Date").reset_index(drop=True)

    if feature_df.empty:
        LOGGER.info("Built empty sales feature table")
    else:
        LOGGER.info(
            "Built sales feature table: %d rows from %s to %s",
            feature_df.shape[0],
            feature_df["Order_Date"].min().date(),
            feature_df["Order_Date"].max().date(),
        )

    return feature_df
