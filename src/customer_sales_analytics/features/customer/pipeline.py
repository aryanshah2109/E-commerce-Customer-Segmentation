"""Customer feature table assembly."""

from __future__ import annotations

import logging

import pandas as pd

from customer_sales_analytics.features.customer.behavioral import (
    compute_behavioral_features,
)
from customer_sales_analytics.features.customer.rfm import compute_rfm_features

LOGGER = logging.getLogger(__name__)


def build_customer_feature_table(
    data_frame: pd.DataFrame,
    reference_date: pd.Timestamp | None,
) -> pd.DataFrame:
    """Build the combined customer-level feature table.

    Args:
        data_frame: Preprocessed transaction-level sales data.
        reference_date: Date used for recency, or the dataset maximum date.

    Returns:
        One row per customer containing RFM and behavioral features.
    """
    rfm_df = compute_rfm_features(data_frame, reference_date)
    behavioral_df = compute_behavioral_features(data_frame)
    feature_df = rfm_df.merge(
        behavioral_df,
        on="Customer_ID",
        how="inner",
        validate="one_to_one",
    )

    LOGGER.info(
        "Built customer feature table: %d rows, %d columns",
        feature_df.shape[0],
        feature_df.shape[1],
    )

    return feature_df
