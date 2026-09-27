"""Pure inference service for the promoted segmentation model."""

from __future__ import annotations

import logging
from collections import Counter

import pandas as pd

from customer_sales_analytics.segmentation.preprocessing import (
    CUSTOMER_CATEGORICAL_COLUMNS,
    CUSTOMER_NUMERIC_COLUMNS,
)

from api.schema.prediction import ClusterAssignment, CustomerFeatures
from api.services.model_registry import ModelRegistry

LOGGER = logging.getLogger(__name__)
FEATURE_COLUMNS = CUSTOMER_NUMERIC_COLUMNS + CUSTOMER_CATEGORICAL_COLUMNS


def _profile_value(value: object) -> float | str:
    if pd.isna(value):
        return ""
    if isinstance(value, str):
        return value
    return float(value)


def predict_clusters(
    customers: list[CustomerFeatures],
    registry: ModelRegistry,
) -> list[ClusterAssignment]:
    """Transform engineered customers and assign them to existing clusters.

    Args:
        customers: Validated, already-engineered customer feature records.
        registry: Registry containing the fitted preprocessing and estimator.

    Returns:
        Cluster assignments with matching profile summaries.

    Raises:
        AttributeError: If the promoted estimator cannot predict new rows.
        ValueError: If a predicted cluster has no profile row.
    """
    preprocessing, model, profile_df, _ = registry.get_bundle()
    rows = [customer.model_dump(exclude={"customer_id"}) for customer in customers]
    features = pd.DataFrame(rows, columns=FEATURE_COLUMNS)

    transformed = preprocessing.transform(features)
    predict = getattr(model, "predict", None)
    if not callable(predict):
        raise AttributeError(
            "The promoted model type must support out-of-sample prediction."
        )
    labels = predict(transformed)

    if "cluster" not in profile_df.columns:
        raise ValueError("Cluster profile is missing the 'cluster' column.")

    assignments = []
    cluster_counts = Counter(int(label) for label in labels)
    for customer, label in zip(customers, labels):
        cluster = int(label)
        matching = profile_df[profile_df["cluster"].round().astype(int) == cluster]
        if matching.empty:
            raise ValueError(f"No cluster profile exists for cluster {cluster}.")
        profile_row = matching.iloc[0]
        cluster_size = int(profile_row["cluster_size"])
        profile = {
            column: _profile_value(value)
            for column, value in profile_row.to_dict().items()
        }
        assignments.append(
            ClusterAssignment(
                customer_id=customer.customer_id,
                cluster=cluster,
                cluster_size=cluster_size,
                cluster_profile=profile,
            )
        )

    LOGGER.info(
        "Predicted %d customers; cluster distribution=%s",
        len(customers),
        dict(cluster_counts),
    )
    return assignments
