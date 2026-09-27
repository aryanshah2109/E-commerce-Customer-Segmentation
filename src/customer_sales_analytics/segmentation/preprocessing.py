"""Preprocessing pipeline construction for customer segmentation."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

CUSTOMER_NUMERIC_COLUMNS = [
    "recency_days",
    "frequency",
    "monetary_value",
    "avg_order_value",
    "order_value_std",
    "product_diversity",
    "purchase_velocity",
    "return_rate",
    "coupon_usage_rate",
]
CUSTOMER_CATEGORICAL_COLUMNS = [
    "preferred_category",
    "preferred_payment_method",
]


def write_customer_feature_correlation_report(
    customer_features: pd.DataFrame,
    output_path: Path,
    correlation_threshold: float = 0.80,
) -> dict[str, object]:
    """Compute and write Pearson correlations for customer numeric features.

    Args:
        customer_features: Customer-level feature table.
        output_path: JSON report destination.
        correlation_threshold: Absolute correlation threshold to flag.

    Returns:
        A serializable correlation matrix and flagged feature pairs.

    Raises:
        KeyError: If a configured numeric feature is missing.
        ValueError: If the threshold is outside the unit interval.
    """
    if not 0.0 <= correlation_threshold <= 1.0:
        raise ValueError("correlation_threshold must be between zero and one")

    missing_columns = set(CUSTOMER_NUMERIC_COLUMNS).difference(
        customer_features.columns
    )
    if missing_columns:
        raise KeyError(f"Missing customer numeric columns: {sorted(missing_columns)}")

    numeric_features = customer_features[CUSTOMER_NUMERIC_COLUMNS].apply(
        pd.to_numeric,
        errors="raise",
    )
    correlation_matrix = numeric_features.corr(method="pearson")
    flagged_pairs = []
    for left_column, right_column in combinations(CUSTOMER_NUMERIC_COLUMNS, 2):
        correlation = float(correlation_matrix.loc[left_column, right_column])
        if abs(correlation) > correlation_threshold:
            flagged_pairs.append(
                {
                    "feature_1": left_column,
                    "feature_2": right_column,
                    "correlation": correlation,
                    "absolute_correlation": abs(correlation),
                }
            )

    report = {
        "columns": CUSTOMER_NUMERIC_COLUMNS,
        "correlation_threshold": correlation_threshold,
        "correlation_matrix": correlation_matrix.round(6).to_dict(),
        "flagged_pairs": flagged_pairs,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _build_numeric_transformers(
    use_robust_scaler: bool = False,
) -> list[tuple[str, object, list[str]]]:
    """Build the standard and skew-aware numeric transformations."""
    standard_numeric_transformer = (
        RobustScaler() if use_robust_scaler else StandardScaler()
    )
    skewed_numeric_transformer = Pipeline(
        steps=[
            ("log1p", FunctionTransformer(np.log1p, validate=False)),
            ("scaler", StandardScaler()),
        ]
    )

    return [
        (
            "standard_numeric",
            standard_numeric_transformer,
            [
                "recency_days",
                "frequency",
                "order_value_std",
                "product_diversity",
                "purchase_velocity",
                "return_rate",
                "coupon_usage_rate",
            ],
        ),
        (
            "skewed_numeric",
            skewed_numeric_transformer,
            ["monetary_value", "avg_order_value"],
        ),
    ]


def build_customer_preprocessing_pipeline() -> ColumnTransformer:
    """Build train-fitted scaling and categorical encoding transformations.

    Returns:
        A column transformer that standardizes numeric features and one-hot
        encodes categorical features while dropping ``Customer_ID``.
    """
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )
    preprocessing_pipeline = ColumnTransformer(
        transformers=[
            *_build_numeric_transformers(),
            ("categorical", categorical_transformer, CUSTOMER_CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )

    return preprocessing_pipeline


def build_customer_numeric_only_pipeline() -> ColumnTransformer:
    """Build a train-fitted numeric-only customer transformation.

    Returns:
        A column transformer containing scaled numeric features without
        categorical one-hot columns.
    """
    return ColumnTransformer(
        transformers=_build_numeric_transformers(),
        remainder="drop",
    )


def build_customer_pca_pipeline(n_components: int) -> Pipeline:
    """Build numeric preprocessing followed by PCA dimensionality reduction.

    Args:
        n_components: Number of principal components to retain.

    Returns:
        A train-fitted pipeline containing numeric preprocessing and PCA.

    Raises:
        ValueError: If ``n_components`` is not positive.
    """
    if n_components <= 0:
        raise ValueError("n_components must be positive")

    return Pipeline(
        steps=[
            ("numeric", build_customer_numeric_only_pipeline()),
            ("pca", PCA(n_components=n_components)),
        ]
    )


def build_customer_robust_numeric_pipeline() -> ColumnTransformer:
    """Build a numeric-only pipeline using RobustScaler for outlier resistance.

    Returns:
        A train-fitted numeric transformer with robust scaling for the
        non-log-transformed customer features.
    """
    return ColumnTransformer(
        transformers=_build_numeric_transformers(use_robust_scaler=True),
        remainder="drop",
    )


def build_customer_robust_pca_pipeline(n_components: int) -> Pipeline:
    """Build robust numeric preprocessing followed by PCA.

    Args:
        n_components: Number of principal components to retain.

    Returns:
        A train-fitted robust numeric pipeline followed by PCA.

    Raises:
        ValueError: If ``n_components`` is not positive.
    """
    if n_components <= 0:
        raise ValueError("n_components must be positive")

    return Pipeline(
        steps=[
            ("numeric", build_customer_robust_numeric_pipeline()),
            ("pca", PCA(n_components=n_components)),
        ]
    )
