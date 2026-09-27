"""Inference services built on the existing promoted model bundle."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd
import streamlit as st

from api.config import Settings
from api.schema.prediction import CustomerFeatures
from api.services.model_registry import ModelNotAvailableError, ModelRegistry
from api.services.prediction_service import predict_clusters


@lru_cache(maxsize=1)
def get_registry() -> ModelRegistry:
    """Return one model registry for the Streamlit process."""
    return ModelRegistry(Settings())


def predict_customer(customer: CustomerFeatures) -> Any:
    """Run pure inference through the existing API prediction service."""
    registry = get_registry()
    registry.reload_if_stale()
    return predict_clusters([customer], registry)[0]


@st.cache_data(show_spinner=False)
def predict_feature_table(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, Any]:
    """Predict all feature-table rows and return labels and 2D coordinates."""
    registry = get_registry()
    preprocessing, model, _, _ = registry.reload_if_stale()
    customer_ids = features["Customer_ID"].astype(str).reset_index(drop=True)
    model_features = features.drop(columns=["Customer_ID"])
    transformed = preprocessing.transform(model_features)
    labels = model.predict(transformed)
    return (
        pd.DataFrame(
            {
                "Customer_ID": customer_ids,
                "cluster": labels.astype(int),
            }
        ),
        transformed,
    )


def model_available() -> bool:
    """Return whether the promoted model can currently be loaded."""
    try:
        get_registry().reload_if_stale()
    except (ModelNotAvailableError, OSError, ValueError, KeyError):
        return False
    return True
