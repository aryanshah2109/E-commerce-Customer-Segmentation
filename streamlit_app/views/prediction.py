"""Single-customer prediction page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from api.schema.prediction import CustomerFeatures
from streamlit_app.components.common import page_header, unavailable
from streamlit_app.config import DashboardPaths
from streamlit_app.services.prediction import (
    get_registry,
    model_available,
    predict_customer,
)


def render(paths: DashboardPaths) -> None:
    """Render a pure-inference customer prediction form."""
    page_header(
        "MODEL INFERENCE",
        "Predict customer cluster",
        "Submit already-engineered customer features to the promoted PCA + KMeans model. Prediction never retrains the model.",
    )
    if not model_available():
        unavailable("Model not available — run the pipeline first.")
        return
    with st.form("prediction_form"):
        left, right = st.columns(2)
        with left:
            customer_id = st.text_input("Customer ID (optional)") or None
            recency_days = st.number_input("Recency days", min_value=0.0, value=60.82)
            frequency = st.number_input("Frequency", min_value=0.0, value=3.86)
            monetary_value = st.number_input("Monetary value", min_value=0.0, value=2537.29)
            avg_order_value = st.number_input("Average order value", min_value=0.0, value=746.61)
            order_value_std = st.number_input("Order value standard deviation", min_value=0.0, value=817.27)
            product_diversity = st.number_input("Product diversity", min_value=0.0, value=3.86)
        with right:
            purchase_velocity = st.number_input("Purchase velocity", min_value=0.0, value=0.0169, format="%.6f")
            return_rate = st.number_input("Return rate", min_value=0.0, max_value=1.0, value=0.0414, format="%.4f")
            coupon_usage_rate = st.number_input("Coupon usage rate", min_value=0.0, max_value=1.0, value=0.1489, format="%.4f")
            preferred_category = st.text_input("Preferred category", value="Electronics")
            preferred_payment_method = st.text_input("Preferred payment method", value="Credit Card")
        submitted = st.form_submit_button("Predict cluster", type="primary", width="stretch")

    if not submitted:
        return
    try:
        customer = CustomerFeatures(
            customer_id=customer_id,
            recency_days=recency_days,
            frequency=frequency,
            monetary_value=monetary_value,
            avg_order_value=avg_order_value,
            order_value_std=order_value_std,
            product_diversity=product_diversity,
            purchase_velocity=purchase_velocity,
            return_rate=return_rate,
            coupon_usage_rate=coupon_usage_rate,
            preferred_category=preferred_category,
            preferred_payment_method=preferred_payment_method,
        )
        result = predict_customer(customer)
    except (
        AttributeError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        st.error(f"Prediction unavailable: {error}")
        return

    st.success(f"Assigned to cluster {result.cluster}")
    columns = st.columns(4)
    columns[0].metric("Predicted cluster", str(result.cluster))
    columns[1].metric("Cluster size", f"{result.cluster_size:,}")
    _, _, _, trained_at = get_registry().get_bundle()
    columns[2].metric("Model", "PCA-2 + KMeans-2")
    columns[3].metric("Model timestamp", str(trained_at or "—"))
    st.subheader("Cluster profile")
    st.dataframe(
        pd.DataFrame([result.cluster_profile]),
        width="stretch",
        hide_index=True,
    )
