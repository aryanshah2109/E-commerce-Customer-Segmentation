"""Prediction request and response schemas."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """Already-engineered features for one customer."""

    customer_id: str | None = None
    recency_days: Annotated[float, Field(ge=0)]
    frequency: Annotated[float, Field(ge=0)]
    monetary_value: Annotated[float, Field(ge=0)]
    avg_order_value: Annotated[float, Field(ge=0)]
    order_value_std: Annotated[float, Field(ge=0)]
    product_diversity: Annotated[float, Field(ge=0)]
    purchase_velocity: Annotated[float, Field(ge=0)]
    return_rate: Annotated[float, Field(ge=0, le=1)]
    coupon_usage_rate: Annotated[float, Field(ge=0, le=1)]
    preferred_category: str
    preferred_payment_method: str


class PredictRequest(BaseModel):
    """Single-request batch of customer feature rows."""

    customers: Annotated[list[CustomerFeatures], Field(min_length=1)]


class ClusterAssignment(BaseModel):
    """One predicted cluster and its profile summary."""

    customer_id: str | None
    cluster: int
    cluster_size: int
    cluster_profile: dict[str, float | str]


class PredictResponse(BaseModel):
    """Prediction response for one or more customers."""

    predictions: list[ClusterAssignment]
    model_trained_at_utc: str | None
