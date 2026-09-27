"""Formatting and chart helpers shared by dashboard views."""

from __future__ import annotations

from typing import Any

import pandas as pd

CLUSTER_COLORS = {
    0: "#2F6BFF",
    1: "#F28E5B",
}


def money(value: float | None) -> str:
    """Format a numeric amount as a compact USD value."""
    if value is None or pd.isna(value):
        return "—"
    return f"${float(value):,.0f}"


def number(value: float | None) -> str:
    """Format a number with thousands separators."""
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):,.0f}"


def percent(value: float | None, decimals: int = 1) -> str:
    """Format a ratio as a percentage."""
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value) * 100:.{decimals}f}%"


def metric_delta(value: float | None) -> str | None:
    """Return a signed percentage delta suitable for Streamlit metrics."""
    if value is None or pd.isna(value):
        return None
    return f"{float(value) * 100:+.1f}%"


def cluster_color(cluster: int) -> str:
    """Return a stable color for a cluster label."""
    return CLUSTER_COLORS.get(int(cluster), "#7B8794")


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert report values to floats without crashing a page."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
