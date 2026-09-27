"""KPI and status card components."""

from __future__ import annotations

import streamlit as st


def render_kpis(items: list[tuple[str, str, str | None]]) -> None:
    """Render a row of label/value/delta KPI cards."""
    columns = st.columns(len(items))
    for column, (label, value, delta) in zip(columns, items):
        with column:
            st.metric(label, value, delta=delta)


def render_status(
    label: str,
    value: str,
    tone: str = "neutral",
) -> None:
    """Render a compact status block using the global CSS classes."""
    st.markdown(
        f'<div class="status-card {tone}"><span>{label}</span>'
        f'<strong>{value}</strong></div>',
        unsafe_allow_html=True,
    )
