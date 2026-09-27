"""KPI and status card components."""

from __future__ import annotations

from html import escape

import streamlit as st


def render_kpis(items: list[tuple[str, str, str | None]]) -> None:
    """Render a row of label/value/delta KPI cards."""
    columns = st.columns(len(items))
    for column, (label, value, delta) in zip(columns, items):
        with column:
            tooltip = escape(f"{label}: {value}")
            delta_markup = (
                f'<div class="metric-card-delta" title="{escape(delta)}">'
                f"{escape(delta)}</div>"
                if delta
                else ""
            )
            st.markdown(
                f'<div class="metric-card" title="{tooltip}">'
                f'<div class="metric-card-label">{escape(label)}</div>'
                f'<div class="metric-card-value">{escape(value)}</div>'
                f"{delta_markup}</div>",
                unsafe_allow_html=True,
            )


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
