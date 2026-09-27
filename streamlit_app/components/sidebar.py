"""Sidebar navigation and project status."""

from __future__ import annotations

import streamlit as st

PAGES = [
    "Overview",
    "EDA",
    "Sales Analytics",
    "Customer Segmentation",
    "Model Performance",
    "Predict Customer",
    "Pipeline",
    "About",
]


def render_sidebar(model_loaded: bool) -> str:
    """Render navigation and return the selected page."""
    with st.sidebar:
        st.markdown(
            '<div class="brand"><div class="brand-mark">CS</div>'
            '<div><strong>Customer & Sales</strong>'
            '<span>Analytics workspace</span></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown("<div class='sidebar-label'>Workspace</div>", unsafe_allow_html=True)
        selected = st.radio("Navigate", PAGES, label_visibility="collapsed")
        st.divider()
        status = "Model ready" if model_loaded else "Model unavailable"
        tone = "positive" if model_loaded else "warning"
        st.markdown(
            f'<div class="sidebar-status {tone}"><span class="dot"></span>{status}</div>',
            unsafe_allow_html=True,
        )
        st.caption("Sales analytics is descriptive. Customer segmentation is the only ML component.")
    return selected
