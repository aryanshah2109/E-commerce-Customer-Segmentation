"""About and scope page."""

from __future__ import annotations

import streamlit as st

from streamlit_app.components.common import page_header, section_title
from streamlit_app.config import DashboardPaths


def render(paths: DashboardPaths) -> None:
    """Render project scope and implementation notes."""
    page_header(
        "PROJECT NOTES",
        "About this workspace",
        "A presentation layer over the existing customer and sales analytics pipeline.",
    )
    section_title("What this dashboard does")
    st.markdown("""
    - Presents descriptive sales analytics generated from the e-commerce transaction data.
    - Explores customer-level behavioral features and the promoted PCA-2 + KMeans-2 segmentation.
    - Uses the existing fitted model for inference without fitting or retraining.
    - Provides a manual control to run the existing full batch pipeline.
    """)
    section_title("What it does not do")
    st.markdown("""
    - It does not forecast sales.
    - It does not create new clusters or rename clusters into business tiers.
    - It does not modify the scientific methodology in `src/customer_sales_analytics/`.
    """)
