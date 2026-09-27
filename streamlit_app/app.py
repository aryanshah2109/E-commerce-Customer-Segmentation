"""Streamlit entry point for the customer and sales analytics workspace."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.config import get_paths
from streamlit_app.services.prediction import model_available
from streamlit_app.views import (
    about,
    eda,
    model,
    overview,
    pipeline,
    prediction,
    sales,
    segmentation,
)


def _load_css() -> None:
    css_path = Path(__file__).parent / "styles" / "style.css"
    st.markdown(
        f"<style>{css_path.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )


def main() -> None:
    """Configure the app and render the selected explicit-navigation view."""
    st.set_page_config(
        page_title="Customer & Sales Analytics",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _load_css()
    try:
        paths = get_paths()
    except (OSError, KeyError, TypeError, ValueError) as error:
        st.error(f"Dashboard configuration is unavailable: {error}")
        return

    selected_page = render_sidebar(model_available())
    views = {
        "Overview": overview.render,
        "EDA": eda.render,
        "Sales Analytics": sales.render,
        "Customer Segmentation": segmentation.render,
        "Model Performance": model.render,
        "Predict Customer": prediction.render,
        "Pipeline": pipeline.render,
        "About": about.render,
    }
    try:
        views[selected_page](paths)
    except (
        AttributeError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        st.error("This dashboard section could not be rendered.")
        with st.expander("Technical details"):
            st.write(str(error))


if __name__ == "__main__":
    main()
