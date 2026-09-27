"""Common page framing and graceful-state helpers."""

from __future__ import annotations

import streamlit as st


def page_header(eyebrow: str, title: str, description: str) -> None:
    """Render the shared page header."""
    st.markdown(f"<div class='eyebrow'>{eyebrow}</div>", unsafe_allow_html=True)
    st.title(title)
    st.markdown(f"<p class='page-description'>{description}</p>", unsafe_allow_html=True)


def section_title(title: str, description: str | None = None) -> None:
    """Render a compact section heading."""
    st.subheader(title)
    if description:
        st.caption(description)


def unavailable(message: str) -> None:
    """Render a non-fatal missing-artifact message."""
    st.info(message, icon="ℹ️")
