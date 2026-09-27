"""Cached loading of project datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def load_csv(path_string: str, parse_dates: tuple[str, ...] = ()) -> pd.DataFrame:
    """Load a CSV once per path and optionally parse date columns."""
    path = Path(path_string)
    data_frame = pd.read_csv(path)
    for column in parse_dates:
        if column in data_frame.columns:
            data_frame[column] = pd.to_datetime(
                data_frame[column],
                errors="coerce",
            )
    return data_frame


def load_optional_csv(
    path: Path,
    parse_dates: tuple[str, ...] = (),
) -> pd.DataFrame | None:
    """Load a configured CSV when present, otherwise return ``None``."""
    if not path.is_file():
        return None
    return load_csv(str(path), parse_dates)
