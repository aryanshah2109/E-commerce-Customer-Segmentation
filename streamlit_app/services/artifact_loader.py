"""Cached loading and normalization of generated report artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def load_json(path_string: str) -> dict[str, Any]:
    """Load one generated JSON report."""
    return json.loads(Path(path_string).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_profile(path_string: str) -> pd.DataFrame:
    """Load the promoted cluster profile."""
    return pd.read_csv(path_string)


def optional_json(path: Path) -> dict[str, Any] | None:
    """Return a report mapping when a configured artifact exists."""
    if not path.is_file():
        return None
    return load_json(str(path))


def optional_profile(path: Path) -> pd.DataFrame | None:
    """Return a cluster profile when it exists."""
    if not path.is_file():
        return None
    return load_profile(str(path))
