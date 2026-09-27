"""Raw CSV loading and schema-directed dtype coercion."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from customer_sales_analytics.data.schema import _BOOLEAN_COLUMNS, _FLOAT_COLUMNS, _INTEGER_COLUMNS

LOGGER = logging.getLogger(__name__)


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load and coerce a configured raw CSV without mutating the source.

    Args:
        path: Path to the raw CSV file.

    Returns:
        A newly loaded DataFrame with schema-compatible date and primitive dtypes.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a required conversion fails.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Raw data file does not exist: {path}")
    data_frame = pd.read_csv(path)
    try:
        data_frame["Order_Date"] = pd.to_datetime(data_frame["Order_Date"], errors="raise")
        for column in _INTEGER_COLUMNS:
            data_frame[column] = pd.to_numeric(data_frame[column], errors="raise").astype("int64")
        for column in _FLOAT_COLUMNS:
            data_frame[column] = pd.to_numeric(data_frame[column], errors="raise").astype("float64")
        for column in _BOOLEAN_COLUMNS:
            if not pd.api.types.is_bool_dtype(data_frame[column]):
                data_frame[column] = data_frame[column].astype("boolean")
            data_frame[column] = data_frame[column].astype(bool)
    except (KeyError, TypeError, ValueError) as error:
        LOGGER.exception("Failed dtype coercion for raw data at %s", path)
        raise ValueError(f"Raw data dtype coercion failed: {error}") from error
    LOGGER.info("Loaded raw data from %s: %d rows, %d columns", path, *data_frame.shape)
    return data_frame
