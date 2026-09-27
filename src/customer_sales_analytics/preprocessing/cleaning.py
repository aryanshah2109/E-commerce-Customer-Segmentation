"""Deterministic cleaning operations for the Phase 4 preprocessing step."""

from __future__ import annotations

import logging

import pandas as pd

LOGGER = logging.getLogger(__name__)


def remove_duplicate_rows(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows and log the number removed.

    Args:
        data_frame: Validated input dataset.

    Returns:
        A copied DataFrame without exact duplicate rows.
    """
    duplicate_count = int(data_frame.duplicated().sum())
    cleaned_df = data_frame.drop_duplicates().copy()

    LOGGER.info("Removed %d exact duplicate row(s)", duplicate_count)

    return cleaned_df


def fix_dtypes(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure ``Order_Date`` is datetime while keeping time separate.

    Args:
        data_frame: Dataset whose date dtype should be checked.

    Returns:
        A copied DataFrame with ``Order_Date`` converted to datetime.

    Raises:
        KeyError: If ``Order_Date`` is not present.
        ValueError: If a date value cannot be converted.
    """
    if "Order_Date" not in data_frame.columns:
        raise KeyError("Order_Date is required for preprocessing")

    cleaned_df = data_frame.copy()
    try:
        cleaned_df["Order_Date"] = pd.to_datetime(
            cleaned_df["Order_Date"],
            errors="raise",
        )
    except (TypeError, ValueError) as error:
        LOGGER.exception("Could not convert Order_Date to datetime")
        raise ValueError("Order_Date contains an invalid date value") from error

    return cleaned_df


def flag_future_order_dates(data_frame: pd.DataFrame) -> dict[str, object]:
    """Count future order dates without changing or dropping rows.

    Args:
        data_frame: Dataset with an ``Order_Date`` column.

    Returns:
        A report containing the future-date count and current date.

    Raises:
        KeyError: If ``Order_Date`` is not present.
    """
    if "Order_Date" not in data_frame.columns:
        raise KeyError("Order_Date is required for future-date checks")

    order_dates = pd.to_datetime(data_frame["Order_Date"], errors="raise")
    current_date = pd.Timestamp.now().normalize()

    # The dataset spans all of 2026, so future dates are expected and documented.
    future_mask = order_dates > current_date
    future_count = int(future_mask.sum())
    future_row_indices = []
    for row_index in data_frame.index[future_mask]:
        future_row_indices.append(str(row_index))

    LOGGER.info(
        "Found %d future Order_Date value(s); these are expected for this dataset",
        future_count,
    )

    return {
        "future_date_count": future_count,
        "future_row_indices": future_row_indices,
        "current_date": current_date.date().isoformat(),
        "decision": "Future dates are documented and retained because the dataset spans all of 2026.",
    }
