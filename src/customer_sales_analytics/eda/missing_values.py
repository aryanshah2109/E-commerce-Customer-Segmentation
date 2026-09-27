"""Checklist step 3: audit missing values and flag high-null columns."""

from __future__ import annotations

import pandas as pd


def compute_missing_values(
    data_frame: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Calculate missing counts, rates, and threshold flags.

    The high-null fields in this retail dataset are likely systematic optional
    fields, such as return reasons for non-returned orders.  This is an
    observation for EDA, not an MCAR/MNAR test.

    Args:
        data_frame: Input sales data.
        threshold: Null-rate threshold in the inclusive range [0, 1].

    Returns:
        DataFrame with ``null_count``, ``null_rate``, and ``is_flagged``.

    Raises:
        ValueError: If threshold is outside [0, 1].
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    null_counts = data_frame.isna().sum()
    null_rates = data_frame.isna().mean()

    result = pd.DataFrame(
        {
            "null_count": null_counts,
            "null_rate": null_rates,
        }
    )
    result["is_flagged"] = result["null_rate"] > threshold

    result = result.sort_values("null_rate", ascending=False)

    return result
