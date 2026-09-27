"""Document the Phase 4 decision to retain plausible business outliers."""

from __future__ import annotations

import json
from pathlib import Path

OUTLIER_DECISION = (
    "Monetary and business outliers are not capped or removed in this phase "
    "because high-value transactions are plausible real orders, not data "
    "errors. Removing them could bias sales-trend and customer-segmentation "
    "analysis."
)


def summarize_outlier_decision(
    eda_summary_path: Path,
) -> dict[str, object]:
    """Read EDA outlier counts and attach the explicit retention decision.

    Args:
        eda_summary_path: Path to the consolidated EDA JSON report.

    Returns:
        Outlier counts unchanged from the EDA report and the Phase 4 decision.

    Raises:
        FileNotFoundError: If the EDA report is missing.
        KeyError: If the report has no outlier-count section.
        json.JSONDecodeError: If the report is not valid JSON.
    """
    with eda_summary_path.open(encoding="utf-8") as file_handle:
        eda_summary = json.load(file_handle)

    outlier_counts = eda_summary["outlier_counts"]

    return {
        "outlier_counts": outlier_counts,
        "decision": OUTLIER_DECISION,
        "untouched_columns": [
            "Discount_Amount_USD",
            "Order_Profit_USD",
            "Unit_Price_USD",
            "Tax_USD",
            "Total_Order_Value_USD",
            "Net_Sales_USD",
            "Gross_Sales_USD",
        ],
    }
