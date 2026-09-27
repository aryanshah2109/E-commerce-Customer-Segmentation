"""CLI entrypoint for deterministic Phase 5 feature engineering."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from customer_sales_analytics.config.io import load_yaml
from customer_sales_analytics.config.logging import configure_logging
from customer_sales_analytics.data.loader import load_raw_data
from customer_sales_analytics.features.customer.pipeline import (
    build_customer_feature_table,
)
from customer_sales_analytics.features.sales.pipeline import build_sales_feature_table

LOGGER = logging.getLogger(__name__)


def _resolve_reference_date(
    configured_date: str | None,
    data_frame: pd.DataFrame,
) -> pd.Timestamp:
    """Resolve a configured or data-derived recency reference date."""
    if configured_date is None:
        return pd.to_datetime(data_frame["Order_Date"], errors="raise").max()

    return pd.Timestamp(configured_date)


def main() -> int:
    """Build and save both Phase 5 feature tables.

    Returns:
        Zero on success, one when configuration, input, or processing fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data_config.yaml"),
    )
    parser.add_argument(
        "--paths-config",
        type=Path,
        default=Path("configs/paths.yaml"),
    )
    parser.add_argument(
        "--preprocessing-config",
        type=Path,
        default=Path("configs/preprocessing_config.yaml"),
    )
    args = parser.parse_args()
    configure_logging("build_features")

    try:
        data_config = load_yaml(args.data_config)
        paths_config = load_yaml(args.paths_config)
        preprocessing_config = load_yaml(args.preprocessing_config)
        feature_config = data_config["feature_engineering"]
        input_path = Path(paths_config["data"]["preprocessed_data_path"])
        customer_output_path = Path(paths_config["data"]["customer_features"])
        sales_output_path = Path(paths_config["data"]["sales_features"])
        report_path = (
            Path(paths_config["artifacts"]["reports"])
            / "feature_engineering_report.json"
        )

        data_frame = load_raw_data(input_path)
        reference_date = _resolve_reference_date(
            feature_config.get("reference_date"),
            data_frame,
        )
        customer_features = build_customer_feature_table(
            data_frame,
            reference_date,
        )
        sales_features = build_sales_feature_table(data_frame)

        customer_output_path.parent.mkdir(parents=True, exist_ok=True)
        sales_output_path.parent.mkdir(parents=True, exist_ok=True)
        customer_features.to_csv(customer_output_path, index=False)
        sales_features.to_csv(sales_output_path, index=False)
        zero_transaction_days = int(
            (sales_features["order_count"] == 0).sum()
        )

        report = {
            "customer_features": {
                "rows": customer_features.shape[0],
                "columns": customer_features.shape[1],
                "output_path": str(customer_output_path),
            },
            "sales_features": {
                "rows": sales_features.shape[0],
                "columns": sales_features.shape[1],
                "output_path": str(sales_output_path),
            },
            "reference_date": reference_date.date().isoformat(),
            "deliberately_reused_modeling_exclude_columns": {
                "Return_Status": (
                    "Used to calculate return_rate as a descriptive customer "
                    "behavior signal for segmentation; segmentation has no "
                    "future-outcome target."
                ),
            },
            "zero_transaction_days_reindexed": (
                f"{zero_transaction_days} zero-transaction days were reindexed "
                "as zero-value rows, not dropped."
            ),
            "gap_day_column": "is_gap_day marks zero-transaction data gaps.",
            "gap_filled_sales_usage": (
                "gap_filled_sales is retained as a gap-smoothed descriptive "
                "series; total_sales remains the observed daily total."
            ),
            "scaling_or_encoding": "Not introduced in Phase 5.",
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        LOGGER.info("Wrote customer features to %s", customer_output_path)
        LOGGER.info("Wrote sales features to %s", sales_output_path)
        LOGGER.info("Wrote feature engineering report to %s", report_path)
        return 0
    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        LOGGER.error("Feature engineering failed: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
