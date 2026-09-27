"""CLI entrypoint for deterministic Phase 4 preprocessing."""

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
from customer_sales_analytics.preprocessing.cleaning import (
    fix_dtypes,
    flag_future_order_dates,
    remove_duplicate_rows,
)
from customer_sales_analytics.preprocessing.leakage_documenter import (
    write_modeling_exclusions,
)
from customer_sales_analytics.preprocessing.missing_value_handler import (
    check_remaining_missing_values,
    fill_not_applicable_values,
)
from customer_sales_analytics.preprocessing.outlier_documenter import (
    summarize_outlier_decision,
)

LOGGER = logging.getLogger(__name__)


def _count_filled_values(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    fill_map: dict[str, str],
) -> dict[str, int]:
    """Count changes made by the configured fill operation."""
    filled_counts = {}

    for column in fill_map:
        missing_before = int(before_df[column].isna().sum())
        missing_after = int(after_df[column].isna().sum())
        filled_counts[column] = missing_before - missing_after

    return filled_counts


def main() -> int:
    """Run the configured deterministic preprocessing workflow.

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
    configure_logging("preprocess_data")

    try:
        paths_config = load_yaml(args.paths_config)
        preprocessing_config = load_yaml(args.preprocessing_config)

        input_path = Path(paths_config["data"]["interim"])
        output_path = Path(paths_config["data"]["preprocessed_data_path"])
        report_path = (
            Path(paths_config["artifacts"]["reports"])
            / "preprocessing_report.json"
        )
        eda_summary_path = (
            Path(paths_config["artifacts"]["reports"])
            / "eda_summary.json"
        )

        data_frame = load_raw_data(input_path)
        input_row_count = len(data_frame)
        data_frame = remove_duplicate_rows(data_frame)
        duplicate_rows_removed = input_row_count - len(data_frame)
        data_frame = fix_dtypes(data_frame)
        future_date_report = flag_future_order_dates(data_frame)

        fill_map = preprocessing_config["not_applicable_fill_values"]
        data_before_fill = data_frame.copy()
        data_frame = fill_not_applicable_values(data_frame, fill_map)
        filled_counts = _count_filled_values(
            data_before_fill,
            data_frame,
            fill_map,
        )
        remaining_missing_columns = check_remaining_missing_values(
            data_frame,
            preprocessing_config["null_rate_threshold"],
        )
        outlier_report = summarize_outlier_decision(eda_summary_path)
        leakage_columns = write_modeling_exclusions(
            eda_summary_path,
            args.preprocessing_config,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        data_frame.to_csv(output_path, index=False)
        file_size_bytes = output_path.stat().st_size

        report = {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "duplicate_rows_removed": duplicate_rows_removed,
            "future_dates": future_date_report,
            "filled_values": filled_counts,
            "remaining_missing_columns": remaining_missing_columns,
            "outlier_decision": outlier_report,
            "modeling_exclude_columns": leakage_columns,
            "output_shape": list(data_frame.shape),
            "output_file_size_bytes": file_size_bytes,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        LOGGER.info(
            "Wrote preprocessed data: %d rows, %d columns, %d bytes",
            data_frame.shape[0],
            data_frame.shape[1],
            file_size_bytes,
        )
        return 0
    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        LOGGER.error("Preprocessing failed: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
