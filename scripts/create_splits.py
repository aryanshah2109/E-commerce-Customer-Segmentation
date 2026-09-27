"""CLI entrypoint for Phase 6 train/validation/test splitting."""

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
from customer_sales_analytics.data.splitting import (
    split_customer_features,
    split_sales_features,
)

LOGGER = logging.getLogger(__name__)


def _date_range_report(data_frame: pd.DataFrame) -> dict[str, str | None]:
    """Return the inclusive date range for one sales split."""
    if data_frame.empty:
        return {"min": None, "max": None}

    order_dates = pd.to_datetime(data_frame["Order_Date"], errors="raise")
    return {
        "min": order_dates.min().date().isoformat(),
        "max": order_dates.max().date().isoformat(),
    }


def main() -> int:
    """Create and save independent customer and sales data splits.

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
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        data_config = load_yaml(args.data_config)
        paths_config = load_yaml(args.paths_config)
        seed = int(data_config["seed"])
        splitting_config = data_config["splitting"]
        train_fraction = float(splitting_config["train_fraction"])
        validation_fraction = float(splitting_config["validation_fraction"])
        LOGGER.info("Using seed %d for customer row-level split", seed)

        customer_input_path = Path(paths_config["data"]["customer_features"])
        sales_input_path = Path(paths_config["data"]["sales_features"])
        report_path = (
            Path(paths_config["artifacts"]["reports"])
            / "split_report.json"
        )
        customer_data = pd.read_csv(customer_input_path)
        sales_data = pd.read_csv(sales_input_path)
        customer_splits = split_customer_features(
            customer_data,
            train_fraction,
            validation_fraction,
            seed,
        )
        sales_splits = split_sales_features(
            sales_data,
            train_fraction,
            validation_fraction,
        )

        split_names = ("train", "validation", "test")
        customer_paths = paths_config["splits"]["customer_segmentation"]
        sales_paths = paths_config["splits"]["sales_analysis"]
        customer_report = {}
        sales_report = {}
        for split_name, split_data in zip(split_names, customer_splits):
            output_directory = Path(customer_paths[split_name])
            output_directory.mkdir(parents=True, exist_ok=True)
            output_path = output_directory / f"{split_name}.csv"
            split_data.to_csv(output_path, index=False)
            customer_report[split_name] = {
                "rows": len(split_data),
                "columns": split_data.shape[1],
                "output_path": str(output_path),
            }

        for split_name, split_data in zip(split_names, sales_splits):
            output_directory = Path(sales_paths[split_name])
            output_directory.mkdir(parents=True, exist_ok=True)
            output_path = output_directory / f"{split_name}.csv"
            split_data.to_csv(output_path, index=False)
            sales_report[split_name] = {
                "rows": len(split_data),
                "columns": split_data.shape[1],
                "output_path": str(output_path),
                "date_range": _date_range_report(split_data),
            }

        report = {
            "seed": seed,
            "train_fraction": train_fraction,
            "validation_fraction": validation_fraction,
            "test_fraction": round(
                1.0 - train_fraction - validation_fraction,
                10,
            ),
            "customer_segmentation": customer_report,
            "sales_analysis": sales_report,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        LOGGER.info("Wrote split report to %s", report_path)
        return 0
    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        LOGGER.error("Split creation failed: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
