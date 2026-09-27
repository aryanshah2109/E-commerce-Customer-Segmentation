"""CLI entrypoint for loading, validating, and writing interim data."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from customer_sales_analytics.config.io import load_yaml
from customer_sales_analytics.data.loader import load_raw_data
from customer_sales_analytics.data.validation import DataValidationError, validate_data_frame

LOGGER = logging.getLogger(__name__)


def main() -> int:
    """Run the data preparation CLI.

    Returns:
        Process exit code, zero on successful validation and write.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, default=Path("configs/data_config.yaml"))
    parser.add_argument("--paths-config", type=Path, default=Path("configs/paths.yaml"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    data_config = load_yaml(args.data_config)
    paths_config = load_yaml(args.paths_config)
    seed = int(data_config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    LOGGER.info("Using seed %d", seed)
    raw_path = Path(paths_config["data"]["raw"])
    output_path = Path(paths_config["data"]["interim"])
    report_path = Path(paths_config["artifacts"]["reports"]) / "validation_report.json"
    try:
        data_frame = load_raw_data(raw_path)
        validation_config = data_config["validation"]
        report = validate_data_frame(
            data_frame,
            max_null_rate=validation_config["max_null_rate"],
            allowed_categories=validation_config.get("categorical_values", {}),
            allow_duplicate_rows=validation_config["allow_duplicate_rows"],
            nullable_columns=validation_config.get("nullable_columns", []),
        )
    except (DataValidationError, FileNotFoundError, ValueError, KeyError) as error:
        LOGGER.error("Data preparation failed: %s", error)
        if isinstance(error, DataValidationError):
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps({"passed": False, "issues": [issue.__dict__ for issue in error.report.issues]}, default=str, indent=2), encoding="utf-8")
        return 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    data_frame.to_csv(output_path, index=False)
    report_path.write_text(json.dumps({"passed": report.passed, "issues": [issue.__dict__ for issue in report.issues], "rows": report.row_count, "columns": report.column_count}, default=str, indent=2), encoding="utf-8")
    LOGGER.info("Wrote validated data to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
