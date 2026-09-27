"""Run the complete customer and sales analytics workflow."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from customer_sales_analytics.config.logging import configure_logging

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"


@dataclass(frozen=True)
class PipelineStage:
    """Describe one command in the end-to-end workflow."""

    name: str
    script_path: Path
    arguments: tuple[str, ...]


def _build_stages(
    data_config_path: Path,
    paths_config_path: Path,
    preprocessing_config_path: Path,
) -> tuple[PipelineStage, ...]:
    """Build the ordered pipeline stages from shared configuration paths.

    Args:
        data_config_path: YAML file containing data and modeling settings.
        paths_config_path: YAML file containing project input/output paths.
        preprocessing_config_path: YAML file containing cleaning settings.

    Returns:
        Ordered stages from raw-data validation through segmentation.
    """
    common_arguments = (
        "--data-config",
        str(data_config_path),
        "--paths-config",
        str(paths_config_path),
    )
    return (
        PipelineStage(
            name="prepare data",
            script_path=SCRIPTS_ROOT / "prepare_data.py",
            arguments=common_arguments,
        ),
        PipelineStage(
            name="run descriptive EDA",
            script_path=SCRIPTS_ROOT / "run_eda.py",
            arguments=common_arguments,
        ),
        PipelineStage(
            name="preprocess data",
            script_path=SCRIPTS_ROOT / "preprocess_data.py",
            arguments=common_arguments
            + ("--preprocessing-config", str(preprocessing_config_path)),
        ),
        PipelineStage(
            name="build features",
            script_path=SCRIPTS_ROOT / "build_features.py",
            arguments=common_arguments
            + ("--preprocessing-config", str(preprocessing_config_path)),
        ),
        PipelineStage(
            name="create data splits",
            script_path=SCRIPTS_ROOT / "create_splits.py",
            arguments=common_arguments,
        ),
        PipelineStage(
            name="train selected segmentation model",
            script_path=SCRIPTS_ROOT / "train_segmentation.py",
            arguments=common_arguments,
        ),
    )


def run_pipeline(
    data_config_path: Path,
    paths_config_path: Path,
    preprocessing_config_path: Path,
) -> int:
    """Run every pipeline stage in dependency order.

    Args:
        data_config_path: YAML file containing data and modeling settings.
        paths_config_path: YAML file containing project input/output paths.
        preprocessing_config_path: YAML file containing cleaning settings.

    Returns:
        Zero when every stage succeeds, otherwise the failed stage exit code.
    """
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    warnings.filterwarnings(
        "ignore",
        message="the number of connected components.*",
        category=UserWarning,
        module="sklearn",
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = PROJECT_ROOT / "logs" / f"pipeline_{timestamp}.log"
    os.environ["PROJECT_PIPELINE_LOG_PATH"] = str(log_path)
    configure_logging("pipeline")
    LOGGER.info("Pipeline log file: %s", log_path)

    stages = _build_stages(
        data_config_path,
        paths_config_path,
        preprocessing_config_path,
    )
    for stage in stages:
        command = [
            sys.executable,
            str(stage.script_path),
            *stage.arguments,
        ]
        LOGGER.info("Starting stage: %s", stage.name)
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
        )
        if result.returncode != 0:
            LOGGER.error(
                "Stage failed: %s (exit code %d)",
                stage.name,
                result.returncode,
            )
            return result.returncode
        LOGGER.info("Completed stage: %s", stage.name)

    LOGGER.info("End-to-end pipeline completed successfully")
    return 0


def main() -> int:
    """Parse pipeline options and run the complete workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data_config.yaml",
    )
    parser.add_argument(
        "--paths-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "paths.yaml",
    )
    parser.add_argument(
        "--preprocessing-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "preprocessing_config.yaml",
    )
    args = parser.parse_args()
    return run_pipeline(
        args.data_config,
        args.paths_config,
        args.preprocessing_config,
    )


if __name__ == "__main__":
    raise SystemExit(main())
