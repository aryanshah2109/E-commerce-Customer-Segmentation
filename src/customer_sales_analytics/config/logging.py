"""Shared timestamped logging configuration for CLI pipeline stages."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path


def configure_logging(run_name: str) -> Path:
    """Configure console and timestamped file logging for one run.

    Args:
        run_name: Prefix used when creating a standalone log file.

    Returns:
        Path to the active log file.
    """
    configured_path = os.environ.get("PROJECT_PIPELINE_LOG_PATH")
    if configured_path:
        log_path = Path(configured_path)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_path = Path("logs") / f"{run_name}_{timestamp}.log"

    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )
    return log_path
