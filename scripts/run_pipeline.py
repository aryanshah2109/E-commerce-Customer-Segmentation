"""CLI wrapper for the complete customer and sales analytics pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from customer_sales_analytics.pipelines.main_pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())
