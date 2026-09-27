"""CLI entrypoint for descriptive EDA on the full validated interim dataset."""

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
from customer_sales_analytics.config.logging import configure_logging
from customer_sales_analytics.data.loader import load_raw_data
from customer_sales_analytics.eda.categorical_analysis import compute_categorical_analysis
from customer_sales_analytics.eda.correlation_analysis import compute_correlation_analysis
from customer_sales_analytics.eda.customer_analysis import compute_customer_analysis
from customer_sales_analytics.eda.dataset_overview import compute_dataset_overview
from customer_sales_analytics.eda.distributions import compute_distributions
from customer_sales_analytics.eda.missing_values import compute_missing_values
from customer_sales_analytics.eda.product_analysis import compute_product_analysis
from customer_sales_analytics.eda.sales_analysis import (
    analyze_zero_transaction_days,
    compute_sales_analysis,
    compute_sales_kpis,
    compute_monthly_trend,
    compute_weekday_pattern,
)
from customer_sales_analytics.eda.visualizations import (
    render_categorical_bars,
    render_correlations,
    render_dtypes_table,
    render_missing_values,
    render_numeric_histograms,
    render_outlier_summary,
    render_sales_trends,
    render_monthly_trend,
    render_weekday_pattern,
    render_top_products,
)

LOGGER = logging.getLogger(__name__)


def main() -> int:
    """Run all descriptive EDA computations and render artifacts.

    Returns:
        Process exit code.
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
    configure_logging("eda")
    data_config = load_yaml(args.data_config)
    paths_config = load_yaml(args.paths_config)
    seed = int(data_config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    LOGGER.info("Using seed %d", seed)
    interim_path = Path(paths_config["data"]["interim"])
    input_path = interim_path if interim_path.is_file() else Path(paths_config["data"]["raw"])
    data_frame = load_raw_data(input_path)
    eda_config = data_config["eda"]
    report_dir = Path(paths_config["artifacts"]["reports"])
    figure_dir = Path(paths_config["artifacts"]["eda"])
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    overview = compute_dataset_overview(data_frame)
    missing = compute_missing_values(
        data_frame,
        data_config["validation"]["max_null_rate"],
    )
    distributions = compute_distributions(
        data_frame,
        eda_config["low_cardinality_max"],
    )
    categorical = compute_categorical_analysis(
        data_frame,
        eda_config["rare_category_threshold"],
        eda_config["top_n"],
    )
    correlations = compute_correlation_analysis(
        data_frame,
        eda_config["correlation_threshold"],
    )
    customer = compute_customer_analysis(data_frame)
    products = compute_product_analysis(
        data_frame,
        eda_config["top_n"],
    )
    sales = compute_sales_analysis(
        data_frame,
        eda_config["rolling_window"],
    )
    zero_transaction_analysis = analyze_zero_transaction_days(data_frame)
    sales_kpis = compute_sales_kpis(data_frame)
    monthly_trend = compute_monthly_trend(data_frame)
    weekday_pattern = compute_weekday_pattern(data_frame)

    # These columns are the most interpretable numeric measures for this retail
    # dataset.  They are used because no chart-column list exists in config.
    numeric_histogram_columns = [
        "Gross_Sales_USD",
        "Net_Sales_USD",
        "Total_Order_Value_USD",
        "Order_Profit_USD",
    ]
    categorical_bar_columns = [
        "Customer_Segment",
        "Product_Category",
        "Payment_Status",
    ]

    render_missing_values(missing, figure_dir / "missing_values.png")
    render_correlations(correlations, figure_dir / "correlations.png")
    render_sales_trends(sales, figure_dir / "sales_trends.png")
    render_monthly_trend(monthly_trend, figure_dir / "monthly_trend.png")
    render_weekday_pattern(weekday_pattern, figure_dir / "weekday_pattern.png")
    render_top_products(products, figure_dir / "top_products.png")
    render_dtypes_table(overview, figure_dir / "dtypes_table.png")
    render_numeric_histograms(
        data_frame,
        numeric_histogram_columns,
        figure_dir / "numeric_histograms.png",
    )
    render_categorical_bars(
        distributions,
        categorical_bar_columns,
        figure_dir / "categorical_bars.png",
    )
    render_outlier_summary(
        distributions,
        figure_dir / "outlier_summary.png",
    )

    sorted_outlier_counts = sorted(
        distributions.outlier_counts.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    ordered_outlier_counts = {}
    for column, count in sorted_outlier_counts:
        ordered_outlier_counts[column] = count

    zero_dates = ", ".join(
        zero_transaction_analysis["zero_transaction_dates"]
    )
    if zero_transaction_analysis["all_within_final_three_days_of_month"]:
        zero_transaction_finding = (
            "The 13 zero-transaction dates are "
            f"{zero_dates}. All fall within the final three calendar days "
            "of a month, but they are not evenly spaced and span multiple "
            "weekdays and day-of-month values. This is a month-end data-gap "
            "pattern; no calendar feature was added because it is not a "
            "stable single-date signal."
        )
    else:
        zero_transaction_finding = (
            f"The {zero_transaction_analysis['count']} zero-transaction dates "
            f"are {zero_dates}. No clear calendar pattern was found, so no "
            "calendar feature was added."
        )
    notable_findings = [
        zero_transaction_finding,
        "Gross sales, net sales, total order value, tax, order profit, quantity, and unit price form correlation clusters. This multicollinearity is expected because these fields are arithmetically derived from one another, such as Net Sales = Gross Sales - Discount.",
    ]
    summary = {
        "overview": overview.__dict__,
        "dtypes": overview.dtypes,
        "missing_values": missing.reset_index().to_dict("records"),
        "categorical": categorical,
        "correlations": correlations.flagged_pairs,
        "customer": {
            "repeat_purchase_rate": customer["repeat_purchase_rate"],
        },
        "top_products": products["top_by_revenue"].reset_index().to_dict(
            "records"
        ),
        "possible_leakage_columns": overview.possible_leakage_columns,
        "sanity_checks": overview.sanity_checks,
        "outlier_counts": ordered_outlier_counts,
        "near_zero_variance_columns": distributions.near_zero_variance_columns,
        "zero_transaction_analysis": zero_transaction_analysis,
        "sales_kpis": sales_kpis,
        "notable_findings": notable_findings,
    }
    report_path = report_dir / "eda_summary.json"
    report_path.write_text(
        json.dumps(summary, default=str, indent=2),
        encoding="utf-8",
    )
    LOGGER.info(
        "Rows: %d; top missing: %s; notable correlations: %d",
        len(data_frame),
        list(missing.head(3).index),
        len(correlations.flagged_pairs),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
