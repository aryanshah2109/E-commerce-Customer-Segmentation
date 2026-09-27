"""Checklist visualizations: simple histograms, bars, and a Pearson heatmap."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from customer_sales_analytics.eda.correlation_analysis import CorrelationResults
from customer_sales_analytics.eda.dataset_overview import DatasetOverview
from customer_sales_analytics.eda.distributions import DistributionResults


def _save_figure(
    figure: Any,
    output_path: Path,
) -> None:
    """Save a figure and close it so scripts do not display plots."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def render_missing_values(
    result: pd.DataFrame,
    output_path: Path,
) -> None:
    """Render missing percentages as a horizontal bar chart."""
    figure, axis = plt.subplots()

    missing_rates = result["null_rate"].head(20).sort_values()
    missing_rates.plot.barh(
        ax=axis,
        title="Missing percentage by column",
    )

    _save_figure(figure, output_path)


def render_distributions(
    result: DistributionResults,
    output_path: Path,
) -> None:
    """Render numeric histograms and a simple boxplot summary."""
    numeric_summary = result.numeric_summary
    numeric_columns = list(numeric_summary.index)
    figure, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10, 8),
    )

    if numeric_columns:
        means = numeric_summary["mean"]
        means.plot.bar(ax=axes[0], title="Numeric column means")
        numeric_summary[["min", "max"]].T.boxplot(ax=axes[1])
        axes[1].set_title("Numeric minimum and maximum ranges")

    _save_figure(figure, output_path)


def render_correlations(
    result: CorrelationResults,
    output_path: Path,
) -> None:
    """Render the Pearson correlation matrix as a heatmap."""
    figure, axis = plt.subplots(figsize=(10, 8))

    correlation_image = axis.imshow(
        result.pearson.fillna(0),
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )
    axis.set_xticks(
        range(len(result.pearson.columns)),
        result.pearson.columns,
        rotation=90,
    )
    axis.set_yticks(
        range(len(result.pearson.index)),
        result.pearson.index,
    )
    figure.colorbar(correlation_image, ax=axis)
    axis.set_title("Pearson correlation")

    _save_figure(figure, output_path)


def render_sales_trends(
    result: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    """Render daily sales and the rolling mean."""
    figure, axis = plt.subplots(figsize=(12, 5))
    daily_df = result["daily"]

    daily_df[["sales", "rolling_mean"]].plot(
        ax=axis,
        title="Daily sales",
    )

    _save_figure(figure, output_path)


def render_monthly_trend(
    result: pd.DataFrame,
    output_path: Path,
) -> None:
    """Render total monthly sales with simple month-over-month labels."""
    figure, axis = plt.subplots(figsize=(12, 5))
    result.plot(
        x="month",
        y="total_sales",
        kind="line",
        marker="o",
        ax=axis,
        legend=False,
        title="Monthly sales trend",
    )
    for month_row in result.itertuples(index=False):
        if pd.notna(month_row.month_over_month_change):
            axis.annotate(
                f"{month_row.month_over_month_change:.1%}",
                (month_row.month, month_row.total_sales),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
            )
    axis.set_xlabel("Month")
    axis.set_ylabel("Net sales (USD)")
    _save_figure(figure, output_path)


def render_weekday_pattern(
    result: pd.Series,
    output_path: Path,
) -> None:
    """Render average daily sales ordered from Monday through Sunday."""
    figure, axis = plt.subplots(figsize=(10, 5))
    result.plot.bar(
        ax=axis,
        title="Average daily sales by weekday",
    )
    axis.set_xlabel("Weekday")
    axis.set_ylabel("Average daily sales (USD)")
    axis.tick_params(axis="x", rotation=0)
    _save_figure(figure, output_path)


def render_top_products(
    result: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    """Render top products by revenue as a horizontal bar chart."""
    figure, axis = plt.subplots(figsize=(10, 6))
    top_products_df = result["top_by_revenue"].sort_values(
        "revenue",
        ascending=True,
    ).reset_index()

    product_labels = []
    for _, product_row in top_products_df.iterrows():
        product_label = (
            f"{product_row['Product_Name']} "
            f"({product_row['Product_ID']})"
        )
        product_labels.append(product_label)

    top_products_df["product_label"] = product_labels
    revenue = top_products_df.set_index("product_label")["revenue"]

    revenue.plot.barh(
        ax=axis,
        title="Top products by revenue",
    )

    _save_figure(figure, output_path)


def render_dtypes_table(
    result: DatasetOverview,
    output_path: Path,
) -> None:
    """Render the dataset column names and dtypes as a simple table."""
    sorted_dtypes = sorted(result.dtypes.items())
    table_rows = []

    for column, dtype in sorted_dtypes:
        table_rows.append([column, dtype])

    figure_height = max(4, len(table_rows) * 0.25)
    figure, axis = plt.subplots(
        figsize=(8, figure_height),
    )
    axis.axis("off")
    axis.set_title("Column data types")
    table = axis.table(
        cellText=table_rows,
        colLabels=["Column", "Dtype"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width([0, 1])

    _save_figure(figure, output_path)


def render_numeric_histograms(
    data_frame: pd.DataFrame,
    columns: list[str],
    output_path: Path,
) -> None:
    """Render one histogram for each selected numeric column."""
    selected_columns = []
    for column in columns:
        if column in data_frame.columns:
            selected_columns.append(column)

    columns_per_row = 2
    plot_count = max(1, len(selected_columns))
    row_count = math.ceil(plot_count / columns_per_row)
    figure, axes = plt.subplots(
        nrows=row_count,
        ncols=columns_per_row,
        figsize=(10, 4 * row_count),
        squeeze=False,
    )
    axes_list = list(axes.flatten())

    if not selected_columns:
        axes_list[0].text(
            0.5,
            0.5,
            "No selected numeric columns",
        )
        axes_list[0].set_axis_off()

    for index, column in enumerate(selected_columns):
        axis = axes_list[index]
        data_frame[column].dropna().hist(
            ax=axis,
            bins=20,
        )
        axis.set_title(column)
        axis.set_xlabel(column)
        axis.set_ylabel("Count")

    for axis in axes_list[len(selected_columns) :]:
        axis.set_axis_off()

    _save_figure(figure, output_path)


def render_categorical_bars(
    result: DistributionResults,
    columns: list[str],
    output_path: Path,
) -> None:
    """Render one value-count bar chart for each selected category column."""
    selected_columns = []
    for column in columns:
        if column in result.categorical_counts:
            selected_columns.append(column)

    columns_per_row = 2
    plot_count = max(1, len(selected_columns))
    row_count = math.ceil(plot_count / columns_per_row)
    figure, axes = plt.subplots(
        nrows=row_count,
        ncols=columns_per_row,
        figsize=(12, 4 * row_count),
        squeeze=False,
    )
    axes_list = list(axes.flatten())

    if not selected_columns:
        axes_list[0].text(
            0.5,
            0.5,
            "No selected categorical columns",
        )
        axes_list[0].set_axis_off()

    for index, column in enumerate(selected_columns):
        axis = axes_list[index]
        counts = result.categorical_counts[column]
        counts.plot.bar(
            ax=axis,
            title=column,
        )
        axis.set_xlabel(column)
        axis.set_ylabel("Count")
        axis.tick_params(axis="x", rotation=45)

    for axis in axes_list[len(selected_columns) :]:
        axis.set_axis_off()

    _save_figure(figure, output_path)


def render_outlier_summary(
    result: DistributionResults,
    output_path: Path,
) -> None:
    """Render IQR outlier counts and any near-zero variance columns."""
    outlier_series = pd.Series(result.outlier_counts, dtype="int64")
    outlier_series = outlier_series.sort_values(ascending=True)
    figure, axis = plt.subplots(figsize=(10, 6))

    outlier_series.plot.barh(
        ax=axis,
        title="IQR outlier count by numeric column",
    )
    axis.set_xlabel("Outlier count")
    axis.set_ylabel("Column")

    if result.near_zero_variance_columns:
        variance_text = ", ".join(result.near_zero_variance_columns)
        axis.text(
            1.02,
            0.5,
            f"Near-zero variance:\n{variance_text}",
            transform=axis.transAxes,
            va="center",
        )

    _save_figure(figure, output_path)
