"""Checklist step 4: describe product volume, revenue, and category mix."""

from __future__ import annotations

import pandas as pd


def compute_product_analysis(
    data_frame: pd.DataFrame,
    top_n: int,
) -> dict[str, pd.DataFrame]:
    """Compute the most important products by volume and revenue.

    Args:
        data_frame: Transaction-level sales data.
        top_n: Number of products to return in each top-products table.

    Returns:
        Top products by revenue, top products by volume, and category mix.

    Raises:
        ValueError: If top_n is not positive.
    """
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    product_groups = data_frame.groupby(
        ["Product_ID", "Product_Name"],
        dropna=False,
    )
    product_df = product_groups.agg(
        volume=("Quantity", "sum"),
        revenue=("Net_Sales_USD", "sum"),
    )

    top_by_revenue = product_df.sort_values(
        "revenue",
        ascending=False,
    ).head(top_n)
    top_by_volume = product_df.sort_values(
        "volume",
        ascending=False,
    ).head(top_n)

    result = {
        "top_by_revenue": top_by_revenue,
        "top_by_volume": top_by_volume,
    }

    if "Product_Category" in data_frame.columns:
        category_groups = data_frame.groupby(
            "Product_Category",
            dropna=False,
        )
        category_mix = category_groups.agg(
            volume=("Quantity", "sum"),
            revenue=("Net_Sales_USD", "sum"),
        )
        result["category_mix"] = category_mix.sort_values(
            "revenue",
            ascending=False,
        )

    return result
