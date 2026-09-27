"""Pandera schema and column metadata for the raw customer sales dataset."""

from __future__ import annotations

from typing import Final

import pandera.pandas as pa
from pandera.pandas import Check, DataFrameSchema

REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "Order_ID", "Order_Date", "Order_Time", "Customer_ID", "Customer_Name",
    "Customer_Age", "Customer_Gender", "Customer_Country", "Customer_Continent",
    "Customer_City", "Customer_Region", "Customer_Segment", "Product_ID", "SKU",
    "Product_Name", "Product_Category", "Product_Subcategory", "Brand", "Quantity",
    "Unit_Price_USD", "Gross_Sales_USD", "Discount_Percentage", "Discount_Amount_USD",
    "Net_Sales_USD", "Tax_Rate_Percentage", "Tax_USD", "Shipping_Cost_USD",
    "Total_Order_Value_USD", "Payment_Method", "Payment_Status", "Order_Status",
    "Shipping_Method", "Delivery_Days", "Return_Status", "Return_Reason",
    "Marketing_Channel", "Device_Type", "Coupon_Used", "Coupon_Code",
    "Customer_Rating", "Review_Sentiment", "Is_Repeat_Customer", "Sales_Region",
    "Warehouse", "Supplier", "Order_Profit_USD", "Profit_Margin_Percentage",
    "Is_Bestseller_Product", "Is_New_Customer", "Is_First_Order",
)

_TEXT_COLUMNS: Final[tuple[str, ...]] = (
    "Order_ID", "Order_Time", "Customer_ID", "Customer_Name", "Customer_Gender",
    "Customer_Country", "Customer_Continent", "Customer_City", "Customer_Region",
    "Customer_Segment", "Product_ID", "SKU", "Product_Name", "Product_Category",
    "Product_Subcategory", "Brand", "Payment_Method", "Payment_Status", "Order_Status",
    "Shipping_Method", "Return_Status", "Return_Reason", "Marketing_Channel",
    "Device_Type", "Coupon_Code", "Review_Sentiment", "Sales_Region", "Warehouse",
    "Supplier",
)
_INTEGER_COLUMNS: Final[tuple[str, ...]] = ("Customer_Age", "Quantity", "Delivery_Days")
_FLOAT_COLUMNS: Final[tuple[str, ...]] = (
    "Unit_Price_USD", "Gross_Sales_USD", "Discount_Percentage", "Discount_Amount_USD",
    "Net_Sales_USD", "Tax_Rate_Percentage", "Tax_USD", "Shipping_Cost_USD",
    "Total_Order_Value_USD", "Customer_Rating", "Order_Profit_USD",
    "Profit_Margin_Percentage",
)
_BOOLEAN_COLUMNS: Final[tuple[str, ...]] = (
    "Coupon_Used", "Is_Repeat_Customer", "Is_Bestseller_Product", "Is_New_Customer",
    "Is_First_Order",
)


def build_schema() -> DataFrameSchema:
    """Build the explicit Pandera schema for validated transaction rows.

    Returns:
        A strict Pandera schema with required columns, dtypes, nullability, and
        known business-domain checks.
    """
    columns = {name: pa.Column(str, nullable=False) for name in _TEXT_COLUMNS}
    columns["Order_Date"] = pa.Column(pa.DateTime, nullable=False)
    columns.update({name: pa.Column(int, nullable=False) for name in _INTEGER_COLUMNS})
    columns.update({name: pa.Column(float, nullable=False) for name in _FLOAT_COLUMNS})
    columns.update({name: pa.Column(bool, nullable=False) for name in _BOOLEAN_COLUMNS})
    columns["Return_Reason"] = pa.Column(str, nullable=True)
    columns["Coupon_Code"] = pa.Column(str, nullable=True)
    columns["Customer_Age"] = pa.Column(int, Check.ge(0), nullable=False)
    columns["Quantity"] = pa.Column(int, Check.gt(0), nullable=False)
    columns["Discount_Percentage"] = pa.Column(float, Check.in_range(0, 100), nullable=False)
    columns["Tax_Rate_Percentage"] = pa.Column(float, Check.in_range(0, 100), nullable=False)
    columns["Customer_Rating"] = pa.Column(float, Check.in_range(0, 5), nullable=False)
    return DataFrameSchema(columns, strict=True, coerce=False)


RAW_SCHEMA: Final[DataFrameSchema] = build_schema()
