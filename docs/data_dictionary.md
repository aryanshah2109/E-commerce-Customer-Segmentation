# Data dictionary

Confirmed from `data/raw/raw.csv` on 2026-09-23. The file contains 200,000 rows and 50 columns.

| Column | Observed type | Nullable | Meaning |
|---|---|---:|---|
| Order_ID | string | No | Order identifier |
| Order_Date | date | No | Calendar date of order |
| Order_Time | string | No | Time of order |
| Customer_ID, Customer_Name | string | No | Customer identity fields |
| Customer_Age | integer | No | Customer age |
| Customer_Gender, Customer_Country, Customer_Continent, Customer_City, Customer_Region | string | No | Customer geography/demographics |
| Customer_Segment | string | No | Existing descriptive segment label |
| Product_ID, SKU, Product_Name | string | No | Product identifiers and name |
| Product_Category, Product_Subcategory, Brand | string | No | Product taxonomy |
| Quantity | integer | No | Units in transaction |
| Unit_Price_USD, Gross_Sales_USD, Discount_Amount_USD, Net_Sales_USD, Tax_USD, Shipping_Cost_USD, Total_Order_Value_USD, Order_Profit_USD | float | No | Monetary transaction measures |
| Discount_Percentage, Tax_Rate_Percentage, Profit_Margin_Percentage | float | No | Percentage measures |
| Payment_Method, Payment_Status, Order_Status, Shipping_Method | string | No | Transaction fulfillment state |
| Delivery_Days | integer | No | Delivery duration |
| Return_Status, Return_Reason | string | `Return_Reason` yes | Return information |
| Marketing_Channel, Device_Type | string | No | Acquisition and device dimensions |
| Coupon_Used | boolean | No | Whether a coupon was used |
| Coupon_Code | string | Yes | Coupon identifier |
| Customer_Rating | float | No | Rating from 1 to 5 |
| Review_Sentiment | string | No | Positive, Neutral, or Negative |
| Is_Repeat_Customer, Is_Bestseller_Product, Is_New_Customer, Is_First_Order | boolean | No | Existing business flags |
| Sales_Region, Warehouse, Supplier | string | No | Commercial operations dimensions |

Observed null rates are 95.84% for `Return_Reason` and 85.08% for `Coupon_Code`; these are intentional nullable fields but exceed the general 20% warning threshold.
