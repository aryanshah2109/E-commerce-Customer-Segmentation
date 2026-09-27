# Customer and Sales Analytics

An end-to-end, reproducible analytics and unsupervised-learning project for an e-commerce transaction dataset. The workflow turns transaction-level data into validated datasets, descriptive sales intelligence, customer-level behavioral features, and a reusable customer-segmentation model.

The project intentionally separates two objectives:

- **Sales analytics** is descriptive: KPIs, distributions, product/customer analysis, trends, weekday patterns, correlations, and transaction-gap diagnostics.
- **Customer segmentation** is the machine-learning component: customer behavior is represented with engineered features, compared across clustering approaches, and modeled with a stable PCA + KMeans pipeline.

## Current project status

The production-ready portion of the repository is the Python batch pipeline under `src/` and `scripts/`. It includes validation, EDA, preprocessing, feature engineering, deterministic splitting, model comparison, training, artifact generation, logging, and best-model promotion.

The `api/` and `streamlit_app/` directories are reserved for a FastAPI service and Streamlit dashboard respectively, but their current files are empty scaffolding. They are not required to run the pipeline and should not be treated as active application entry points yet.

## Key results

The current generated artifacts represent a run on an e-commerce dataset containing **200,000 transaction records and 50 source columns**.

| Item | Result |
|---|---:|
| Customer feature rows | 69,728 |
| Daily sales feature rows | 365 |
| Selected model | PCA-2 + KMeans-2 |
| Customer features used | 12 |
| Full-data silhouette score | 0.3876 |
| Full-data Davies–Bouldin score | 0.9758 |
| Cluster 0 / Cluster 1 | 35,311 / 34,417 |
| Random seed | 42 |

The selected model is a useful behavioral segmentation baseline, not a claim of perfectly separated customer populations. Cluster meaning should be interpreted through `cluster_profile.csv` and the engineered features rather than through cluster numbers alone.

## Workflow

The complete pipeline executes these stages in dependency order:

```text
raw.csv
  │
  ├─ prepare_data.py       → validated.csv + validation_report.json
  ├─ run_eda.py            → EDA reports + PNG figures
  ├─ preprocess_data.py    → preprocessed.csv + preprocessing_report.json
  ├─ build_features.py     → customer_features.csv + sales_features.csv
  ├─ create_splits.py      → deterministic train/validation/test partitions
  └─ train_segmentation.py→ model bundle + metrics + cluster profile
```

The orchestration code is in `src/customer_sales_analytics/pipelines/main_pipeline.py`; the command-line files in `scripts/` are thin stage entry points.

## Dataset

The expected input file is `data/raw/raw.csv`.

The supplied dataset contains transaction, customer, product, payment, fulfillment, marketing, return, and review fields. Important groups include:

- Identifiers: `Order_ID`, `Customer_ID`, `Product_ID`, `SKU`
- Dates and transaction measures: `Order_Date`, `Quantity`, `Unit_Price_USD`, `Gross_Sales_USD`, `Net_Sales_USD`, `Total_Order_Value_USD`, `Order_Profit_USD`
- Customer attributes: age, gender, geography, segment, repeat/new-customer flags
- Product attributes: category, subcategory, brand, bestseller flag
- Operational fields: payment, shipping, delivery, order status, warehouse, supplier
- Post-purchase fields: return status/reason, rating, review sentiment

See [`docs/data_dictionary.md`](docs/data_dictionary.md) for the documented source schema. The raw file is retained unchanged; derived datasets are written to `data/interim/` and `data/processed/`.

## Installation

The project targets **Python 3.14 or newer** and uses `uv` for environment and dependency management.

From the repository root:

```powershell
uv sync
```

Alternatively, use the project virtual environment directly after dependencies have been installed:

```powershell
.\.venv\Scripts\python.exe --version
```

The main dependencies include pandas, NumPy, scikit-learn, Pandera, PyYAML, matplotlib, seaborn, statsmodels, pytest, Ruff, Black, mypy, and Streamlit.

## Run the complete pipeline

From the project root:

```powershell
uv run customer-sales-pipeline
```

The equivalent direct command is:

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline.py
```

Configuration files can be overridden when running experiments or using a different dataset layout:

```powershell
uv run customer-sales-pipeline `
  --data-config configs/data_config.yaml `
  --paths-config configs/paths.yaml `
  --preprocessing-config configs/preprocessing_config.yaml
```

Each pipeline run writes a timestamped log to `logs/` and mirrors log messages to the console. The pipeline stops at the first failed stage and returns that stage's exit code.

## Run individual stages

Run individual scripts when debugging or inspecting an intermediate result:

```powershell
uv run python scripts\prepare_data.py
uv run python scripts\run_eda.py
uv run python scripts\preprocess_data.py
uv run python scripts\build_features.py
uv run python scripts\create_splits.py
uv run python scripts\train_segmentation.py
```

To compare candidate segmentation models independently of the end-to-end training stage:

```powershell
uv run python scripts\compare_segmentation_models.py
```

Model comparison is an analysis script and is not automatically executed by `run_pipeline.py`.

## Data validation and preprocessing

Validation is implemented under `src/customer_sales_analytics/data/` and includes:

- Required-column and schema checks using Pandera.
- Date, numeric, integer, and boolean type coercion.
- Duplicate-row detection.
- Null-rate reporting.
- Categorical-domain checks.
- Range checks for age, quantity, discounts, tax, rating, and delivery days.
- Aggregate error reporting so multiple data-quality issues can be inspected together.

Preprocessing removes duplicate rows, fixes types, documents future dates and outlier decisions, fills semantically not-applicable values, and records modeling exclusions. The default not-applicable replacements are:

```yaml
Return_Reason: Not Returned
Coupon_Code: No Coupon
```

The following fields are excluded from the default modeling input because they are outcome, review, or leakage-prone fields:

```text
Return_Status
Return_Reason
Customer_Rating
Review_Sentiment
Order_Profit_USD
```

The project does not silently delete unusual observations as part of EDA. Outlier treatment and exclusion decisions are documented in generated reports.

## Feature engineering

Customer features are built at customer level from transaction-level data. The current feature table includes:

```text
Customer_ID
recency_days
frequency
monetary_value
avg_order_value
order_value_std
product_diversity
purchase_velocity
preferred_category
preferred_payment_method
return_rate
coupon_usage_rate
```

The customer pipeline combines RFM-style features with behavioral summaries. `Customer_ID` is retained for traceability but identity and raw transaction text are not used as behavioral signals by the selected numeric modeling pipeline.

Sales features are daily and include calendar fields, aggregated sales measures, rolling statistics, and explicit gap-day indicators. The complete calendar range is reindexed so zero-transaction days remain visible. `gap_filled_sales` is retained for descriptive smoothing; observed `total_sales` remains the authoritative total.

## Segmentation methodology

The model-comparison workflow evaluates multiple representations and clustering families, including:

- Standardized numeric features with KMeans.
- PCA representations with KMeans.
- Gaussian mixture candidates.
- Agglomerative clustering.
- Robust-scaling and robust-PCA candidates.
- Optional HDBSCAN candidates when available in the comparison workflow.

Models are evaluated primarily with:

- **Silhouette score**, maximized.
- **Davies–Bouldin score**, minimized.
- **Multi-seed stability**, measured using seeds `42, 43, 44, 45, 46`.

The comparison selection rule prefers the highest `silhouette_mean - silhouette_std` among candidates whose silhouette standard deviation is at most `0.05`. The promoted production artifact currently uses standardized numeric customer features, PCA with two components, and two-cluster KMeans.

The final training script fits the selected pipeline on all customer features, computes full-data metrics, writes a candidate bundle, and promotes it to `models/segmentation/best_model/` only when it improves the stored silhouette score. Davies–Bouldin is used as the tie-breaker.

## Generated outputs

Important output locations are configured in [`configs/paths.yaml`](configs/paths.yaml):

| Location | Contents |
|---|---|
| `data/interim/` | Validated and preprocessed transaction tables |
| `data/processed/customer_features/` | Customer-level feature table |
| `data/processed/sales_features/` | Daily sales feature table |
| `data/splits/` | Customer and sales train/validation/test partitions |
| `artifacts/reports/` | JSON reports, cluster profiles, and diagnostics |
| `artifacts/eda/` | PNG charts from descriptive EDA |
| `artifacts/model_comparison/` | Candidate comparison tables and stability artifacts |
| `models/segmentation/candidates/` | Selected candidate model bundle |
| `models/segmentation/best_model/` | Stable promoted model directory |
| `logs/` | Timestamped pipeline logs |

The stable model directory contains:

```text
models/segmentation/best_model/
├── best_model.joblib       # fitted preprocessing pipeline + estimator + metadata
├── best_metrics.json       # metrics, hyperparameters, feature names, cluster sizes
└── cluster_profile.csv     # original-feature summaries by cluster
```

Useful reports include:

```text
artifacts/reports/validation_report.json
artifacts/reports/eda_summary.json
artifacts/reports/preprocessing_report.json
artifacts/reports/feature_engineering_report.json
artifacts/reports/split_report.json
artifacts/reports/segmentation_model_comparison_report.json
artifacts/reports/segmentation_training_report.json
```

Generated data, artifacts, models, logs, and documentation build outputs are ignored by Git. Recreate them by running the pipeline after placing the raw dataset at the configured input path.

## Configuration

| File | Responsibility |
|---|---|
| `configs/paths.yaml` | Input, output, artifact, model, and split paths; global seed |
| `configs/data_config.yaml` | Dataset validation, EDA, feature-engineering, split, and model-comparison settings |
| `configs/preprocessing_config.yaml` | Null-rate threshold, not-applicable fills, and modeling exclusions |
| `pyproject.toml` | Package metadata, dependencies, build configuration, and CLI entry points |

The default split configuration is:

```text
train:      70%
validation: 15%
test:       15%
seed:       42
```

Customer segmentation splits are deterministic. Sales splits are chronological so future sales information does not enter earlier partitions.

## Repository layout

```text
.
├── api/                         # Reserved FastAPI service structure
├── artifacts/                   # Generated reports, charts, and comparisons
├── configs/                     # YAML configuration
├── data/                        # Raw, interim, processed, and split data
├── docs/                        # Architecture and data documentation
├── logs/                        # Runtime logs
├── models/                      # Candidate and promoted model artifacts
├── scripts/                     # Thin stage-specific CLI entry points
├── src/customer_sales_analytics/
│   ├── config/                  # YAML loading and logging
│   ├── data/                    # Loading, schema, validation, splitting
│   ├── eda/                     # Compute-only analysis and rendering
│   ├── features/                # Customer and sales feature engineering
│   ├── preprocessing/           # Cleaning and exclusion documentation
│   ├── segmentation/            # Pipelines, models, comparison, profiling
│   └── pipelines/               # End-to-end orchestration
├── streamlit_app/               # Reserved dashboard structure
├── pyproject.toml
└── README.md
```

## Development checks

Run the available test suite and static checks from the repository root:

```powershell
uv run pytest
uv run ruff check .
uv run black --check .
uv run mypy src scripts
```

If a check is not configured for a local environment, install the dependencies with `uv sync` first. Running the full pipeline is the strongest integration check because it exercises the data contract, feature outputs, splitting, model training, and artifact promotion together.

## Limitations and scope

- This is an unsupervised segmentation workflow; there is no labeled target or claim of causal business impact.
- Sales outputs are descriptive and are not forecasts.
- Cluster numbers are arbitrary labels and can change if the data or selected model changes.
- The current production model is intentionally compact and stable rather than an exhaustive automated model-serving system.
- FastAPI and Streamlit files are present as future integration points but are currently empty and do not expose working endpoints or dashboard pages.

## License and ownership

No license file is currently included. Treat this repository as a project artifact and add an explicit license before redistributing it.
