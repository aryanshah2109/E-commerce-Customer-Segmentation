# Architecture additions: Phases 2 and 3

Phase 2 introduces an ingestion boundary under `src/customer_sales_analytics/data/`:

1. `loader.py` reads the configured raw CSV and coerces dates/numeric/boolean types.
2. `schema.py` owns the explicit Pandera schema and domain checks.
3. `validation.py` collects missing-column, schema, duplicate, null-rate, and category issues into a report and raises one aggregate error.
4. `splitting.py` provides deterministic train/validation/test splits for future modeling artifacts; EDA deliberately does not use splits.
5. `scripts/prepare_data.py` writes only to `data/interim/` and `artifacts/reports/`.

Phase 3 introduces compute-only descriptive modules under `src/customer_sales_analytics/eda/`. `visualizations.py` is the only rendering boundary and writes to `artifacts/eda/`. `scripts/run_eda.py` is a thin CLI that loads configuration, calls compute functions, renders figures, and writes a JSON summary.

The raw data is retained unchanged. Preprocessing and feature engineering are intentionally out of scope for this phase.
