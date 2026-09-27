# Customer and Sales Analytics

This project validates and preprocesses e-commerce data, builds customer and
sales features, generates descriptive EDA artifacts, and trains the selected
customer segmentation model. Sales analysis is descriptive; customer
segmentation is the machine-learning component.

## Run the complete pipeline

From the project root:

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline.py
```

The stages run in dependency order:

1. Validate the raw dataset.
2. Generate descriptive EDA reports and charts.
3. Preprocess validated data.
4. Build customer and sales feature tables.
5. Create deterministic train/validation/test splits.
6. Retrain PCA-2 plus KMeans-2 on all customer features and save the model
   bundle and cluster profile.

Each run writes a timestamped log to `logs/` and also streams the same messages
to the console. Model comparison remains available as a standalone research
script and is not part of the end-to-end training pipeline.

The promoted model is stored in `models/segmentation/best_model/`:
`best_model.joblib` contains the fitted preprocessor, estimator, and metadata;
`best_metrics.json` contains the evaluation metrics. A newly trained model is
promoted only when its silhouette score improves, with Davies-Bouldin used as
the tie-breaker.

Configuration files can be overridden when needed:

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline.py `
  --data-config configs/data_config.yaml `
  --paths-config configs/paths.yaml `
  --preprocessing-config configs/preprocessing_config.yaml
```

The orchestration code lives in
`src/customer_sales_analytics/pipelines/main_pipeline.py`. Domain logic stays
grouped under `data`, `preprocessing`, `features`, `eda`, and `segmentation`;
the files in `scripts/` are thin command-line stage entrypoints.
