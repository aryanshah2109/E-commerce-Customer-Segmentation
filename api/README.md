# Customer Sales Analytics API

This package exposes the promoted customer-segmentation model and a complete retraining operation through FastAPI. The API does not implement sales analytics endpoints; sales reporting remains a batch/dashboard concern.

## Run locally

From the repository root, install dependencies and start Uvicorn:

```powershell
uv sync
uv run uvicorn api.main:app --reload --port 8000
```

The service listens on `http://127.0.0.1:8000` by default. Settings can be overridden with `APP_`-prefixed environment variables, for example:

```powershell
$env:APP_API_HOST = "0.0.0.0"
$env:APP_API_PORT = "8000"
$env:APP_PIPELINE_TIMEOUT_SECONDS = "7200"
```

Interactive OpenAPI documentation is available at `/docs`.

## API surface

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/health` | Liveness and promoted-model readiness |
| `POST` | `/api/v1/predict` | Pure inference for one or more engineered customers |
| `POST` | `/api/v1/pipeline/run` | Run every batch stage and return training metrics |

`/predict` never fits or retrains a transformer or estimator. It loads the promoted `best_model.joblib` bundle, calls the fitted preprocessing object's `transform`, then calls the estimator's `predict`.

`/pipeline/run` invokes `customer_sales_analytics.pipelines.main_pipeline.run_pipeline` in a worker thread. It does not duplicate stage orchestration or shell out to the wrapper script. After a successful run, the model registry is invalidated/reloaded when the promoted model file changes.

## Prediction request

The request contains the same engineered customer fields used by the segmentation preprocessing module. `customer_id` is optional and is passed through only for response traceability.

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/predict `
  -H "Content-Type: application/json" `
  -d '{
    "customers": [{
      "customer_id": "CUST-EXAMPLE",
      "recency_days": 60.82,
      "frequency": 3.86,
      "monetary_value": 2537.29,
      "avg_order_value": 746.61,
      "order_value_std": 817.27,
      "product_diversity": 3.86,
      "purchase_velocity": 0.0169,
      "return_rate": 0.0414,
      "coupon_usage_rate": 0.1489,
      "preferred_category": "Electronics",
      "preferred_payment_method": "Credit Card"
    }]
  }'
```

Example response:

```json
{
  "predictions": [
    {
      "customer_id": "CUST-EXAMPLE",
      "cluster": 0,
      "cluster_size": 35311,
      "cluster_profile": {
        "cluster": 0.0,
        "cluster_size": 35311.0,
        "recency_days": 60.8233411685,
        "frequency": 3.8581745065,
        "monetary_value": 2537.287771516,
        "preferred_category": "Electronics",
        "preferred_payment_method": "Credit Card"
      }
    }
  ],
  "model_trained_at_utc": "2026-09-27T13:15:14.924713+00:00"
}
```

The profile object contains the full matching row from `cluster_profile.csv`; the shortened object above is illustrative.

## Full pipeline request

An empty JSON object runs the complete pipeline with the default YAML files:

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/pipeline/run `
  -H "Content-Type: application/json" `
  -d '{}'
```

Example successful response:

```json
{
  "status": "success",
  "failed_stage": null,
  "duration_seconds": 42.18,
  "log_path": "D:\\College\\Sem 7\\AML\\Project\\logs\\pipeline_20260927_131400_123456.log",
  "segmentation_metrics": {
    "model_name": "PCA-2 + KMeans-2",
    "silhouette_score": 0.3875960162,
    "davies_bouldin_score": 0.9757804067,
    "cluster_sizes": {"0": 35311, "1": 34417},
    "promoted_to_best_model": true,
    "training_rows": 69728,
    "model_path": "models\\segmentation\\candidates\\selected_segmentation_bundle.joblib",
    "cluster_profile_path": "artifacts\\reports\\segmentation_cluster_profile.csv"
  }
}
```

Configuration-path overrides are optional and always still execute all stages:

```json
{
  "data_config_path": "configs/data_config.yaml",
  "paths_config_path": "configs/paths.yaml",
  "preprocessing_config_path": "configs/preprocessing_config.yaml"
}
```

A failed batch stage returns HTTP 200 with `status: "failed"`, the failed stage name, a log path, and empty segmentation metrics. A timeout or API-layer configuration error is returned as an HTTP error.

## Health behavior before the first run

The API starts even if `models/segmentation/best_model/best_model.joblib` does not exist. Startup logs a warning and does not crash. In that state:

```json
{
  "status": "degraded",
  "model_loaded": false,
  "model_trained_at_utc": null
}
```

After a successful pipeline run, the next health or prediction request reloads the promoted artifact automatically.

## Package structure

```text
api/
├── main.py                    # app factory, lifespan, middleware, routers
├── config.py                  # BaseSettings and YAML-derived artifact paths
├── schema/
│   ├── common.py              # health and error responses
│   ├── prediction.py          # engineered-feature prediction schemas
│   └── pipeline.py            # full-run request and metrics schemas
├── services/
│   ├── model_registry.py      # cached joblib/profile/metrics loader
│   ├── prediction_service.py  # fitted transform + predict only
│   └── pipeline_service.py    # threaded full-pipeline orchestration
└── routers/
    ├── health.py
    ├── prediction.py
    └── pipeline.py
```

Paths are resolved from `configs/paths.yaml`; feature column order is imported from `customer_sales_analytics.segmentation.preprocessing`. The API therefore does not maintain a duplicate set of model paths or feature-name rules.
