# ML Models

## Overview

The `models/` package provides a multi-model yield prediction pipeline
optimized for Cameroon agricultural data at scale (10M+ rows).

All heavy imports (torch, lightgbm, xgboost, sklearn) are deferred via
lazy `__getattr__` patterns so that configuration-only imports work without
those libraries installed.

## Available models

### LightGBM (recommended for large datasets)

Native API training (not sklearn wrapper) for maximum performance.

```python
from models.v1.estimators import build_lightgbm_params

params = build_lightgbm_params(use_gpu=False)
```

Default hyperparameters are in `models/v1/config.py::LIGHTGBM_PARAMS`.

### XGBoost

Histogram-based tree method with callback-based early stopping.

```python
from models.v1.estimators import build_xgboost_params

params = build_xgboost_params(use_gpu=False)
```

### YieldNet (PyTorch)

Feedforward neural network: `Linear -> ReLU -> BatchNorm -> Dropout -> ... -> Linear(1)`.

```python
from models.v1.estimators import YieldNet

model = YieldNet(input_dim=50, config={"hidden_layers": [512, 256, 128], "dropout": 0.3})
```

### HybridYieldModel (LSTM + Tabular)

Combines an LSTM branch for weather time series with a dense branch for static features.

```python
from models.v1.time_series import HybridYieldModel, train_hybrid_model

model = HybridYieldModel(static_dim=70, weather_dim=5)
```

### TransformerYieldModel

Transformer encoder for weather sequences + dense tabular branch.
Better than LSTM on long-range climate dependencies (e.g., 15-day droughts).

```python
from models.v1.time_series import TransformerYieldModel

model = TransformerYieldModel(static_dim=70)
```

## Data loading

```python
from models.v1.data_loader import load_dataset, prepare_features, stratified_train_test_split

df = load_dataset("data/features.parquet")  # Polars, supports CSV and Parquet
X, y = prepare_features(df)                 # Removes leakage features, IDs, text
X_train, X_test, y_train, y_test = stratified_train_test_split(X, y, df)
```

## Hyperparameter tuning

```python
from models.v1.tuning import HyperparameterTuner

tuner = HyperparameterTuner(X_train, X_test, y_train, y_test)
best = tuner.optimize_all()          # All 3 models
best = tuner.optimize_lightgbm()     # Single model
```

## Evaluation

```python
from models.v1.evaluator import evaluate, evaluate_by_group, print_report

metrics = evaluate(y_true, y_pred)
group_metrics = evaluate_by_group(y_true, y_pred, groups=df["crop_group"])
print_report("LightGBM", metrics, group_metrics)
```

Metrics: RMSE, MAE, R-squared, MAPE (safe against near-zero actuals).

## CSV to Parquet conversion

```bash
python -m models.v1.convert_parquet data/features.csv
```

Optimizes dtypes (Float64 -> Float32, string -> Categorical) and writes zstd-compressed Parquet.
