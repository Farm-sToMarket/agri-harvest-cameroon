# Architecture

## Overview

Agri-Harvest is organized into four main layers:

```
config/          Settings, database, schemas (Pydantic v2)
models/          ML pipeline (LightGBM, XGBoost, PyTorch, Transformers)
utils/           Constants, date/geo/file helpers
tests/           Unit and integration tests
```

## Module dependency graph

```
config/settings.py
  ← imports utils/constants.py (CAMEROON_BOUNDS, MAIN_CROPS)
  ← used by config/database.py

models/__init__.py          (lazy imports — no sklearn at import time)
  └── models/v1/__init__.py (lazy imports — no torch/lgb/xgb at import time)
      ├── config.py         (ModelConfig dataclass, all hyperparameter dicts)
      ├── data_loader.py    (Polars-based loading, stratified splits)
      ├── estimators.py     (YieldNet, LightGBM/XGBoost param builders)
      ├── time_series.py    (HybridYieldModel, TransformerYieldModel)
      ├── tuning.py         (Optuna HyperparameterTuner)
      ├── evaluator.py      (safe MAPE, per-group metrics)
      └── convert_parquet.py (CSV → Parquet with dtype optimization)
```

## Key design decisions

### Lazy imports
Both `models/__init__.py` and `models/v1/__init__.py` use `__getattr__`-based lazy loading.
This means `import models` or `from models.v1 import ModelConfig` works without
scikit-learn, torch, LightGBM, or XGBoost installed.

### Conditional PyTorch base class
`models/v1/estimators.py` uses a conditional base class pattern:

```python
try:
    import torch.nn as nn
    _BASE = nn.Module
except ImportError:
    _BASE = object

class YieldNet(_BASE): ...
```

This allows the module to be imported without PyTorch while still inheriting
from `nn.Module` when it's available.

### Configuration sync
`TIMESERIES_CONFIG` in `models/v1/config.py` is built from `ModelConfig` defaults
via `_build_timeseries_config()`, ensuring the dict and the dataclass stay in sync.

### Safe MAPE
Both `models/evaluator.py` and `models/v1/evaluator.py` use `_safe_mape()` which
guards against division by near-zero actuals using an epsilon floor.

### Database indexes
MongoDB indexes are created once per connection lifecycle via the `_indexes_created`
flag on `DatabaseConfig`, avoiding redundant index creation on reconnects.
