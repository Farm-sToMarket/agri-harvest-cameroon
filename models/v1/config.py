"""Configuration for scaled ML pipeline (10M+ rows).

Supports LightGBM, XGBoost, and PyTorch models with Polars-based loading.
"""

from dataclasses import dataclass

# ── Target ──────────────────────────────────────────────────────────────────
TARGET = "yield_tha"
TARGET_ALT = "yield_kg_ha"  # fallback for legacy 50K dataset

# ── Leakage guard ──────────────────────────────────────────────────────────
LEAKAGE_FEATURES = [
    "nue",
    "wue",
    "yield_gap_ratio",
    "harvest_index",
    "biomass_kg_ha",
]

# ── Columns to drop (IDs + raw text already one-hot encoded) ──────────────
ID_COLUMNS = [
    "field_id",
    "observation_date",
    "planting_date",
    "harvest_date",
]

TEXT_COLUMNS = [
    "data_source",
    "agroecological_zone",
    "season",
    "crop_name",
    "crop_type",
    "crop_group",
    "variety",
]

DROP_COLUMNS = ID_COLUMNS + TEXT_COLUMNS

STRATIFY_COLUMN = "agroecological_zone"

# ── LightGBM (recommended for 10M+) ────────────────────────────────────────
LIGHTGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 128,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "n_jobs": -1,
}

# ── XGBoost ─────────────────────────────────────────────────────────────────
XGBOOST_PARAMS = {
    "objective": "reg:squarederror",
    "learning_rate": 0.05,
    "max_depth": 12,
    "subsample": 0.7,
    "colsample_bytree": 0.8,
    "n_estimators": 800,
    "tree_method": "hist",
}

# ── PyTorch YieldNet ────────────────────────────────────────────────────────
YIELDNET_CONFIG = {
    "hidden_layers": [512, 256, 128],
    "dropout": 0.3,
    "batch_size": 8192,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "epochs": 30,
    "num_workers": 4,
}

# ── Time series (Hybrid LSTM + Tabular) ────────────────────────────────────
TIMESERIES_CONFIG = {
    "weather_features": [
        "temperature_min",
        "temperature_max",
        "precipitation_daily",
        "relative_humidity",
        "solar_radiation",
    ],
    "max_sequence_length": 180,
    "lstm_hidden": 128,
    "lstm_layers": 2,
    "attention_heads": 8,
    "batch_size": 4096,
    "epochs": 30,
    "learning_rate": 0.001,
}


@dataclass
class ModelConfig:
    """Training configuration."""

    target: str = TARGET
    test_size: float = 0.15
    random_state: int = 42
    use_gpu: bool = False

    # LightGBM
    lgb_num_boost_round: int = 2000
    lgb_early_stopping: int = 50
    lgb_log_period: int = 100

    # XGBoost
    xgb_early_stopping: int = 50

    # PyTorch
    yieldnet_epochs: int = 30
    yieldnet_batch_size: int = 8192
    yieldnet_lr: float = 0.001

    # Saving
    save_all_models: bool = False

    # Optuna
    optuna_n_trials: int = 50
    optuna_timeout: int | None = None

    # SHAP
    shap_sample_size: int = 50_000
