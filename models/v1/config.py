"""Configuration for scaled ML pipeline (10M+ rows).

Supports LightGBM, XGBoost, and PyTorch models with Polars-based loading.
"""

from dataclasses import dataclass

from config.yaml_loader import load_models_v1

_cfg = load_models_v1()

# ── Target 
TARGET = _cfg["target"]
TARGET_ALT = _cfg["target_alt"]

LEAKAGE_FEATURES = _cfg["leakage_features"]

# ── Columns to drop (IDs + raw text already one-hot encoded) 
ID_COLUMNS = _cfg["id_columns"]
TEXT_COLUMNS = _cfg["text_columns"]
DROP_COLUMNS = ID_COLUMNS + TEXT_COLUMNS

STRATIFY_COLUMN = _cfg["stratify_column"]

# ── LightGBM (recommended for 10M+) 
LIGHTGBM_PARAMS = _cfg["lightgbm"]

# ── XGBoost
XGBOOST_PARAMS = _cfg["xgboost"]

# ── PyTorch YieldNet
YIELDNET_CONFIG = _cfg["yieldnet"]

_training = _cfg["training"]


@dataclass
class ModelConfig:
    """Training configuration."""

    target: str = TARGET
    test_size: float = _cfg["test_size"]
    random_state: int = _cfg["random_state"]
    use_gpu: bool = False

    # LightGBM
    lgb_num_boost_round: int = _training["lgb_num_boost_round"]
    lgb_early_stopping: int = _training["lgb_early_stopping"]
    lgb_log_period: int = _training["lgb_log_period"]

    # XGBoost
    xgb_early_stopping: int = _training["xgb_early_stopping"]

    # PyTorch
    yieldnet_epochs: int = _training["yieldnet_epochs"]
    yieldnet_batch_size: int = _training["yieldnet_batch_size"]
    yieldnet_lr: float = _training["yieldnet_lr"]

    # Saving
    save_all_models: bool = _training["save_all_models"]

    # Optuna
    optuna_n_trials: int = _training["optuna_n_trials"]
    optuna_timeout: int | None = _training["optuna_timeout"]

    # Time series
    ts_lstm_hidden: int = _cfg["time_series"]["lstm_hidden"]
    ts_lstm_layers: int = _cfg["time_series"]["lstm_layers"]
    ts_attention_heads: int = _cfg["time_series"]["attention_heads"]
    ts_batch_size: int = _cfg["time_series"]["batch_size"]
    ts_epochs: int = _cfg["time_series"]["epochs"]
    ts_lr: float = _cfg["time_series"]["learning_rate"]
    ts_max_seq_len: int = _cfg["time_series"]["max_sequence_length"]

    # SHAP
    shap_sample_size: int = _training["shap_sample_size"]


# ── Time series (Hybrid LSTM + Tabular) ────────────────────────────────────
_ts_cfg = _cfg["time_series"]


def _build_timeseries_config() -> dict:
    """Build TIMESERIES_CONFIG from ModelConfig defaults so they stay in sync."""
    _defaults = ModelConfig()
    return {
        "weather_features": _ts_cfg["weather_features"],
        "max_sequence_length": _defaults.ts_max_seq_len,
        "lstm_hidden": _defaults.ts_lstm_hidden,
        "lstm_layers": _defaults.ts_lstm_layers,
        "attention_heads": _defaults.ts_attention_heads,
        "batch_size": _defaults.ts_batch_size,
        "epochs": _defaults.ts_epochs,
        "learning_rate": _defaults.ts_lr,
    }


TIMESERIES_CONFIG = _build_timeseries_config()
