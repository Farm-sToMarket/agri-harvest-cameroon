"""ML v1 — Scaled pipeline for 10M+ rows.

Models: LightGBM, XGBoost, PyTorch YieldNet, Hybrid LSTM+Tabular, Transformer.

All heavy imports (torch, lightgbm, xgboost) are deferred so that
``from models.v1 import ModelConfig`` works without those libraries.
"""

from models.v1.config import ModelConfig

__all__ = [
    "YieldModelTrainer",
    "YieldPredictor",
    "ModelConfig",
    "HybridYieldModel",
    "TransformerYieldModel",
    "HyperparameterTuner",
]

_LAZY_IMPORTS = {
    "YieldModelTrainer": ("models.v1.trainer", "YieldModelTrainer"),
    "YieldPredictor": ("models.v1.predict", "YieldPredictor"),
    "HybridYieldModel": ("models.v1.time_series", "HybridYieldModel"),
    "TransformerYieldModel": ("models.v1.time_series", "TransformerYieldModel"),
    "HyperparameterTuner": ("models.v1.tuning", "HyperparameterTuner"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module 'models.v1' has no attribute {name!r}")
