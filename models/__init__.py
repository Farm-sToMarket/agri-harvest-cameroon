"""ML module for Cameroon agricultural yield prediction."""

from models.config import ModelConfig

__all__ = ["YieldModelTrainer", "YieldPredictor", "ModelConfig"]

_LAZY_IMPORTS = {
    "YieldModelTrainer": ("models.trainer", "YieldModelTrainer"),
    "YieldPredictor": ("models.predict", "YieldPredictor"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module 'models' has no attribute {name!r}")
