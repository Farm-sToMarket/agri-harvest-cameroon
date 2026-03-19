"""Feature groups, leakage guard, and model configuration."""

from dataclasses import dataclass, field

from config.yaml_loader import load_models_v0

_cfg = load_models_v0()

TARGET = _cfg["target"]

LEAKAGE_FEATURES = _cfg["leakage_features"]

DROP_COLUMNS = _cfg["drop_columns"]

CONTINUOUS_FEATURES = _cfg["continuous_features"]

BINARY_FEATURES = _cfg["binary_features"]

ORDINAL_FEATURES = _cfg["ordinal_features"]


@dataclass
class ModelConfig:
    """Configuration for model training."""

    test_size: float = _cfg["test_size"]
    random_state: int = _cfg["random_state"]
    n_jobs: int = _cfg["n_jobs"]
