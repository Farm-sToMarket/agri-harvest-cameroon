"""YAML configuration loader with caching."""

from functools import lru_cache
from pathlib import Path

import yaml

_YAML_DIR = Path(__file__).resolve().parent / "yaml"


def _load(filename: str) -> dict:
    path = _YAML_DIR / filename
    with open(path) as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=None)
def load_geography() -> dict:
    return _load("geography.yaml")


@lru_cache(maxsize=None)
def load_agriculture() -> dict:
    return _load("agriculture.yaml")


@lru_cache(maxsize=None)
def load_models_v0() -> dict:
    return _load("models_v0.yaml")


@lru_cache(maxsize=None)
def load_models_v1() -> dict:
    return _load("models_v1.yaml")
