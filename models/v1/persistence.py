"""Multi-format model persistence.

- LightGBM  : native .txt (smallest, fastest reload)
- XGBoost   : native .json
- PyTorch   : state_dict .pt + architecture config
- Generic   : joblib fallback
- Metadata  : JSON sidecar for all formats
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

DEFAULT_MODEL_DIR = Path("data/models")


def _meta_path(model_path: Path) -> Path:
    return model_path.with_name(model_path.stem + "_metadata.json")


def _write_metadata(meta_path: Path, metadata: dict) -> None:
    metadata = {
        **metadata,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def _read_metadata(meta_path: Path) -> dict:
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}


# ── LightGBM ───────────────────────────────────────────────────────────────


def save_lightgbm(model, metadata: dict, path: Path | None = None, name: str = "best_lgb") -> Path:
    """Save LightGBM Booster in native text format."""
    save_dir = Path(path) if path else DEFAULT_MODEL_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / f"{name}.txt"
    model.save_model(str(model_path))
    _write_metadata(_meta_path(model_path), {**metadata, "format": "lightgbm_txt"})
    return model_path


def load_lightgbm(path: str | Path):
    """Load LightGBM Booster from native text file."""
    import lightgbm as lgb

    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = lgb.Booster(model_file=str(model_path))
    metadata = _read_metadata(_meta_path(model_path))
    return model, metadata


# ── XGBoost ─────────────────────────────────────────────────────────────────


def save_xgboost(model, metadata: dict, path: Path | None = None, name: str = "best_xgb") -> Path:
    """Save XGBRegressor in JSON format."""
    save_dir = Path(path) if path else DEFAULT_MODEL_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / f"{name}.json"
    model.save_model(str(model_path))
    _write_metadata(_meta_path(model_path), {**metadata, "format": "xgboost_json"})
    return model_path


def load_xgboost(path: str | Path):
    """Load XGBRegressor from JSON."""
    import xgboost as xgb

    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = xgb.XGBRegressor()
    model.load_model(str(model_path))
    metadata = _read_metadata(_meta_path(model_path))
    return model, metadata


# ── PyTorch ─────────────────────────────────────────────────────────────────


def save_pytorch(model, scaler, metadata: dict, path: Path | None = None, name: str = "best_nn") -> Path:
    """Save PyTorch state_dict + scaler via joblib."""
    import torch

    save_dir = Path(path) if path else DEFAULT_MODEL_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / f"{name}.pt"
    torch.save(model.state_dict(), model_path)

    scaler_path = save_dir / f"{name}_scaler.joblib"
    joblib.dump(scaler, scaler_path)

    _write_metadata(
        _meta_path(model_path),
        {**metadata, "format": "pytorch_pt", "scaler_file": str(scaler_path)},
    )
    return model_path


def load_pytorch(path: str | Path, model_class=None, input_dim: int | None = None):
    """Load PyTorch model state_dict + scaler.

    Requires either ``model_class`` and ``input_dim`` to reconstruct
    the architecture, or a pre-instantiated model passed as ``model_class``.
    """
    import torch

    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    metadata = _read_metadata(_meta_path(model_path))

    # Reconstruct model
    if model_class is not None and input_dim is not None:
        model = model_class(input_dim)
    else:
        raise ValueError("Provide model_class and input_dim to reconstruct the model")

    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # Load scaler
    scaler_path = model_path.with_name(model_path.stem + "_scaler.joblib")
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    return model, scaler, metadata


# ── Generic (joblib) ────────────────────────────────────────────────────────


def save_generic(obj, metadata: dict, path: Path | None = None, name: str = "model") -> Path:
    """Fallback: save any picklable object with joblib."""
    save_dir = Path(path) if path else DEFAULT_MODEL_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / f"{name}.joblib"
    joblib.dump(obj, model_path)
    _write_metadata(_meta_path(model_path), {**metadata, "format": "joblib"})
    return model_path


def load_generic(path: str | Path):
    """Load any joblib-saved object."""
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    obj = joblib.load(model_path)
    metadata = _read_metadata(_meta_path(model_path))
    return obj, metadata
