"""Model persistence: save/load with joblib and JSON metadata."""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib


DEFAULT_MODEL_DIR = Path("data/models")


def save_model(
    model,
    preprocessor,
    metadata: dict,
    path: str | Path | None = None,
    model_name: str = "best_model",
) -> Path:
    """Save model, preprocessor, and metadata.

    Saves:
      - <path>/<model_name>.joblib  (model + preprocessor bundle)
      - <path>/<model_name>_metadata.json
    """
    save_dir = Path(path) if path else DEFAULT_MODEL_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / f"{model_name}.joblib"
    meta_path = save_dir / f"{model_name}_metadata.json"

    bundle = {
        "model": model,
        "preprocessor": preprocessor,
    }
    joblib.dump(bundle, model_path)

    metadata = {
        **metadata,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "model_file": str(model_path),
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return model_path


def load_model(path: str | Path) -> tuple[object, object, dict]:
    """Load a saved model bundle and its metadata.

    Returns (model, preprocessor, metadata).
    """
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    bundle = joblib.load(model_path)
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]

    meta_path = model_path.with_name(
        model_path.stem + "_metadata.json"
    )
    metadata = {}
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    return model, preprocessor, metadata
