"""Inference from saved models (LightGBM, XGBoost, or PyTorch)."""

from pathlib import Path

import numpy as np
import pandas as pd

from models.v1.persistence import load_lightgbm, load_xgboost, load_pytorch
from models.v1.estimators import YieldNet


class YieldPredictor:
    """Unified predictor that auto-detects the model format.

    Supports .txt (LightGBM), .json (XGBoost), .pt (PyTorch).
    """

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.model = None
        self.scaler = None
        self.metadata: dict = {}
        self._format: str = ""
        self._load()

    def _load(self) -> None:
        suffix = self.model_path.suffix
        if suffix == ".txt":
            self.model, self.metadata = load_lightgbm(self.model_path)
            self._format = "lightgbm"
        elif suffix == ".json":
            self.model, self.metadata = load_xgboost(self.model_path)
            self._format = "xgboost"
        elif suffix == ".pt":
            # Read sidecar metadata for input_dim
            from models.v1.persistence import _meta_path, _read_metadata

            meta = _read_metadata(_meta_path(self.model_path))
            input_dim = meta.get("input_dim")
            if input_dim is None:
                raise ValueError(
                    "Cannot determine input_dim for PyTorch model. "
                    "Provide it in metadata."
                )
            self.model, self.scaler, self.metadata = load_pytorch(
                self.model_path,
                model_class=YieldNet,
                input_dim=input_dim,
            )
            self._format = "pytorch"
        else:
            raise ValueError(f"Unsupported model format: {suffix}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict yield from a feature DataFrame."""
        if self._format == "pytorch":
            import torch

            X_arr = X.values.astype(np.float32)
            if self.scaler is not None:
                X_arr = self.scaler.transform(X_arr)
            self.model.eval()
            with torch.no_grad():
                tensor = torch.tensor(X_arr, dtype=torch.float32)
                preds = self.model(tensor).cpu().numpy().flatten()
            return preds

        # LightGBM and XGBoost both have .predict(DataFrame)
        return np.asarray(self.model.predict(X))

    def predict_single(self, features: dict) -> float:
        """Predict yield for a single observation."""
        X = pd.DataFrame([features])
        return float(self.predict(X)[0])
