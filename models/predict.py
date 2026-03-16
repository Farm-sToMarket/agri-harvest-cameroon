"""Inference from a saved model."""

from pathlib import Path

import numpy as np
import pandas as pd

from models.persistence import load_model


class YieldPredictor:
    """Load a saved model and make predictions."""

    def __init__(self, model_path: str | Path):
        self.model, self.preprocessor, self.metadata = load_model(model_path)
        self.feature_names = self.metadata.get("feature_names", [])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Apply preprocessor and predict on a DataFrame."""
        X_transformed = self.preprocessor.transform(X)
        return self.model.predict(X_transformed)

    def predict_single(self, features: dict) -> float:
        """Predict yield for a single observation given as a dict."""
        X = pd.DataFrame([features])
        predictions = self.predict(X)
        return float(predictions[0])
