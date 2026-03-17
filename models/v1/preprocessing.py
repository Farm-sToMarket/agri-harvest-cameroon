"""Preprocessing utilities for 10M+ pipeline.

LightGBM and XGBoost handle raw features natively (no scaling needed).
Scaling is only required for the PyTorch path.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def prepare_for_torch(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Scale features for PyTorch. Returns float32 arrays + fitted scaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(
        X_train.values.astype(np.float32)
    )
    X_test_scaled = scaler.transform(
        X_test.values.astype(np.float32)
    )
    return X_train_scaled, X_test_scaled, scaler


def handle_missing(
    X: pd.DataFrame,
    fill_values: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Fill NaN with column median. Returns (X_filled, fill_values).

    LightGBM handles NaN natively, but XGBoost and PyTorch need this.
    Fit on train (fill_values=None), reuse on test (pass fill_values back).
    """
    numeric = X.select_dtypes(include="number")
    if fill_values is None:
        fill_values = numeric.median()
    if numeric.isna().any().any():
        X = X.copy()
        X[numeric.columns] = numeric.fillna(fill_values)
    return X, fill_values


def convert_booleans(X: pd.DataFrame) -> pd.DataFrame:
    """Convert boolean columns to int for model compatibility."""
    bool_cols = X.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        X = X.copy()
        X[bool_cols] = X[bool_cols].astype(int)
    return X
