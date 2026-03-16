"""Data loading, feature preparation, and spatial train/test splitting."""

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from models.config import (
    LEAKAGE_FEATURES,
    TARGET,
    DROP_COLUMNS,
    CONTINUOUS_FEATURES,
    BINARY_FEATURES,
    ORDINAL_FEATURES,
    ModelConfig,
)


def load_dataset(path: str) -> pd.DataFrame:
    """Load the feature-engineered CSV and validate expected columns."""
    df = pd.read_csv(path)

    expected = {TARGET} | set(LEAKAGE_FEATURES) | set(DROP_COLUMNS)
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build X and y, excluding leakage features, target, and text columns."""
    exclude = set(LEAKAGE_FEATURES) | {TARGET} | set(DROP_COLUMNS)
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].copy()
    y = df[TARGET].copy()
    return X, y


def spatial_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    df: pd.DataFrame,
    config: ModelConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split using GroupShuffleSplit on agroecological_zone.

    Ensures observations from the same zone do not leak across train/test.
    Returns (X_train, X_test, y_train, y_test).
    """
    if config is None:
        config = ModelConfig()

    groups = df["agroecological_zone"]
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    train_idx, test_idx = next(gss.split(X, y, groups))

    return (
        X.iloc[train_idx],
        X.iloc[test_idx],
        y.iloc[train_idx],
        y.iloc[test_idx],
    )
