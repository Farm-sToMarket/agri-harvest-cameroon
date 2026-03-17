"""Polars-based data loading optimized for 10M+ rows.

Supports CSV and Parquet, with automatic dtype optimization and
stratified splitting.
"""

from pathlib import Path

import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split

from models.v1.config import (
    DROP_COLUMNS,
    LEAKAGE_FEATURES,
    STRATIFY_COLUMN,
    TARGET,
    TARGET_ALT,
    ModelConfig,
)


def load_dataset(path: str | Path) -> pl.DataFrame:
    """Load CSV or Parquet using Polars (10M rows in < 10s)."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pl.read_parquet(path)
    if path.suffix == ".csv":
        return pl.read_csv(path, low_memory=True)
    raise ValueError(f"Unsupported format: {path.suffix}. Use .csv or .parquet")


def detect_target(df: pl.DataFrame, config: ModelConfig | None = None) -> str:
    """Auto-detect target column name, respecting config.target if set."""
    config = config or ModelConfig()
    if config.target in df.columns:
        return config.target
    if TARGET in df.columns:
        return TARGET
    if TARGET_ALT in df.columns:
        return TARGET_ALT
    raise ValueError(f"No target column found. Expected '{config.target}', '{TARGET}', or '{TARGET_ALT}'")


def optimize_dtypes(df: pl.DataFrame) -> pl.DataFrame:
    """Reduce memory: Float64 -> Float32, text -> Categorical."""
    exprs = []

    float64_cols = [c for c in df.columns if df[c].dtype == pl.Float64]
    if float64_cols:
        exprs.append(pl.col(float64_cols).cast(pl.Float32))

    _string_type = pl.String if hasattr(pl, "String") else pl.Utf8
    cat_candidates = [STRATIFY_COLUMN, "crop_type", "crop_name", "season", "crop_group"]
    for c in cat_candidates:
        if c in df.columns and df[c].dtype in (pl.Utf8, _string_type):
            exprs.append(pl.col(c).cast(pl.Categorical))

    return df.with_columns(exprs) if exprs else df


def prepare_features(
    df: pl.DataFrame,
    config: ModelConfig | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build X (pandas) and y (pandas), excluding leakage + target + ID/text.

    Returns pandas DataFrames because LightGBM/XGBoost/sklearn expect them.
    """
    config = config or ModelConfig()
    target_col = detect_target(df, config)

    exclude = set(LEAKAGE_FEATURES) | {target_col} | set(DROP_COLUMNS)
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df.select(feature_cols).to_pandas()
    y = df[target_col].to_pandas()

    return X, y


def stratified_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    df: pl.DataFrame,
    config: ModelConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified split on agroecological_zone (better than group split at 10M+)."""
    config = config or ModelConfig()

    stratify = None
    if STRATIFY_COLUMN in df.columns:
        stratify = df[STRATIFY_COLUMN].to_pandas()

    return train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify,
    )
