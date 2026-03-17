"""Evaluation metrics: RMSE, MAE, R-squared, MAPE — same API as v0."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


@dataclass
class MetricsResult:
    """Container for regression metrics."""

    rmse: float
    mae: float
    r2: float
    mape: float
    n_samples: int


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-3) -> float:
    """MAPE guarded against near-zero actuals."""
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> MetricsResult:
    """Compute global regression metrics."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return MetricsResult(
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        mae=float(mean_absolute_error(y_true, y_pred)),
        r2=float(r2_score(y_true, y_pred)),
        mape=_safe_mape(y_true, y_pred),
        n_samples=len(y_true),
    )


def evaluate_by_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: pd.Series,
) -> dict[str, MetricsResult]:
    """Compute metrics per group (crop_group, zone, etc.)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    groups_arr = np.asarray(groups)

    results = {}
    for g in np.unique(groups_arr):
        mask = groups_arr == g
        if mask.sum() > 0:
            results[str(g)] = evaluate(y_true[mask], y_pred[mask])
    return results


def compare_models(results: dict[str, MetricsResult]) -> pd.DataFrame:
    """Build a comparison table sorted by RMSE."""
    rows = [
        {
            "model": name,
            "RMSE": m.rmse,
            "MAE": m.mae,
            "R2": m.r2,
            "MAPE": m.mape,
            "n_samples": m.n_samples,
        }
        for name, m in results.items()
    ]
    return pd.DataFrame(rows).set_index("model").sort_values("RMSE")


def print_report(
    model_name: str,
    metrics: MetricsResult,
    group_metrics: dict[str, MetricsResult] | None = None,
) -> None:
    """Print a formatted evaluation report."""
    print(f"\n{'=' * 60}")
    print(f"  Model: {model_name}")
    print(f"{'=' * 60}")
    print(f"  RMSE  : {metrics.rmse:>12,.4f}")
    print(f"  MAE   : {metrics.mae:>12,.4f}")
    print(f"  R2    : {metrics.r2:>12.4f}")
    print(f"  MAPE  : {metrics.mape:>12.2%}")
    print(f"  N     : {metrics.n_samples:>12,}")

    if group_metrics:
        print(f"\n  {'Group':<35} {'RMSE':>10} {'MAE':>10} {'R2':>8} {'N':>7}")
        print(f"  {'-' * 70}")
        for group, gm in sorted(group_metrics.items()):
            print(
                f"  {group:<35} {gm.rmse:>10,.4f} {gm.mae:>10,.4f} "
                f"{gm.r2:>8.4f} {gm.n_samples:>7,}"
            )
    print()
