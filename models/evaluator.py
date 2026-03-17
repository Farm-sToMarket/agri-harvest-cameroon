"""Evaluation metrics: RMSE, MAE, R-squared, MAPE by crop/zone."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


@dataclass
class MetricsResult:
    """Container for regression evaluation metrics."""

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
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
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
    """Compute metrics per group (e.g. crop_group or agroecological_zone)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = groups.values if hasattr(groups, "values") else np.asarray(groups)

    results = {}
    for g in np.unique(groups):
        mask = groups == g
        if mask.sum() > 0:
            results[str(g)] = evaluate(y_true[mask], y_pred[mask])
    return results


def compare_models(results: dict[str, MetricsResult]) -> pd.DataFrame:
    """Build a comparison DataFrame from a dict of model_name -> MetricsResult."""
    rows = []
    for name, m in results.items():
        rows.append(
            {
                "model": name,
                "RMSE": m.rmse,
                "MAE": m.mae,
                "R2": m.r2,
                "MAPE": m.mape,
                "n_samples": m.n_samples,
            }
        )
    df = pd.DataFrame(rows).set_index("model")
    return df.sort_values("RMSE")


def print_report(
    model_name: str,
    metrics: MetricsResult,
    group_metrics: dict[str, MetricsResult] | None = None,
) -> None:
    """Print a formatted evaluation report."""
    print(f"\n{'='*60}")
    print(f"  Model: {model_name}")
    print(f"{'='*60}")
    print(f"  RMSE  : {metrics.rmse:>12,.1f} kg/ha")
    print(f"  MAE   : {metrics.mae:>12,.1f} kg/ha")
    print(f"  R2    : {metrics.r2:>12.4f}")
    print(f"  MAPE  : {metrics.mape:>12.2%}")
    print(f"  N     : {metrics.n_samples:>12,}")

    if group_metrics:
        print(f"\n  {'Group':<35} {'RMSE':>10} {'MAE':>10} {'R2':>8} {'N':>7}")
        print(f"  {'-'*70}")
        for group, gm in sorted(group_metrics.items()):
            print(
                f"  {group:<35} {gm.rmse:>10,.1f} {gm.mae:>10,.1f} "
                f"{gm.r2:>8.4f} {gm.n_samples:>7,}"
            )
    print()
