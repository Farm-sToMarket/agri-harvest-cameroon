"""Feature importance analysis: tree-based and permutation importances."""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def get_feature_importances(
    model,
    feature_names: list[str],
) -> pd.Series | None:
    """Extract tree-based feature importances if available.

    Works with RandomForest, HistGradientBoosting, and other tree models.
    Returns None for models without feature_importances_.
    """
    if not hasattr(model, "feature_importances_"):
        return None

    importances = pd.Series(
        model.feature_importances_,
        index=feature_names,
        name="importance",
    )
    return importances.sort_values(ascending=False)


def get_permutation_importances(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute permutation importances (model-agnostic, more reliable)."""
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    df = pd.DataFrame(
        {
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        },
        index=feature_names,
    )
    return df.sort_values("importance_mean", ascending=False)
