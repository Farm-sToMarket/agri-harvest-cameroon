"""Feature importance analysis: LightGBM native + SHAP (sampled for 10M+)."""

import numpy as np
import pandas as pd


def get_lightgbm_importances(
    model,
    feature_names: list[str],
    importance_type: str = "gain",
) -> pd.Series:
    """Extract native LightGBM feature importances.

    Parameters
    ----------
    importance_type : 'gain' (default, more reliable) or 'split'.
    """
    importances = model.feature_importance(importance_type=importance_type)
    return (
        pd.Series(importances, index=feature_names, name="importance")
        .sort_values(ascending=False)
    )


def get_xgboost_importances(
    model,
    importance_type: str = "gain",
) -> pd.Series:
    """Extract XGBoost feature importances from a fitted XGBRegressor."""
    imp = model.get_booster().get_score(importance_type=importance_type)
    return pd.Series(imp, name="importance").sort_values(ascending=False)


def get_shap_values(
    model,
    X_sample: pd.DataFrame,
    max_samples: int = 50_000,
    random_state: int = 42,
):
    """Compute SHAP values using TreeExplainer (fast for LightGBM/XGBoost).

    Subsamples to ``max_samples`` rows for speed on large datasets.
    Returns (explainer, shap_values, X_subsample).
    """
    import shap

    if len(X_sample) > max_samples:
        X_sample = X_sample.sample(max_samples, random_state=random_state)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    return explainer, shap_values, X_sample


def plot_shap_summary(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    max_display: int = 20,
) -> None:
    """Generate SHAP bar plot + beeswarm plot."""
    import shap
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    plt.title(f"Top {max_display} features (SHAP)")
    plt.tight_layout()
    plt.show()

    shap.summary_plot(
        shap_values,
        X_sample,
        max_display=min(15, max_display),
        show=True,
    )
