"""Registry of 5 regression models for yield prediction."""

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
    StackingRegressor,
)

from config.yaml_loader import load_models_v0

_est_cfg = load_models_v0()["estimators"]


def _build_random_forest(random_state: int = 42) -> RandomForestRegressor:
    rf = _est_cfg["random_forest"]
    return RandomForestRegressor(
        n_estimators=rf["n_estimators"],
        max_depth=rf["max_depth"],
        min_samples_leaf=rf["min_samples_leaf"],
        random_state=random_state,
        n_jobs=-1,
    )


def _build_hist_gradient_boosting(
    random_state: int = 42,
) -> HistGradientBoostingRegressor:
    hgb = _est_cfg["hist_gradient_boosting"]
    return HistGradientBoostingRegressor(
        max_iter=hgb["max_iter"],
        max_depth=hgb["max_depth"],
        learning_rate=hgb["learning_rate"],
        min_samples_leaf=hgb["min_samples_leaf"],
        random_state=random_state,
    )


_MODEL_REGISTRY: dict[str, object] = {
    "baseline": lambda rs: DummyRegressor(strategy="mean"),
    "ridge": lambda rs: Ridge(alpha=_est_cfg["ridge"]["alpha"]),
    "random_forest": lambda rs: _build_random_forest(rs),
    "hist_gradient_boosting": lambda rs: _build_hist_gradient_boosting(rs),
    "stacking": lambda rs: StackingRegressor(
        estimators=[
            ("rf", _build_random_forest(rs)),
            ("hgb", _build_hist_gradient_boosting(rs)),
        ],
        final_estimator=Ridge(alpha=_est_cfg["ridge"]["alpha"]),
        n_jobs=-1,
    ),
}


def get_model(name: str, random_state: int = 42):
    """Return a fresh estimator instance by name."""
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[name](random_state)


def get_all_models(random_state: int = 42) -> dict[str, object]:
    """Return a dict of all available model instances."""
    return {name: factory(random_state) for name, factory in _MODEL_REGISTRY.items()}
