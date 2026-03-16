"""Registry of 5 regression models for yield prediction."""

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
    StackingRegressor,
)


def _build_random_forest(random_state: int = 42) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=10,
        random_state=random_state,
        n_jobs=-1,
    )


def _build_hist_gradient_boosting(
    random_state: int = 42,
) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=500,
        max_depth=8,
        learning_rate=0.05,
        min_samples_leaf=20,
        random_state=random_state,
    )


_MODEL_REGISTRY: dict[str, callable] = {
    "baseline": lambda rs: DummyRegressor(strategy="mean"),
    "ridge": lambda rs: Ridge(alpha=1.0),
    "random_forest": lambda rs: _build_random_forest(rs),
    "hist_gradient_boosting": lambda rs: _build_hist_gradient_boosting(rs),
    "stacking": lambda rs: StackingRegressor(
        estimators=[
            ("rf", _build_random_forest(rs)),
            ("hgb", _build_hist_gradient_boosting(rs)),
        ],
        final_estimator=Ridge(alpha=1.0),
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
