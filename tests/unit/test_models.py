"""Unit tests for the ML yield prediction module."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.config import (
    LEAKAGE_FEATURES,
    TARGET,
    DROP_COLUMNS,
    CONTINUOUS_FEATURES,
    BINARY_FEATURES,
    ORDINAL_FEATURES,
    ModelConfig,
)
from models.data_loader import load_dataset, prepare_features, spatial_train_test_split
from models.preprocessing import build_preprocessor
from models.estimators import get_model, get_all_models
from models.evaluator import evaluate, evaluate_by_group, compare_models, MetricsResult
from models.persistence import save_model, load_model

DATA_PATH = Path("data/generated/cameroon_agricultural_features.csv")
SKIP_NO_DATA = not DATA_PATH.exists()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class TestConfig:
    def test_leakage_not_in_continuous(self):
        overlap = set(LEAKAGE_FEATURES) & set(CONTINUOUS_FEATURES)
        assert overlap == set(), f"Leakage features in CONTINUOUS: {overlap}"

    def test_leakage_not_in_binary(self):
        overlap = set(LEAKAGE_FEATURES) & set(BINARY_FEATURES)
        assert overlap == set(), f"Leakage features in BINARY: {overlap}"

    def test_leakage_not_in_ordinal(self):
        overlap = set(LEAKAGE_FEATURES) & set(ORDINAL_FEATURES)
        assert overlap == set(), f"Leakage features in ORDINAL: {overlap}"

    def test_target_not_in_features(self):
        all_features = set(CONTINUOUS_FEATURES) | set(BINARY_FEATURES) | set(ORDINAL_FEATURES)
        assert TARGET not in all_features

    def test_model_config_defaults(self):
        cfg = ModelConfig()
        assert cfg.test_size == 0.2
        assert cfg.random_state == 42


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------
class TestDataLoader:
    @pytest.mark.skipif(SKIP_NO_DATA, reason="Dataset not available")
    def test_load_dataset(self):
        df = load_dataset(DATA_PATH)
        assert df.shape[0] > 0
        assert TARGET in df.columns

    @pytest.mark.skipif(SKIP_NO_DATA, reason="Dataset not available")
    def test_prepare_features_excludes_leakage(self):
        df = load_dataset(DATA_PATH)
        X, y = prepare_features(df)
        for col in LEAKAGE_FEATURES:
            assert col not in X.columns, f"Leakage feature {col} still in X"
        assert TARGET not in X.columns

    @pytest.mark.skipif(SKIP_NO_DATA, reason="Dataset not available")
    def test_spatial_split(self):
        df = load_dataset(DATA_PATH)
        X, y = prepare_features(df)
        X_train, X_test, y_train, y_test = spatial_train_test_split(X, y, df)
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------
class TestPreprocessor:
    @pytest.mark.skipif(SKIP_NO_DATA, reason="Dataset not available")
    def test_build_and_transform(self):
        df = load_dataset(DATA_PATH)
        X, y = prepare_features(df)
        preprocessor = build_preprocessor(X)
        X_t = preprocessor.fit_transform(X)
        assert X_t.shape[0] == X.shape[0]
        assert X_t.shape[1] > 0


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------
class TestEstimators:
    @pytest.fixture
    def mini_data(self):
        rng = np.random.RandomState(42)
        n = 200
        X = pd.DataFrame({
            "f1": rng.randn(n),
            "f2": rng.randn(n),
            "f3": rng.randint(0, 2, n).astype(float),
        })
        y = 1000 + 50 * X["f1"] + 30 * X["f2"] + rng.randn(n) * 10
        return X, y

    @pytest.mark.parametrize(
        "name",
        ["baseline", "ridge", "random_forest", "hist_gradient_boosting", "stacking"],
    )
    def test_fit_predict(self, name, mini_data):
        X, y = mini_data
        model = get_model(name)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_get_all_models(self):
        models = get_all_models()
        assert len(models) == 5
        assert "baseline" in models

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            get_model("nonexistent")


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------
class TestEvaluator:
    def test_evaluate_perfect(self):
        y = np.array([100, 200, 300, 400, 500], dtype=float)
        m = evaluate(y, y)
        assert m.rmse == pytest.approx(0.0)
        assert m.mae == pytest.approx(0.0)
        assert m.r2 == pytest.approx(1.0)
        assert m.n_samples == 5

    def test_evaluate_known_values(self):
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 310])
        m = evaluate(y_true, y_pred)
        assert m.rmse > 0
        assert m.mae > 0
        assert m.r2 < 1.0
        assert m.n_samples == 3

    def test_evaluate_by_group(self):
        y_true = np.array([100, 200, 300, 400])
        y_pred = np.array([110, 190, 310, 390])
        groups = pd.Series(["A", "A", "B", "B"])
        result = evaluate_by_group(y_true, y_pred, groups)
        assert "A" in result
        assert "B" in result
        assert result["A"].n_samples == 2
        assert result["B"].n_samples == 2

    def test_compare_models(self):
        results = {
            "m1": MetricsResult(rmse=100, mae=80, r2=0.9, mape=0.1, n_samples=50),
            "m2": MetricsResult(rmse=200, mae=150, r2=0.7, mape=0.2, n_samples=50),
        }
        df = compare_models(results)
        assert df.index[0] == "m1"  # lower RMSE first
        assert "RMSE" in df.columns


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
class TestPersistence:
    def test_save_load_roundtrip(self):
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        model = Ridge(alpha=1.0)
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        y = np.array([10, 20, 30], dtype=float)
        model.fit(X, y)

        scaler = StandardScaler()
        scaler.fit(X)

        metadata = {"model_name": "test_ridge", "feature_names": ["a", "b"]}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_model(model, scaler, metadata, path=tmpdir, model_name="test")
            assert path.exists()

            loaded_model, loaded_preprocessor, loaded_meta = load_model(path)
            preds_orig = model.predict(X)
            X_scaled = loaded_preprocessor.transform(X)
            preds_loaded = loaded_model.predict(X_scaled)
            np.testing.assert_array_almost_equal(
                model.predict(scaler.transform(X)), preds_loaded
            )
            assert loaded_meta["model_name"] == "test_ridge"


# ---------------------------------------------------------------------------
# Trainer (integration, uses subsample)
# ---------------------------------------------------------------------------
class TestTrainer:
    @pytest.mark.skipif(SKIP_NO_DATA, reason="Dataset not available")
    def test_run_single_model(self, tmp_path):
        from models.trainer import YieldModelTrainer
        from models.persistence import DEFAULT_MODEL_DIR

        # Use a subsample for speed
        df = pd.read_csv(DATA_PATH)
        sample_path = tmp_path / "sample.csv"
        df.sample(n=min(1000, len(df)), random_state=42).to_csv(
            sample_path, index=False
        )

        trainer = YieldModelTrainer(
            data_path=sample_path,
            config=ModelConfig(random_state=42),
        )
        comparison = trainer.run(model_names=["ridge"])
        assert isinstance(comparison, pd.DataFrame)
        assert "ridge" in comparison.index
        assert comparison.loc["ridge", "R2"] is not None
