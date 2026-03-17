"""Unit tests for the scaled ML pipeline (models.v1)."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from models.v1.config import (
    LEAKAGE_FEATURES,
    TARGET,
    TARGET_ALT,
    DROP_COLUMNS,
    LIGHTGBM_PARAMS,
    XGBOOST_PARAMS,
    ModelConfig,
)
from models.v1.data_loader import (
    load_dataset,
    detect_target,
    optimize_dtypes,
    prepare_features,
    stratified_train_test_split,
)
from models.v1.preprocessing import prepare_for_torch, handle_missing, convert_booleans
from models.v1.evaluator import evaluate, evaluate_by_group, compare_models, MetricsResult
from models.v1.persistence import save_generic, load_generic

DATA_PATH = Path("data/generated/cameroon_agricultural_features.csv")
SKIP_NO_DATA = not DATA_PATH.exists()


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_mini_df(n: int = 500) -> pl.DataFrame:
    """Create a small synthetic Polars DataFrame for testing."""
    rng = np.random.RandomState(42)
    return pl.DataFrame(
        {
            "latitude": rng.uniform(2, 13, n).tolist(),
            "longitude": rng.uniform(8, 16, n).tolist(),
            "elevation": rng.uniform(0, 4000, n).tolist(),
            "temperature_mean": rng.uniform(15, 35, n).tolist(),
            "precipitation_daily": rng.uniform(0, 30, n).tolist(),
            "ph_water": rng.uniform(4, 8, n).tolist(),
            "fertilizer_nitrogen": rng.uniform(0, 200, n).tolist(),
            "irrigation_applied": rng.randint(0, 2, n).tolist(),
            "agroecological_zone": rng.choice(
                ["highlands", "sahel", "forest"], n
            ).tolist(),
            "crop_name": rng.choice(["maize", "cassava", "cocoa"], n).tolist(),
            "yield_kg_ha": (rng.uniform(500, 20000, n)).tolist(),
            "nue": rng.uniform(10, 80, n).tolist(),
            "wue": rng.uniform(100, 600, n).tolist(),
            "yield_gap_ratio": rng.uniform(0.3, 1.0, n).tolist(),
            "harvest_index": rng.uniform(0.2, 0.6, n).tolist(),
            "biomass_kg_ha": rng.uniform(1000, 40000, n).tolist(),
            "observation_date": ["2023-06-15"] * n,
            "data_source": ["synthetic"] * n,
        }
    )


# ── Config ──────────────────────────────────────────────────────────────────


class TestConfig:
    def test_leakage_isolation(self):
        """Leakage features must not appear in model params."""
        all_param_keys = set(LIGHTGBM_PARAMS.keys()) | set(XGBOOST_PARAMS.keys())
        for feat in LEAKAGE_FEATURES:
            assert feat not in all_param_keys

    def test_model_config_defaults(self):
        cfg = ModelConfig()
        assert cfg.test_size == 0.15
        assert cfg.random_state == 42
        assert cfg.target == TARGET

    def test_model_config_new_fields(self):
        cfg = ModelConfig()
        assert cfg.yieldnet_lr == 0.001
        assert cfg.save_all_models is False
        assert cfg.optuna_n_trials == 50
        assert cfg.optuna_timeout is None

    def test_no_n_jobs_field(self):
        """n_jobs was removed from ModelConfig."""
        assert not hasattr(ModelConfig(), "n_jobs")

    def test_lazy_import_config(self):
        """Importing config should not require torch/lgb/xgb."""
        from models.v1.config import ModelConfig as MC
        assert MC().target == TARGET


# ── DataLoader ──────────────────────────────────────────────────────────────


class TestDataLoader:
    def test_detect_target_new(self):
        df = pl.DataFrame({TARGET: [1.0, 2.0]})
        assert detect_target(df) == TARGET

    def test_detect_target_legacy(self):
        df = pl.DataFrame({TARGET_ALT: [1000.0, 2000.0]})
        assert detect_target(df) == TARGET_ALT

    def test_detect_target_missing_raises(self):
        df = pl.DataFrame({"x": [1]})
        with pytest.raises(ValueError, match="No target column"):
            detect_target(df)

    def test_detect_target_with_config(self):
        """config.target takes priority over auto-detection."""
        df = pl.DataFrame({TARGET: [1.0], TARGET_ALT: [1000.0]})
        cfg = ModelConfig(target=TARGET_ALT)
        assert detect_target(df, cfg) == TARGET_ALT

    def test_optimize_dtypes(self):
        df = pl.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
        df_opt = optimize_dtypes(df)
        assert df_opt["a"].dtype == pl.Float32

    def test_prepare_features_excludes_leakage(self):
        df = _make_mini_df()
        X, y = prepare_features(df)
        for col in LEAKAGE_FEATURES:
            assert col not in X.columns
        assert TARGET not in X.columns
        assert TARGET_ALT not in X.columns

    def test_stratified_split(self):
        df = _make_mini_df()
        X, y = prepare_features(df)
        X_train, X_test, y_train, y_test = stratified_train_test_split(X, y, df)
        assert len(X_train) + len(X_test) == len(X)

    @pytest.mark.skipif(SKIP_NO_DATA, reason="Dataset not available")
    def test_load_real_csv(self):
        df = load_dataset(DATA_PATH)
        assert df.shape[0] > 0


# ── Preprocessing ───────────────────────────────────────────────────────────


class TestPreprocessing:
    def test_prepare_for_torch(self):
        rng = np.random.RandomState(0)
        X_train = pd.DataFrame(rng.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        X_test = pd.DataFrame(rng.randn(20, 5), columns=[f"f{i}" for i in range(5)])
        X_tr, X_te, scaler = prepare_for_torch(X_train, X_test)
        assert X_tr.shape == (100, 5)
        assert X_te.shape == (20, 5)
        assert X_tr.dtype == np.float32

    def test_handle_missing_returns_tuple(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, np.nan]})
        result, fill_values = handle_missing(df)
        assert not result.isna().any().any()
        assert isinstance(fill_values, pd.Series)

    def test_handle_missing_reuse_fill_values(self):
        """Fill values fitted on train should be reused on test (no leakage)."""
        train = pd.DataFrame({"a": [1.0, 3.0, 5.0], "b": [10.0, 20.0, 30.0]})
        test = pd.DataFrame({"a": [np.nan, 2.0], "b": [np.nan, 15.0]})

        _, fill_vals = handle_missing(train)
        test_filled, _ = handle_missing(test, fill_vals)

        # 'a' median from train is 3.0, 'b' median from train is 20.0
        assert test_filled["a"].iloc[0] == pytest.approx(3.0)
        assert test_filled["b"].iloc[0] == pytest.approx(20.0)

    def test_convert_booleans(self):
        df = pd.DataFrame({"flag": [True, False, True]})
        result = convert_booleans(df)
        assert result["flag"].dtype in (np.int64, np.int32, int)


# ── Estimators ──────────────────────────────────────────────────────────────


class TestEstimators:
    def test_yieldnet_forward(self):
        import torch
        from models.v1.estimators import YieldNet

        model = YieldNet(10)
        x = torch.randn(32, 10)
        out = model(x)
        assert out.shape == (32, 1)

    def test_lightgbm_params(self):
        from models.v1.estimators import build_lightgbm_params

        params = build_lightgbm_params(use_gpu=False)
        assert params["objective"] == "regression"
        assert "device" not in params

        params_gpu = build_lightgbm_params(use_gpu=True)
        assert params_gpu["device"] == "gpu"

    def test_xgboost_params(self):
        from models.v1.estimators import build_xgboost_params

        params = build_xgboost_params(use_gpu=False)
        assert params["tree_method"] == "hist"


# ── Evaluator ───────────────────────────────────────────────────────────────


class TestEvaluator:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        m = evaluate(y, y)
        assert m.rmse == pytest.approx(0.0)
        assert m.r2 == pytest.approx(1.0)

    def test_known_error(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 310.0])
        m = evaluate(y_true, y_pred)
        assert m.rmse == pytest.approx(10.0)
        assert m.mae == pytest.approx(10.0)

    def test_safe_mape_near_zero(self):
        """MAPE should not blow up when y_true contains near-zero values."""
        y_true = np.array([0.0, 0.001, 100.0])
        y_pred = np.array([0.1, 0.002, 110.0])
        m = evaluate(y_true, y_pred)
        assert np.isfinite(m.mape)

    def test_evaluate_by_group(self):
        y_true = np.array([1, 2, 3, 4], dtype=float)
        y_pred = np.array([1.1, 1.9, 3.1, 3.9], dtype=float)
        groups = pd.Series(["A", "A", "B", "B"])
        result = evaluate_by_group(y_true, y_pred, groups)
        assert "A" in result and "B" in result
        assert result["A"].n_samples == 2

    def test_compare_models(self):
        results = {
            "m1": MetricsResult(rmse=1.0, mae=0.8, r2=0.95, mape=0.05, n_samples=100),
            "m2": MetricsResult(rmse=2.0, mae=1.5, r2=0.80, mape=0.10, n_samples=100),
        }
        df = compare_models(results)
        assert df.index[0] == "m1"


# ── Persistence ─────────────────────────────────────────────────────────────


class TestPersistence:
    def test_generic_roundtrip(self):
        obj = {"weights": [1, 2, 3], "bias": 0.5}
        meta = {"model_name": "test"}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_generic(obj, meta, path=tmpdir, name="test")
            assert path.exists()
            loaded, loaded_meta = load_generic(path)
            assert loaded == obj
            assert loaded_meta["model_name"] == "test"

    def test_pytorch_roundtrip(self):
        import torch
        from models.v1.estimators import YieldNet
        from models.v1.persistence import save_pytorch, load_pytorch
        from sklearn.preprocessing import StandardScaler

        model = YieldNet(5)
        scaler = StandardScaler()
        scaler.fit(np.random.randn(10, 5))
        meta = {"model_name": "test_nn", "input_dim": 5}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_pytorch(model, scaler, meta, path=tmpdir, name="test")
            loaded_model, loaded_scaler, loaded_meta = load_pytorch(
                path, model_class=YieldNet, input_dim=5
            )
            x = torch.randn(4, 5)
            with torch.no_grad():
                orig_out = model(x)
                loaded_out = loaded_model(x)
            torch.testing.assert_close(orig_out, loaded_out)


# ── Time Series ─────────────────────────────────────────────────────────────


class TestTimeSeries:
    def test_dataset_and_collate(self):
        from models.v1.time_series import CropTimeSeriesDataset, collate_timeseries

        static = np.random.randn(10, 5).astype(np.float32)
        seqs = [np.random.randn(np.random.randint(30, 120), 3).astype(np.float32) for _ in range(10)]
        targets = np.random.randn(10).astype(np.float32)

        ds = CropTimeSeriesDataset(static, seqs, targets)
        assert len(ds) == 10

        batch = [ds[i] for i in range(4)]
        s, sq, t, lengths = collate_timeseries(batch)
        assert s.shape == (4, 5)
        assert sq.shape[0] == 4
        assert t.shape == (4,)
        assert lengths.shape == (4,)

    def test_hybrid_model_forward(self):
        import torch
        from models.v1.time_series import HybridYieldModel

        model = HybridYieldModel(static_dim=10, weather_dim=3)
        static = torch.randn(8, 10)
        seq = torch.randn(8, 60, 3)
        out = model(static, seq)
        assert out.shape == (8, 1)

    def test_hybrid_model_with_lengths(self):
        import torch
        from models.v1.time_series import HybridYieldModel

        model = HybridYieldModel(static_dim=10, weather_dim=3)
        static = torch.randn(8, 10)
        seq = torch.randn(8, 60, 3)
        lengths = torch.tensor([30, 45, 60, 20, 50, 10, 55, 40])
        out = model(static, seq, lengths)
        assert out.shape == (8, 1)

    def test_transformer_model_forward(self):
        import torch
        from models.v1.time_series import TransformerYieldModel

        model = TransformerYieldModel(static_dim=10, seq_dim=3, d_model=32, nhead=4, num_layers=2)
        static = torch.randn(8, 10)
        seq = torch.randn(8, 90, 3)
        out = model(static, seq)
        assert out.shape == (8, 1)

    def test_transformer_with_lengths(self):
        import torch
        from models.v1.time_series import TransformerYieldModel

        model = TransformerYieldModel(static_dim=10, seq_dim=3, d_model=32, nhead=4, num_layers=2)
        static = torch.randn(8, 10)
        seq = torch.randn(8, 90, 3)
        lengths = torch.tensor([30, 45, 60, 20, 50, 10, 55, 40])
        out = model(static, seq, lengths)
        assert out.shape == (8, 1)

    def test_transformer_variable_length(self):
        """Transformer handles different sequence lengths via padding."""
        import torch
        from models.v1.time_series import TransformerYieldModel

        model = TransformerYieldModel(static_dim=5, seq_dim=3, d_model=16, nhead=4, num_layers=1)
        static = torch.randn(4, 5)
        short_seq = torch.randn(4, 30, 3)
        long_seq = torch.randn(4, 150, 3)
        out_short = model(static, short_seq)
        out_long = model(static, long_seq)
        assert out_short.shape == out_long.shape == (4, 1)


# ── Tuner ────────────────────────────────────────────────────────────────────


class TestTuner:
    @pytest.fixture
    def synth_data(self):
        """Small synthetic dataset for fast Optuna tests."""
        rng = np.random.RandomState(42)
        n = 200
        X = pd.DataFrame(rng.randn(n, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.Series(rng.uniform(500, 5000, n))
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_tr, X_te, y_tr, y_te

    def test_optimize_lightgbm(self, synth_data):
        pytest.importorskip("optuna")
        pytest.importorskip("lightgbm")
        from models.v1.tuning import HyperparameterTuner

        X_tr, X_te, y_tr, y_te = synth_data
        cfg = ModelConfig(lgb_num_boost_round=20, lgb_early_stopping=5)
        tuner = HyperparameterTuner(X_tr, X_te, y_tr, y_te, cfg)
        params = tuner.optimize_lightgbm(n_trials=3)
        assert "num_leaves" in params
        assert "learning_rate" in params

    def test_optimize_xgboost(self, synth_data):
        pytest.importorskip("optuna")
        pytest.importorskip("xgboost")
        from models.v1.tuning import HyperparameterTuner

        X_tr, X_te, y_tr, y_te = synth_data
        tuner = HyperparameterTuner(X_tr, X_te, y_tr, y_te, ModelConfig(xgb_early_stopping=5))
        params = tuner.optimize_xgboost(n_trials=3)
        assert "max_depth" in params

    def test_optimize_yieldnet(self, synth_data):
        pytest.importorskip("optuna")
        pytest.importorskip("torch")
        from models.v1.tuning import HyperparameterTuner

        X_tr, X_te, y_tr, y_te = synth_data
        tuner = HyperparameterTuner(
            X_tr, X_te, y_tr, y_te,
            ModelConfig(yieldnet_epochs=3),
        )
        params = tuner.optimize_yieldnet(n_trials=2)
        assert "hidden_layers" in params
        assert "dropout" in params
        assert "learning_rate" in params

    def test_optimize_all(self, synth_data):
        pytest.importorskip("optuna")
        pytest.importorskip("lightgbm")
        pytest.importorskip("xgboost")
        pytest.importorskip("torch")
        from models.v1.tuning import HyperparameterTuner

        X_tr, X_te, y_tr, y_te = synth_data
        tuner = HyperparameterTuner(
            X_tr, X_te, y_tr, y_te,
            ModelConfig(
                lgb_num_boost_round=20,
                lgb_early_stopping=5,
                xgb_early_stopping=5,
                yieldnet_epochs=3,
            ),
        )
        results = tuner.optimize_all(n_trials=2)
        assert "lightgbm" in results
        assert "xgboost" in results
        assert "yieldnet" in results


# ── Trainer (integration, small scale) ──────────────────────────────────────


class TestTrainer:
    @pytest.mark.skipif(SKIP_NO_DATA, reason="Dataset not available")
    def test_run_lightgbm_subsample(self, tmp_path):
        """Train LightGBM on a 1000-row subsample."""
        pytest.importorskip("lightgbm")

        df = pl.read_csv(DATA_PATH).sample(n=1000, seed=42)
        sample_path = tmp_path / "sample.csv"
        df.write_csv(sample_path)

        from models.v1.trainer import YieldModelTrainer

        trainer = YieldModelTrainer(
            data_path=sample_path,
            config=ModelConfig(
                lgb_num_boost_round=50,
                lgb_early_stopping=10,
                lgb_log_period=10,
            ),
        )
        comparison = trainer.run(["lightgbm"])
        assert isinstance(comparison, pd.DataFrame)
        assert "lightgbm" in comparison.index

    @pytest.mark.skipif(SKIP_NO_DATA, reason="Dataset not available")
    def test_run_with_optimize(self, tmp_path):
        """Train LightGBM with Optuna optimization on small subsample."""
        pytest.importorskip("lightgbm")
        pytest.importorskip("optuna")

        df = pl.read_csv(DATA_PATH).sample(n=500, seed=42)
        sample_path = tmp_path / "sample.csv"
        df.write_csv(sample_path)

        from models.v1.trainer import YieldModelTrainer

        trainer = YieldModelTrainer(
            data_path=sample_path,
            config=ModelConfig(
                lgb_num_boost_round=20,
                lgb_early_stopping=5,
                lgb_log_period=0,
                optuna_n_trials=2,
            ),
        )
        comparison = trainer.run(["lightgbm"], optimize=True)
        assert isinstance(comparison, pd.DataFrame)
        assert "lightgbm" in comparison.index
