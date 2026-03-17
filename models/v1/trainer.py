"""Orchestrator: train LightGBM, XGBoost, and/or PyTorch on 10M+ rows.

Memory-conscious: triggers ``gc.collect()`` between models and uses
LightGBM's native Dataset (no full copy in RAM).
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd

from models.v1.config import ModelConfig, STRATIFY_COLUMN
from models.v1.data_loader import (
    load_dataset,
    detect_target,
    optimize_dtypes,
    prepare_features,
    stratified_train_test_split,
)
from models.v1.preprocessing import prepare_for_torch, handle_missing, convert_booleans
from models.v1.estimators import (
    YieldNet,
    build_lightgbm_params,
    build_xgboost_params,
)
from models.v1.evaluator import (
    MetricsResult,
    evaluate,
    evaluate_by_group,
    compare_models,
    print_report,
)
from models.v1.feature_analysis import get_lightgbm_importances
from models.v1.persistence import save_lightgbm, save_xgboost, save_pytorch


ALL_MODELS = ("lightgbm", "xgboost", "yieldnet")


class YieldModelTrainer:
    """End-to-end training pipeline scaled for 10M+ rows.

    Usage::

        trainer = YieldModelTrainer("data/features.parquet")
        comparison = trainer.run()                     # all 3 models
        comparison = trainer.run(["lightgbm"])         # LightGBM only
        comparison = trainer.run(["lightgbm"], optimize=True)  # Optuna first
    """

    def __init__(
        self,
        data_path: str | Path,
        config: ModelConfig | None = None,
    ):
        self.data_path = Path(data_path)
        self.config = config or ModelConfig()

    # ── Public API ──────────────────────────────────────────────────────────

    def run(
        self,
        model_names: list[str] | None = None,
        optimize: bool = False,
    ) -> pd.DataFrame:
        """Execute the full pipeline.

        1. Load data (Polars)
        2. Optimize dtypes
        3. Prepare features (leakage guard)
        4. Stratified split
        5. (Optional) Optuna hyperparameter tuning
        6. Train requested models
        7. Evaluate each (global + by group)
        8. Feature importances for LightGBM
        9. Save best model (or all if config.save_all_models)
        10. Return comparison DataFrame
        """
        model_names = model_names or list(ALL_MODELS)
        for name in model_names:
            if name not in ALL_MODELS:
                raise ValueError(
                    f"Unknown model '{name}'. Available: {list(ALL_MODELS)}"
                )

        # 1-2. Load + optimize
        print("Loading dataset...")
        df = load_dataset(self.data_path)
        df = optimize_dtypes(df)
        target_col = detect_target(df, self.config)
        print(f"  {df.shape[0]:,} rows x {df.shape[1]} columns (target: {target_col})")

        # 3. Prepare features
        X, y = prepare_features(df, self.config)
        X = convert_booleans(X)
        print(f"  {X.shape[1]} features after leakage exclusion")

        # 4. Split
        X_train, X_test, y_train, y_test = stratified_train_test_split(
            X, y, df, self.config
        )
        print(
            f"  Train: {len(X_train):,} | Test: {len(X_test):,} "
            f"(stratified on {STRATIFY_COLUMN})"
        )

        # Recover group info for evaluation (use .loc for index-aligned access)
        test_groups = {}
        for col in [STRATIFY_COLUMN, "crop_group", "crop_type", "crop_name"]:
            if col in df.columns:
                s = df[col].to_pandas()
                test_groups[col] = s.loc[X_test.index]

        # Handle missing values once (fit on train, reuse on test)
        X_train_clean, fill_values = handle_missing(X_train)
        X_test_clean, _ = handle_missing(X_test, fill_values)

        # 5. Optional Optuna tuning
        best_params: dict[str, dict] = {}
        if optimize:
            from models.v1.tuning import HyperparameterTuner

            tuner = HyperparameterTuner(
                X_train_clean, X_test_clean, y_train, y_test, self.config
            )
            best_params = tuner.optimize_all(
                model_names=model_names,
                n_trials=self.config.optuna_n_trials,
                timeout=self.config.optuna_timeout,
            )

        # 6-7. Train + evaluate
        all_results: dict[str, MetricsResult] = {}
        best_rmse = float("inf")
        best_name = ""

        if "lightgbm" in model_names:
            lgb_params = best_params.get("lightgbm")
            metrics = self._train_lightgbm(
                X_train, X_test, y_train, y_test, test_groups,
                extra_params=lgb_params,
            )
            all_results["lightgbm"] = metrics
            if metrics.rmse < best_rmse:
                best_rmse, best_name = metrics.rmse, "lightgbm"
            gc.collect()

        if "xgboost" in model_names:
            xgb_params = best_params.get("xgboost")
            metrics = self._train_xgboost(
                X_train_clean, X_test_clean, y_train, y_test, test_groups,
                extra_params=xgb_params,
            )
            all_results["xgboost"] = metrics
            if metrics.rmse < best_rmse:
                best_rmse, best_name = metrics.rmse, "xgboost"
            gc.collect()

        if "yieldnet" in model_names:
            nn_params = best_params.get("yieldnet")
            metrics = self._train_yieldnet(
                X_train_clean, X_test_clean, y_train, y_test, test_groups,
                extra_params=nn_params,
            )
            all_results["yieldnet"] = metrics
            if metrics.rmse < best_rmse:
                best_rmse, best_name = metrics.rmse, "yieldnet"
            gc.collect()

        # 10. Comparison
        comparison = compare_models(all_results)
        print(f"\nBest model: {best_name} (RMSE={best_rmse:,.4f})")
        print(f"\n{'=' * 60}")
        print("Model comparison:")
        print(comparison.to_string())
        print()

        return comparison

    # ── Shared evaluation helper ────────────────────────────────────────────

    def _evaluate_model(self, model_name, y_test, y_pred, test_groups):
        """Evaluate a model globally and per-group, print report."""
        metrics = evaluate(y_test, y_pred)

        group_metrics = {}
        for col_name, groups in test_groups.items():
            gm = evaluate_by_group(y_test, y_pred, groups)
            if gm:
                group_metrics[col_name] = gm

        print_report(model_name, metrics, next(iter(group_metrics.values()), None))

        # Print all group breakdowns
        for col_name, gm in group_metrics.items():
            print(f"  By {col_name}:")
            for g, m in sorted(gm.items()):
                print(f"    {g:<30} RMSE={m.rmse:>10,.4f}  R2={m.r2:.4f}")

        return metrics

    # ── Training methods ────────────────────────────────────────────────────

    def _train_lightgbm(self, X_train, X_test, y_train, y_test, test_groups,
                         extra_params=None):
        import lightgbm as lgb

        print("\nTraining LightGBM...")
        params = build_lightgbm_params(self.config.use_gpu)
        if extra_params:
            params.update(extra_params)

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_valid = lgb.Dataset(X_test, y_test, reference=lgb_train)

        self._lgb_model = lgb.train(
            params,
            lgb_train,
            num_boost_round=self.config.lgb_num_boost_round,
            valid_sets=[lgb_train, lgb_valid],
            callbacks=[
                lgb.early_stopping(self.config.lgb_early_stopping),
                lgb.log_evaluation(self.config.lgb_log_period),
            ],
        )

        y_pred = self._lgb_model.predict(X_test)
        metrics = self._evaluate_model("LightGBM", y_test, y_pred, test_groups)

        # Feature importances
        imp = get_lightgbm_importances(self._lgb_model, list(X_train.columns))
        print("\nTop 15 features (gain):")
        print(imp.head(15).to_string())

        # Save
        should_save = self.config.save_all_models or True  # always save lightgbm for now
        if should_save:
            meta = self._build_metadata("lightgbm", metrics, X_train)
            save_lightgbm(self._lgb_model, meta)
            print("  Saved to data/models/best_lgb.txt")

        return metrics

    def _train_xgboost(self, X_train, X_test, y_train, y_test, test_groups,
                        extra_params=None):
        import xgboost as xgb

        print("\nTraining XGBoost...")
        params = build_xgboost_params(self.config.use_gpu)
        if extra_params:
            params.update(extra_params)

        self._xgb_model = xgb.XGBRegressor(
            **params,
            early_stopping_rounds=self.config.xgb_early_stopping,
        )
        self._xgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=100,
        )

        y_pred = self._xgb_model.predict(X_test)
        metrics = self._evaluate_model("XGBoost", y_test, y_pred, test_groups)

        if self.config.save_all_models:
            meta = self._build_metadata("xgboost", metrics, X_train)
            save_xgboost(self._xgb_model, meta)
            print("  Saved to data/models/best_xgb.json")

        return metrics

    def _train_yieldnet(self, X_train, X_test, y_train, y_test, test_groups,
                         extra_params=None):
        import torch

        print("\nTraining YieldNet (PyTorch)...")

        X_tr_scaled, X_te_scaled, scaler = prepare_for_torch(X_train, X_test)
        device = "cuda" if self.config.use_gpu and torch.cuda.is_available() else "cpu"

        # Apply Optuna params if available
        nn_config = None
        lr = self.config.yieldnet_lr
        weight_decay = 1e-5
        batch_size = self.config.yieldnet_batch_size
        if extra_params:
            nn_config = {
                "hidden_layers": extra_params.get("hidden_layers", [512, 256, 128]),
                "dropout": extra_params.get("dropout", 0.3),
            }
            lr = extra_params.get("learning_rate", lr)
            weight_decay = extra_params.get("weight_decay", weight_decay)
            batch_size = extra_params.get("batch_size", batch_size)

        model = YieldNet(X_train.shape[1], config=nn_config).to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.yieldnet_epochs,
        )

        train_tensor = torch.utils.data.TensorDataset(
            torch.tensor(X_tr_scaled, dtype=torch.float32),
            torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32),
        )
        train_loader = torch.utils.data.DataLoader(
            train_tensor,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(device != "cpu"),
        )

        for epoch in range(self.config.yieldnet_epochs):
            model.train()
            epoch_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()
            if epoch % 5 == 0:
                print(f"  Epoch {epoch:>3d} — Loss: {epoch_loss / len(train_loader):.4f}")

        # Evaluate
        model.eval()
        with torch.no_grad():
            X_te_tensor = torch.tensor(X_te_scaled, dtype=torch.float32).to(device)
            y_pred = model(X_te_tensor).cpu().numpy().flatten()

        metrics = self._evaluate_model("YieldNet", y_test, y_pred, test_groups)

        self._nn_model = model
        if self.config.save_all_models:
            meta = self._build_metadata("yieldnet", metrics, X_train)
            meta["input_dim"] = X_train.shape[1]
            save_pytorch(model, scaler, meta)
            print("  Saved to data/models/best_nn.pt")

        return metrics

    def _build_metadata(self, model_name, metrics, X_train):
        return {
            "model_name": model_name,
            "feature_names": list(X_train.columns),
            "n_train": len(X_train),
            "metrics": {
                "rmse": metrics.rmse,
                "mae": metrics.mae,
                "r2": metrics.r2,
                "mape": metrics.mape,
            },
            "config": {
                "test_size": self.config.test_size,
                "random_state": self.config.random_state,
            },
        }
