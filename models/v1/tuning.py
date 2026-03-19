"""Optuna-based hyperparameter tuning for LightGBM, XGBoost, and YieldNet.

Usage::

    tuner = HyperparameterTuner(X_train, y_train, config)
    best = tuner.optimize_all()          # all 3 models
    best = tuner.optimize_lightgbm()     # single model
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config.yaml_loader import load_models_v1
from models.v1.config import ModelConfig

_tuning_cfg = load_models_v1()["tuning"]


class HyperparameterTuner:
    """Optuna hyperparameter optimization for yield prediction models.

    Splits training data internally into tune/val sets to avoid
    leaking information from the held-out test set.
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        config: ModelConfig | None = None,
    ):
        self.config = config or ModelConfig()
        # Internal split: tune on 80%, validate on 20% of training data
        self.X_tune, self.X_val, self.y_tune, self.y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=self.config.random_state,
        )

    def optimize_lightgbm(self, n_trials: int = 50, timeout: int | None = None) -> dict:
        """Optuna study for LightGBM with pruning callback."""
        import optuna
        import lightgbm as lgb
        from optuna.integration import LightGBMPruningCallback

        ss = _tuning_cfg["lightgbm_search_space"]

        def objective(trial: optuna.Trial) -> float:
            params = {
                "objective": "regression",
                "metric": "rmse",
                "boosting_type": "gbdt",
                "verbose": -1,
                "n_jobs": -1,
                "num_leaves": trial.suggest_int("num_leaves", ss["num_leaves"][0], ss["num_leaves"][1]),
                "learning_rate": trial.suggest_float("learning_rate", ss["learning_rate"][0], ss["learning_rate"][1], log=True),
                "feature_fraction": trial.suggest_float("feature_fraction", ss["feature_fraction"][0], ss["feature_fraction"][1]),
                "bagging_fraction": trial.suggest_float("bagging_fraction", ss["bagging_fraction"][0], ss["bagging_fraction"][1]),
                "bagging_freq": 5,
                "min_child_samples": trial.suggest_int("min_child_samples", ss["min_child_samples"][0], ss["min_child_samples"][1]),
                "lambda_l1": trial.suggest_float("lambda_l1", ss["lambda_l1"][0], ss["lambda_l1"][1], log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", ss["lambda_l2"][0], ss["lambda_l2"][1], log=True),
                "max_depth": trial.suggest_int("max_depth", ss["max_depth"][0], ss["max_depth"][1]),
            }

            lgb_train = lgb.Dataset(self.X_tune, self.y_tune)
            lgb_valid = lgb.Dataset(self.X_val, self.y_val, reference=lgb_train)

            pruning_cb = LightGBMPruningCallback(trial, "rmse")
            model = lgb.train(
                params,
                lgb_train,
                num_boost_round=self.config.lgb_num_boost_round,
                valid_sets=[lgb_valid],
                callbacks=[
                    lgb.early_stopping(self.config.lgb_early_stopping, verbose=False),
                    lgb.log_evaluation(period=0),
                    pruning_cb,
                ],
            )

            y_pred = model.predict(self.X_val)
            rmse = float(np.sqrt(np.mean((self.y_val.values - y_pred) ** 2)))
            return rmse

        study = optuna.create_study(direction="minimize", study_name="lightgbm")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        print(f"  LightGBM best RMSE: {study.best_value:.4f}")
        return study.best_params

    def optimize_xgboost(self, n_trials: int = 50, timeout: int | None = None) -> dict:
        """Optuna study for XGBoost with pruning callback."""
        import optuna
        import xgboost as xgb
        from optuna.integration import XGBoostPruningCallback

        ss = _tuning_cfg["xgboost_search_space"]

        def objective(trial: optuna.Trial) -> float:
            params = {
                "objective": "reg:squarederror",
                "tree_method": "hist",
                "max_depth": trial.suggest_int("max_depth", ss["max_depth"][0], ss["max_depth"][1]),
                "learning_rate": trial.suggest_float("learning_rate", ss["learning_rate"][0], ss["learning_rate"][1], log=True),
                "subsample": trial.suggest_float("subsample", ss["subsample"][0], ss["subsample"][1]),
                "colsample_bytree": trial.suggest_float("colsample_bytree", ss["colsample_bytree"][0], ss["colsample_bytree"][1]),
                "min_child_weight": trial.suggest_int("min_child_weight", ss["min_child_weight"][0], ss["min_child_weight"][1]),
                "gamma": trial.suggest_float("gamma", ss["gamma"][0], ss["gamma"][1], log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", ss["reg_alpha"][0], ss["reg_alpha"][1], log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", ss["reg_lambda"][0], ss["reg_lambda"][1], log=True),
            }

            dtrain = xgb.DMatrix(self.X_tune, label=self.y_tune)
            dval = xgb.DMatrix(self.X_val, label=self.y_val)

            pruning_cb = XGBoostPruningCallback(trial, "val-rmse")
            early_stop_cb = xgb.callback.EarlyStopping(
                rounds=self.config.xgb_early_stopping,
                metric_name="rmse",
                data_name="val",
            )
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=800,
                evals=[(dval, "val")],
                verbose_eval=False,
                callbacks=[early_stop_cb, pruning_cb],
            )

            y_pred = model.predict(dval)
            rmse = float(np.sqrt(np.mean((self.y_val.values - y_pred) ** 2)))
            return rmse

        study = optuna.create_study(direction="minimize", study_name="xgboost")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        print(f"  XGBoost best RMSE: {study.best_value:.4f}")
        return study.best_params

    def optimize_yieldnet(self, n_trials: int = 30, timeout: int | None = None) -> dict:
        """Optuna study for YieldNet architecture + training."""
        import optuna
        import torch
        from sklearn.preprocessing import StandardScaler

        from models.v1.estimators import YieldNet

        ss = _tuning_cfg["yieldnet_search_space"]

        def objective(trial: optuna.Trial) -> float:
            n_layers = trial.suggest_int("n_layers", ss["n_layers"][0], ss["n_layers"][1])
            hidden_layers = []
            prev = trial.suggest_categorical("first_hidden", ss["first_hidden"])
            hidden_layers.append(prev)
            for i in range(1, n_layers):
                next_dim = trial.suggest_categorical(
                    f"hidden_{i}", [v for v in [64, 128, 256, 512] if v <= prev]
                )
                hidden_layers.append(next_dim)
                prev = next_dim

            dropout = trial.suggest_float("dropout", ss["dropout"][0], ss["dropout"][1])
            lr = trial.suggest_float("learning_rate", ss["learning_rate"][0], ss["learning_rate"][1], log=True)
            weight_decay = trial.suggest_float("weight_decay", ss["weight_decay"][0], ss["weight_decay"][1], log=True)
            batch_size = trial.suggest_categorical("batch_size", ss["batch_size"])

            nn_config = {"hidden_layers": hidden_layers, "dropout": dropout}

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(self.X_tune.values.astype(np.float32))
            X_va = scaler.transform(self.X_val.values.astype(np.float32))

            device = "cuda" if self.config.use_gpu and torch.cuda.is_available() else "cpu"
            model = YieldNet(self.X_tune.shape[1], config=nn_config).to(device)
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

            train_ds = torch.utils.data.TensorDataset(
                torch.tensor(X_tr, dtype=torch.float32),
                torch.tensor(self.y_tune.values.reshape(-1, 1), dtype=torch.float32),
            )
            loader = torch.utils.data.DataLoader(
                train_ds, batch_size=batch_size, shuffle=True, num_workers=0,
            )

            epochs = min(self.config.yieldnet_epochs, 15)
            for epoch in range(epochs):
                model.train()
                for bx, by in loader:
                    bx, by = bx.to(device), by.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(bx), by)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                # Report validation loss for pruning
                model.eval()
                with torch.no_grad():
                    va_tensor = torch.tensor(X_va, dtype=torch.float32).to(device)
                    y_pred = model(va_tensor).cpu().numpy().flatten()
                val_rmse = float(np.sqrt(np.mean((self.y_val.values - y_pred) ** 2)))
                trial.report(val_rmse, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return val_rmse

        study = optuna.create_study(
            direction="minimize",
            study_name="yieldnet",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
        )
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        print(f"  YieldNet best RMSE: {study.best_value:.4f}")

        # Reconstruct hidden_layers from best params
        bp = study.best_params
        n_layers = bp["n_layers"]
        hidden_layers = [bp["first_hidden"]]
        for i in range(1, n_layers):
            hidden_layers.append(bp[f"hidden_{i}"])

        return {
            "hidden_layers": hidden_layers,
            "dropout": bp["dropout"],
            "learning_rate": bp["learning_rate"],
            "weight_decay": bp["weight_decay"],
            "batch_size": bp["batch_size"],
        }

    def optimize_all(
        self,
        model_names: list[str] | None = None,
        n_trials: int = 50,
        timeout: int | None = None,
    ) -> dict[str, dict]:
        """Run tuners for requested models, return best params per model."""
        model_names = model_names or ["lightgbm", "xgboost", "yieldnet"]
        results: dict[str, dict] = {}

        print("\nOptuna hyperparameter optimization...")

        if "lightgbm" in model_names:
            results["lightgbm"] = self.optimize_lightgbm(n_trials=n_trials, timeout=timeout)

        if "xgboost" in model_names:
            results["xgboost"] = self.optimize_xgboost(n_trials=n_trials, timeout=timeout)

        if "yieldnet" in model_names:
            nn_trials = min(n_trials, 30)
            results["yieldnet"] = self.optimize_yieldnet(n_trials=nn_trials, timeout=timeout)

        return results
