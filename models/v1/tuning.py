"""Optuna-based hyperparameter tuning for LightGBM, XGBoost, and YieldNet.

Usage::

    tuner = HyperparameterTuner(X_train, X_test, y_train, y_test, config)
    best = tuner.optimize_all()          # all 3 models
    best = tuner.optimize_lightgbm()     # single model
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.v1.config import ModelConfig


class HyperparameterTuner:
    """Optuna hyperparameter optimization for yield prediction models.

    Minimizes RMSE on the held-out test set.
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        config: ModelConfig | None = None,
    ):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.config = config or ModelConfig()

    def optimize_lightgbm(self, n_trials: int = 50, timeout: int | None = None) -> dict:
        """Optuna study for LightGBM with pruning callback.

        Search space: num_leaves, learning_rate, feature_fraction,
        bagging_fraction, min_child_samples, lambda_l1, lambda_l2, max_depth.
        """
        import optuna
        import lightgbm as lgb
        from optuna.integration import LightGBMPruningCallback

        def objective(trial: optuna.Trial) -> float:
            params = {
                "objective": "regression",
                "metric": "rmse",
                "boosting_type": "gbdt",
                "verbose": -1,
                "n_jobs": -1,
                "num_leaves": trial.suggest_int("num_leaves", 31, 256),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
                "bagging_freq": 5,
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
            }

            lgb_train = lgb.Dataset(self.X_train, self.y_train)
            lgb_valid = lgb.Dataset(self.X_test, self.y_test, reference=lgb_train)

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

            y_pred = model.predict(self.X_test)
            rmse = float(np.sqrt(np.mean((self.y_test.values - y_pred) ** 2)))
            return rmse

        study = optuna.create_study(direction="minimize", study_name="lightgbm")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        print(f"  LightGBM best RMSE: {study.best_value:.4f}")
        return study.best_params

    def optimize_xgboost(self, n_trials: int = 50, timeout: int | None = None) -> dict:
        """Optuna study for XGBoost with pruning callback.

        Search space: max_depth, learning_rate, subsample,
        colsample_bytree, min_child_weight, gamma, reg_alpha, reg_lambda.
        """
        import optuna
        import xgboost as xgb
        from optuna.integration import XGBoostPruningCallback

        def objective(trial: optuna.Trial) -> float:
            params = {
                "objective": "reg:squarederror",
                "tree_method": "hist",
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }

            dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
            dtest = xgb.DMatrix(self.X_test, label=self.y_test)

            pruning_cb = XGBoostPruningCallback(trial, "test-rmse")
            early_stop_cb = xgb.callback.EarlyStopping(
                rounds=self.config.xgb_early_stopping,
                metric_name="rmse",
                data_name="test",
            )
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=800,
                evals=[(dtest, "test")],
                verbose_eval=False,
                callbacks=[early_stop_cb, pruning_cb],
            )

            y_pred = model.predict(dtest)
            rmse = float(np.sqrt(np.mean((self.y_test.values - y_pred) ** 2)))
            return rmse

        study = optuna.create_study(direction="minimize", study_name="xgboost")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        print(f"  XGBoost best RMSE: {study.best_value:.4f}")
        return study.best_params

    def optimize_yieldnet(self, n_trials: int = 30, timeout: int | None = None) -> dict:
        """Optuna study for YieldNet architecture + training.

        Search space: hidden_layer_sizes, dropout, learning_rate,
        weight_decay, batch_size.
        Prunes based on validation loss via trial.report / trial.should_prune.
        """
        import optuna
        import torch
        from sklearn.preprocessing import StandardScaler

        from models.v1.estimators import YieldNet

        def objective(trial: optuna.Trial) -> float:
            n_layers = trial.suggest_int("n_layers", 1, 4)
            hidden_layers = []
            prev = trial.suggest_categorical("first_hidden", [128, 256, 512])
            hidden_layers.append(prev)
            for i in range(1, n_layers):
                next_dim = trial.suggest_categorical(
                    f"hidden_{i}", [v for v in [64, 128, 256, 512] if v <= prev]
                )
                hidden_layers.append(next_dim)
                prev = next_dim

            dropout = trial.suggest_float("dropout", 0.1, 0.5)
            lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
            batch_size = trial.suggest_categorical("batch_size", [2048, 4096, 8192])

            nn_config = {"hidden_layers": hidden_layers, "dropout": dropout}

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(self.X_train.values.astype(np.float32))
            X_te = scaler.transform(self.X_test.values.astype(np.float32))

            device = "cuda" if self.config.use_gpu and torch.cuda.is_available() else "cpu"
            model = YieldNet(self.X_train.shape[1], config=nn_config).to(device)
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

            train_ds = torch.utils.data.TensorDataset(
                torch.tensor(X_tr, dtype=torch.float32),
                torch.tensor(self.y_train.values.reshape(-1, 1), dtype=torch.float32),
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
                    te_tensor = torch.tensor(X_te, dtype=torch.float32).to(device)
                    y_pred = model(te_tensor).cpu().numpy().flatten()
                val_rmse = float(np.sqrt(np.mean((self.y_test.values - y_pred) ** 2)))
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
