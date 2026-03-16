"""Orchestrator: full training pipeline from data to saved model."""

from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from models.config import ModelConfig
from models.data_loader import load_dataset, prepare_features, spatial_train_test_split
from models.preprocessing import build_preprocessor
from models.estimators import get_model, get_all_models
from models.evaluator import (
    MetricsResult,
    evaluate,
    evaluate_by_group,
    compare_models,
    print_report,
)
from models.feature_analysis import get_feature_importances, get_permutation_importances
from models.persistence import save_model


class YieldModelTrainer:
    """End-to-end training pipeline for Cameroon yield prediction."""

    def __init__(
        self,
        data_path: str | Path,
        config: ModelConfig | None = None,
    ):
        self.data_path = Path(data_path)
        self.config = config or ModelConfig()

    def run(
        self,
        model_names: list[str] | None = None,
    ) -> pd.DataFrame:
        """Execute the full pipeline.

        Steps:
          1. Load data
          2. Prepare features (exclude leakage)
          3. Spatial train/test split
          4. Build preprocessor
          5. Train each model (Pipeline: preprocessor + estimator)
          6. Evaluate (global + by crop_group + by zone)
          7. Feature importances for the best model
          8. Save the best model
          9. Return comparison DataFrame

        Parameters
        ----------
        model_names : list of model names to train. If None, trains all.

        Returns
        -------
        pd.DataFrame with comparative metrics.
        """
        # 1. Load
        print("Loading dataset...")
        df = load_dataset(self.data_path)
        print(f"  {df.shape[0]:,} rows x {df.shape[1]} columns")

        # 2. Prepare features
        X, y = prepare_features(df)
        print(f"  {X.shape[1]} features after leakage exclusion")

        # 3. Spatial split
        X_train, X_test, y_train, y_test = spatial_train_test_split(
            X, y, df, self.config
        )
        print(
            f"  Train: {len(X_train):,} | Test: {len(X_test):,} "
            f"(spatial split on agroecological_zone)"
        )

        # 4. Preprocessor
        preprocessor = build_preprocessor(X_train)

        # 5 & 6. Train and evaluate
        if model_names is None:
            models = get_all_models(self.config.random_state)
        else:
            models = {n: get_model(n, self.config.random_state) for n in model_names}

        all_results: dict[str, MetricsResult] = {}
        best_rmse = float("inf")
        best_name = None
        best_pipeline = None

        # Recover group info for test set
        test_crop_groups = df.iloc[X_test.index]["crop_group"]
        test_zones = df.iloc[X_test.index]["agroecological_zone"]

        for name, estimator in models.items():
            print(f"\nTraining {name}...")
            pipe = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("model", estimator),
                ]
            )
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            # Global metrics
            metrics = evaluate(y_test, y_pred)
            all_results[name] = metrics

            # Group metrics
            crop_metrics = evaluate_by_group(y_test, y_pred, test_crop_groups)
            zone_metrics = evaluate_by_group(y_test, y_pred, test_zones)

            print_report(name, metrics, crop_metrics)

            if metrics.rmse < best_rmse:
                best_rmse = metrics.rmse
                best_name = name
                best_pipeline = pipe

        # 7. Feature importances for best model
        print(f"\nBest model: {best_name} (RMSE={best_rmse:,.1f})")

        best_estimator = best_pipeline.named_steps["model"]
        fitted_preprocessor = best_pipeline.named_steps["preprocessor"]
        feature_names_out = list(fitted_preprocessor.get_feature_names_out())

        tree_importances = get_feature_importances(best_estimator, feature_names_out)
        if tree_importances is not None:
            print("\nTop 15 feature importances (tree-based):")
            print(tree_importances.head(15).to_string())

        # 8. Save best model
        best_metrics = all_results[best_name]
        metadata = {
            "model_name": best_name,
            "feature_names": list(X_train.columns),
            "feature_names_out": feature_names_out,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "metrics": {
                "rmse": best_metrics.rmse,
                "mae": best_metrics.mae,
                "r2": best_metrics.r2,
                "mape": best_metrics.mape,
            },
            "config": {
                "test_size": self.config.test_size,
                "random_state": self.config.random_state,
                "cv_folds": self.config.cv_folds,
            },
        }
        model_path = save_model(
            best_estimator,
            fitted_preprocessor,
            metadata,
            model_name="best_model",
        )
        print(f"Best model saved to {model_path}")

        # 9. Comparison table
        comparison = compare_models(all_results)
        print(f"\n{'='*60}")
        print("Model comparison:")
        print(comparison.to_string())
        print()

        return comparison
