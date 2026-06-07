import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score


class ModelTrainer:
    def __init__(self, model_dir: Path, sport: str, stat_type: str):
        self.model_dir = model_dir / sport
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.sport = sport
        self.stat_type = stat_type
        self.model_path = self.model_dir / f"{stat_type}.json"

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        test_size: float = 0.2,
    ) -> dict:
        if X_test is None or y_test is None:
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        else:
            X_train, y_train = X, y

        params = {
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "seed": 42,
            "n_jobs": -1,
        }

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False,
        )

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Post-hoc directional accuracy: does predicted > trailing_avg match actual > trailing_avg?
        # The "trailing_avg" is the prior expanding mean (what a naive model would predict)
        # We compute it here from X_test features (which already have the rolling features)
        direction_hits = 0
        direction_total = 0
        for idx in range(len(y_test)):
            actual = y_test.iloc[idx]
            pred = y_pred[idx]
            # Direction: did the player exceed their own trailing mean?
            # Use the EWM as a proxy for "projected baseline"
            # We don't have the baseline in the test set, so we use the training mean
            avg_actual = float(y_train.mean())
            actual_dir = 1 if actual > avg_actual else 0
            pred_dir = 1 if pred > avg_actual else 0
            if actual_dir == pred_dir:
                direction_hits += 1
            direction_total += 1

        direction_accuracy = direction_hits / max(direction_total, 1)

        result = {
            "sport": self.sport,
            "stat_type": self.stat_type,
            "mae": round(mae, 4),
            "r2": round(r2, 4),
            "directional_accuracy": round(direction_accuracy, 4),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "feature_count": X.shape[1],
            "train_date": datetime.utcnow().isoformat(),
        }

        model.save_model(self.model_path)

        var_imps = pd.DataFrame({
            "feature": X.columns,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        var_imps.to_csv(self.model_dir / f"{self.stat_type}_importance.csv", index=False)

        return result

    def walk_forward_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 4,
        min_train_size: int = 200,
    ) -> list[dict]:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        results = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            if len(train_idx) < min_train_size:
                continue

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            params = {
                "objective": "reg:squarederror",
                "eval_metric": "mae",
                "learning_rate": 0.05,
                "max_depth": 6,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "seed": 42 + fold,
                "n_jobs": -1,
            }

            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            y_pred = model.predict(X_test)

            results.append({
                "fold": fold,
                "test_start": str(y_test.index[0]),
                "test_end": str(y_test.index[-1]),
                "mae": round(mean_absolute_error(y_test, y_pred), 4),
                "r2": round(r2_score(y_test, y_pred), 4),
                "n_train": len(X_train),
                "n_test": len(X_test),
            })

        return results

    def load(self):
        if self.model_path.exists():
            model = xgb.XGBRegressor()
            model.load_model(self.model_path)
            return model
        return None

    def feature_importance_report(self, X: pd.DataFrame, model) -> pd.DataFrame:
        imp = pd.DataFrame({
            "feature": X.columns,
            "gain": model.feature_importances_,
        }).sort_values("gain", ascending=False)
        imp["cumulative_gain"] = imp["gain"].cumsum()
        imp["pct"] = imp["gain"] / imp["gain"].sum() * 100
        return imp
