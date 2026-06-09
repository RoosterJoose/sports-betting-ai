#!/usr/bin/env python3
"""
Train College Football models for three target types:

  1. spread_margin  (regression) — predict margin of victory
  2. total_points   (regression) — predict combined game score
  3. win           (classification) — predict moneyline (P(win))

Uses walk-forward validation (TimeSeriesSplit), saves models, feature
importance, calibration tables, and metadata to models/cfb/.
"""
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, brier_score_loss, roc_auc_score
from xgboost import XGBRegressor, XGBClassifier

warnings.filterwarnings("ignore")

from src.config.settings import Settings
from src.data.cfb import CFBDataSource
from src.features.cfb import CFBFeatureEngineer, FEATURE_COLS, CFB_STATS

MODEL_DIR = Path("models/cfb")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Targets to train
TARGETS = {
    "spread_margin": {
        "type": "regression",
        "column": "spread_margin",
        "description": "Margin of victory (points_for - points_against)",
        "mae_baseline": 14.0,  # typical CFB spread MAE for naive forecast
    },
    "total_points": {
        "type": "regression",
        "column": "total_points",
        "description": "Total combined game score",
        "mae_baseline": 12.0,
    },
    "win": {
        "type": "classification",
        "column": "win",
        "description": "Moneyline (home/away win probability)",
        "base_rate": None,  # computed from data
    },
}


def fetch_and_featurize() -> pd.DataFrame:
    """Fetch CFB data and build features. Returns featured DataFrame."""
    print("=" * 70)
    print("  CFB MODEL TRAINING")
    print("=" * 70)

    cfg = Settings().load_sport_config("cfb")
    if cfg is None:
        print("  ERROR: config/cfb.toml not found")
        return pd.DataFrame()

    ds = CFBDataSource()
    fe = CFBFeatureEngineer(cfg)

    print(f"\nFetching data (up to {cfg.season_lookback} seasons)...")
    raw = ds.fetch_player_game_logs(
        [str(datetime.now().year - i) for i in range(cfg.season_lookback)]
    )
    if raw.empty:
        print("  No data fetched — check CFBD_API_KEY in .env")
        return pd.DataFrame()

    print(f"  Raw: {len(raw)} team-game rows, {raw['team'].nunique()} teams")

    print("\nBuilding features...")
    featured = fe.build_features(raw)
    if featured.empty:
        print("  Feature engineering failed")
        return pd.DataFrame()

    available = [c for c in FEATURE_COLS if c in featured.columns]
    print(f"  Features: {len(available)} available from FEATURE_COLS")
    print(f"  Total columns in output: {len(featured.columns)}")

    return featured


def train_regression(
    df: pd.DataFrame, target: str, target_info: dict
) -> None:
    """Train an XGBRegressor for a continuous target."""
    print(f"\n{'-' * 66}")
    print(f"  Training: {target} ({target_info['description']})")
    print(f"{'-' * 66}")

    # Identify feature columns (all numeric except targets and metadata)
    drop_cols = {target, "player_id", "team", "opponent", "game_date",
                 "season", "week", "season_type", "win", "spread_margin",
                 "total_points", "points_for", "points_against"}
    feature_cols = [c for c in df.columns
                    if c not in drop_cols
                    and df[c].dtype in ("float64", "int64", "float32", "int32", "int8", "int16", "bool")]

    # Drop rows with missing target
    train_df = df.dropna(subset=[target]).copy()
    if train_df.empty:
        print(f"  No valid rows for {target}")
        return

    X = train_df[feature_cols].fillna(0)
    y = train_df[target].values

    print(f"  Samples: {len(X)}, Features: {len(feature_cols)}")
    print(f"  Target range: [{y.min():.1f}, {y.max():.1f}], mean={y.mean():.1f}, std={y.std():.1f}")

    # Walk-forward validation
    test_size = min(max(200, int(len(X) * 0.1)), int(len(X) * 0.3))
    tscv = TimeSeriesSplit(n_splits=5, test_size=test_size)
    cv_scores = []
    oof_preds = np.full(len(X), np.nan)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=3,
            random_state=42 + fold,
            eval_metric="mae",
            early_stopping_rounds=30,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        preds = model.predict(X_test)
        oof_preds[test_idx] = preds

        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        # Directional accuracy: does pred direction (above/below mean) match actual?
        baseline = y_train.mean()
        dir_acc = ((preds > baseline) == (y_test > baseline)).mean()

        cv_scores.append({
            "fold": fold,
            "mae": round(mae, 2),
            "r2": round(r2, 4),
            "directional_acc": round(dir_acc, 4),
            "n_train": len(X_train),
            "n_test": len(X_test),
        })
        print(f"  Fold {fold}: MAE={mae:.2f}, R²={r2:.3f}, DirAcc={dir_acc:.1%}")

    # Overall OOF metrics
    oof_valid = ~np.isnan(oof_preds)
    oof_mae = mean_absolute_error(y[oof_valid], oof_preds[oof_valid])
    oof_r2 = r2_score(y[oof_valid], oof_preds[oof_valid])
    baseline_mae = target_info.get("mae_baseline", 14.0)
    improvement = (baseline_mae - oof_mae) / baseline_mae * 100

    print(f"\n  OOF MAE: {oof_mae:.2f} (baseline={baseline_mae:.1f}, {improvement:+.0f}%)")
    print(f"  OOF R²:  {oof_r2:.3f}")

    # Retrain on full data
    print("\n  Retraining on full dataset...")
    final_model = XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=3,
        random_state=42,
        eval_metric="mae",
    )
    final_model.fit(X, y)

    # Feature importance
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": final_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\n  Top 15 features:")
    for _, row in importance.head(15).iterrows():
        print(f"    {row['feature']:40s} {row['importance']:.4f}")

    # Save model
    model_file = MODEL_DIR / f"{target}.json"
    final_model.save_model(str(model_file))
    print(f"\n  Model saved: {model_file}")

    # Save metadata
    meta = {
        "target": target,
        "model_type": "XGBRegressor",
        "train_date": pd.Timestamp.now().isoformat(),
        "n_samples": len(X),
        "n_features": len(feature_cols),
        "target_mean": round(float(y.mean()), 2),
        "target_std": round(float(y.std()), 2),
        "cv_scores": cv_scores,
        "oof_mae": round(float(oof_mae), 2),
        "oof_r2": round(float(oof_r2), 4),
        "features": feature_cols,
    }
    meta_file = MODEL_DIR / f"{target}.meta.json"
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved: {meta_file}")

    # Save importance
    imp_file = MODEL_DIR / f"{target}_importance.csv"
    importance.to_csv(imp_file, index=False)
    print(f"  Importance saved: {imp_file}")


def train_classifier(
    df: pd.DataFrame, target: str, target_info: dict
) -> None:
    """Train an XGBClassifier for a binary target (win/loss)."""
    print(f"\n{'-' * 66}")
    print(f"  Training: {target} ({target_info['description']})")
    print(f"{'-' * 66}")

    # Feature columns (exclude metadata + target regressors)
    drop_cols = {target, "player_id", "team", "opponent", "game_date",
                 "season", "week", "season_type",
                 "spread_margin", "total_points", "points_for", "points_against"}
    feature_cols = [c for c in df.columns
                    if c not in drop_cols
                    and df[c].dtype in ("float64", "int64", "float32", "int32", "int8", "int16", "bool")]

    train_df = df.dropna(subset=[target]).copy()
    if train_df.empty:
        print(f"  No valid rows for {target}")
        return

    X = train_df[feature_cols].fillna(0)
    y = train_df[target].values

    base_rate = y.mean()
    target_info["base_rate"] = base_rate
    print(f"  Samples: {len(X)}, Features: {len(feature_cols)}")
    print(f"  Win rate: {base_rate:.1%}")

    # Walk-forward validation
    test_size = min(max(200, int(len(X) * 0.1)), int(len(X) * 0.3))
    tscv = TimeSeriesSplit(n_splits=5, test_size=test_size)
    cv_scores = []
    oof_preds = np.full(len(X), np.nan)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Compute scale_pos_weight for imbalance
        fold_rate = y_train.mean()
        spw = (1 - fold_rate) / max(fold_rate, 0.01)

        model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=spw,
            random_state=42 + fold,
            eval_metric="logloss",
            early_stopping_rounds=30,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        preds = model.predict_proba(X_test)[:, 1]
        oof_preds[test_idx] = preds

        acc = accuracy_score(y_test, (preds >= 0.5).astype(int))
        brier = brier_score_loss(y_test, preds)

        try:
            auc = roc_auc_score(y_test, preds)
        except ValueError:
            auc = 0.5

        cv_scores.append({
            "fold": fold,
            "accuracy": round(acc, 4),
            "brier": round(brier, 4),
            "auc": round(auc, 4),
            "n_train": len(X_train),
            "n_test": len(X_test),
        })
        print(f"  Fold {fold}: Acc={acc:.1%}, Brier={brier:.4f}, AUC={auc:.3f}")

    # Overall OOF metrics
    oof_valid = ~np.isnan(oof_preds)
    oof_acc = accuracy_score(y[oof_valid], (oof_preds[oof_valid] >= 0.5).astype(int))
    oof_brier = brier_score_loss(y[oof_valid], oof_preds[oof_valid])
    naive_brier = base_rate * (1 - base_rate)

    print(f"\n  OOF Accuracy: {oof_acc:.1%} (baseline={max(base_rate, 1-base_rate):.1%})")
    print(f"  OOF Brier:    {oof_brier:.4f} (naive={naive_brier:.4f})")

    # Calibration table
    print("\n  Calibration by confidence bin:")
    bins = np.arange(0, 1.05, 0.05)
    cal_table = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (oof_preds >= lo) & (oof_preds < hi)
        if mask.sum() >= 10:
            actual = y[mask].mean()
            print(f"    [{lo:.0%}-{hi:.0%}): pred={lo+0.025:.1%} actual={actual:.1%}  n={mask.sum()}")
            cal_table.append({
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "model_pred": float(lo + 0.025),
                "actual_rate": float(actual),
                "n": int(mask.sum()),
            })

    # Retrain on full data
    print("\n  Retraining on full dataset...")
    spw_full = (1 - base_rate) / max(base_rate, 0.01)
    final_model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=spw_full,
        random_state=42,
        eval_metric="logloss",
    )
    final_model.fit(X, y)

    # Feature importance
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": final_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\n  Top 15 features:")
    for _, row in importance.head(15).iterrows():
        print(f"    {row['feature']:40s} {row['importance']:.4f}")

    # Save model
    model_file = MODEL_DIR / f"{target}.json"
    final_model.save_model(str(model_file))
    print(f"\n  Model saved: {model_file}")

    # Save metadata
    meta = {
        "target": target,
        "model_type": "XGBClassifier",
        "train_date": pd.Timestamp.now().isoformat(),
        "n_samples": len(X),
        "n_features": len(feature_cols),
        "base_rate": float(base_rate),
        "cv_scores": cv_scores,
        "oof_accuracy": round(float(oof_acc), 4),
        "oof_brier": round(float(oof_brier), 4),
        "features": feature_cols,
    }
    meta_file = MODEL_DIR / f"{target}.meta.json"
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved: {meta_file}")

    # Save calibration
    cal_file = MODEL_DIR / f"{target}_calibration.json"
    with open(cal_file, "w") as f:
        json.dump(cal_table, f, indent=2)
    print(f"  Calibration saved: {cal_file}")

    # Save importance
    imp_file = MODEL_DIR / f"{target}_importance.csv"
    importance.to_csv(imp_file, index=False)
    print(f"  Importance saved: {imp_file}")


def main():
    # Step 1: Fetch data + build features
    featured = fetch_and_featurize()
    if featured.empty:
        return

    # Step 2: Train each target
    for target, info in TARGETS.items():
        if info["type"] == "regression":
            train_regression(featured, target, info)
        elif info["type"] == "classification":
            train_classifier(featured, target, info)

    print(f"\n{'=' * 70}")
    print(f"  All models saved to {MODEL_DIR}/")
    print(f"  {list(TARGETS.keys())}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
