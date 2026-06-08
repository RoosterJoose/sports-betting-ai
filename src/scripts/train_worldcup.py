#!/usr/bin/env python3
"""Train World Cup match outcome classifier.

Trains an LGBM multiclass classifier predicting match outcome:
  0 = home_win, 1 = draw, 2 = away_win

Uses ELO ratings + recent form features from build_feature_dataset().
Saves model + calibration to models/worldcup/.

Usage:
    python -m src.scripts.train_worldcup
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.world_cup import fetch_all_matches, compute_elo, build_feature_dataset
from src.config.settings import PROJECT_ROOT
import lightgbm as lgb

MODEL_DIR = PROJECT_ROOT / "models" / "worldcup"
CALIB_DIR = MODEL_DIR / "calibration"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CALIB_DIR.mkdir(parents=True, exist_ok=True)

OUTCOME_LABELS = ["home_win", "draw", "away_win"]


def train_match_model():
    """Train multiclass model to predict match outcome."""
    print("=" * 65)
    print("  WORLD CUP MATCH OUTCOME CLASSIFIER")
    print("=" * 65)

    # 1. Fetch and compute ELO data
    print("\n1. Fetching match data...")
    df = fetch_all_matches()
    if df.empty:
        print("  No data fetched.")
        return
    print(f"  {len(df)} total matches")

    print("\n2. Computing ELO ratings...")
    elo_df = compute_elo(df)
    if elo_df.empty:
        print("  ELO computation failed.")
        return
    print(f"  {len(elo_df)} matches with ELO")

    print("\n3. Building feature dataset...")
    feat_df = build_feature_dataset(elo_df)
    if feat_df.empty:
        print("  Feature build failed.")
        return
    print(f"  {len(feat_df)} match-rows with features")

    # 4. Prepare labels: create multiclass target from one-hot columns
    # build_feature_dataset creates home_won, draw, away_won columns
    if not all(c in feat_df.columns for c in ["home_won", "draw", "away_won"]):
        print("  Missing outcome columns in feature dataset.")
        return

    y = np.zeros(len(feat_df), dtype=int)
    y[feat_df["away_won"] == 1] = 2
    y[feat_df["draw"] == 1] = 1
    # default 0 = home_won

    # 5. Build feature matrix
    feature_cols = [
        "elo_home", "elo_away", "elo_diff",
        "h_wr", "h_dr", "h_gs", "h_gc", "h_n",
        "a_wr", "a_dr", "a_gs", "a_gc", "a_n",
    ]
    available = [c for c in feature_cols if c in feat_df.columns]
    missing = [c for c in feature_cols if c not in feat_df.columns]
    if missing:
        print(f"  Warning: missing features: {missing}")

    X = feat_df[available].copy().fillna(0)
    y = y[: len(X)]

    # 6. Check class balance
    counts = np.bincount(y, minlength=3)
    print(f"\n  Class distribution:")
    for i, label in enumerate(OUTCOME_LABELS):
        print(f"    {label:12s}: {counts[i]:5d} ({counts[i]/len(y):.1%})")

    # 7. Temporal split (chronological)
    if "match_date" in feat_df.columns:
        dates = pd.to_datetime(feat_df["match_date"])
        sort_idx = dates.argsort()
        X = X.iloc[sort_idx]
        y = y[sort_idx]

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"\n  Train: {len(X_train)}, Test: {len(X_test)}")

    # 8. Train multiclass classifier
    print("\n4. Training model...")
    model = lgb.LGBMModel(
        objective="multiclass",
        num_class=3,
        n_estimators=800,
        num_leaves=31,
        learning_rate=0.03,
        subsample=0.8,
        feature_fraction=0.8,
        reg_alpha=0.3,
        reg_lambda=0.5,
        min_child_samples=20,
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )

    # 9. Evaluate
    preds = model.predict(X_test)
    pred_classes = np.argmax(preds, axis=1)
    accuracy = np.mean(pred_classes == y_test)

    y_onehot = np.zeros((len(y_test), 3))
    y_onehot[np.arange(len(y_test)), y_test] = 1
    brier = float(np.mean(np.sum((preds - y_onehot) ** 2, axis=1)))

    # Naive baseline: always predict the majority class
    majority_class = np.argmax(np.bincount(y_test))
    naive_preds = np.zeros((len(y_test), 3))
    naive_preds[:, majority_class] = 1
    naive_brier = float(np.mean(np.sum((naive_preds - y_onehot) ** 2, axis=1)))

    print(f"\n  Test accuracy: {accuracy:.1%}")
    print(f"  Brier:         {brier:.4f} (naive: {naive_brier:.4f})")
    if brier < naive_brier:
        print(f"  ✅ Beats naive baseline by {(naive_brier - brier) / naive_brier:.0%}")
    else:
        print(f"  ❌ Worse than naive baseline")

    # Per-class accuracy
    for cls_idx, cls_name in enumerate(OUTCOME_LABELS):
        mask = y_test == cls_idx
        if mask.sum() > 0:
            cls_acc = np.mean(pred_classes[mask] == cls_idx)
            print(f"  {cls_name:12s}: acc={cls_acc:.1%} n={mask.sum()}")

    # 10. Feature importance
    imp = pd.DataFrame({"feature": available, "importance": model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False)
    print(f"\n  Top features:")
    for _, r in imp.head(8).iterrows():
        print(f"    {r['feature']:12s} {r['importance']}")

    # 11. Calibration per outcome
    print("\n5. Saving calibration...")
    for cls_idx, cls_name in enumerate(["home", "draw", "away"]):
        cal_table = []
        class_preds = preds[:, cls_idx]
        class_actual = (y_test == cls_idx).astype(int)
        for lo in np.arange(0, 1.0, 0.05):
            hi = min(lo + 0.05, 1.0)
            mask = (class_preds >= lo) & (class_preds < hi)
            if mask.sum() >= 5:
                cal_table.append({
                    "p_pred_min": float(lo),
                    "p_pred_max": float(hi),
                    "p_actual": float(class_actual[mask].mean()),
                    "n": int(mask.sum()),
                })
        cal_path = CALIB_DIR / f"wc_{cls_name}_empirical.json"
        with open(cal_path, "w") as f:
            json.dump({"0": {"bins": cal_table}}, f, indent=2)
        print(f"    Saved {cal_path.name} ({len(cal_table)} bins)")

    # 12. Save model + metadata
    print("\n6. Saving model...")
    model_path = MODEL_DIR / "wc_match_outcome.txt"
    model.booster_.save_model(str(model_path))

    meta = {
        "train_date": datetime.now().isoformat(),
        "n_samples": len(X),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": len(available),
        "features": available,
        "test_accuracy": round(float(accuracy), 4),
        "test_brier": round(float(brier), 4),
        "naive_brier": round(float(naive_brier), 4),
        "best_iteration": int(model.best_iteration_ or 0),
        "top_features": imp.head(10)["feature"].tolist(),
    }
    meta_path = MODEL_DIR / "wc_match_outcome.meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Model:  {model_path}")
    print(f"  Meta:   {meta_path}")
    print(f"\n  Done!")


if __name__ == "__main__":
    train_match_model()
