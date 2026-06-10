#!/usr/bin/env python3
"""Train World Cup match outcome classifier — temporal train/val/test split.

Temporal splits:
  - Train: all matches before 2022 (no 2022 World Cup data)
  - Val:   2022 World Cup only (tournament_code == 'WC')
  - Test:  2023+ matches (2026 qualifiers, friendlies)

This ensures clean out-of-sample validation on the 2022 World Cup.

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
    """Train multiclass model with clean temporal train/val/test split."""
    print("=" * 65)
    print("  WORLD CUP MATCH OUTCOME CLASSIFIER")
    print("  Temporal split: train ≤2021 (all pre-WC), val 2022 WC, test 2023+" )
    print("=" * 65)

    # 1. Fetch data
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

    # Merge tournament_code back for proper splitting
    print("\n3. Merging tournament codes...")
    elo_df = elo_df.merge(
        df[["match_date", "home_team", "away_team", "tournament_code"]],
        on=["match_date", "home_team", "away_team"],
        how="left",
    )
    tc_missing = elo_df["tournament_code"].isna().sum()
    print(f"  {tc_missing} rows with missing tournament_code")

    print("\n4. Building feature dataset...")
    feat_df = build_feature_dataset(elo_df)
    if feat_df.empty:
        print("  Feature build failed.")
        return
    print(f"  {len(feat_df)} match-rows with features")

    # Carry tournament_code through to the feature dataset
    feat_df = feat_df.merge(
        elo_df[["match_date", "home_team", "away_team", "tournament_code"]],
        on=["match_date", "home_team", "away_team"],
        how="left",
    )

    # 5. Prepare labels
    if not all(c in feat_df.columns for c in ["home_won", "draw", "away_won"]):
        print("  Missing outcome columns in feature dataset.")
        return

    y = np.zeros(len(feat_df), dtype=int)
    y[feat_df["away_won"] == 1] = 2
    y[feat_df["draw"] == 1] = 1

    # 6. Build feature matrix — add derived features for draw/away prediction
    feat_df["elo_diff_abs"] = feat_df["elo_diff"].abs()
    feat_df["h_goal_diff"] = feat_df["h_gs"] - feat_df["h_gc"]
    feat_df["a_goal_diff"] = feat_df["a_gs"] - feat_df["a_gc"]
    # Tournament type: qualifier vs friendly (friendlies have more draws/upsets)
    feat_df["is_friendly"] = feat_df.get("tournament_code", "").fillna("").str.contains("FR", case=False).astype(int)

    feature_cols = [
        "elo_home", "elo_away", "elo_diff", "elo_diff_abs",
        "h_perf", "h_opp_elo", "h_gs", "h_gc", "h_goal_diff", "h_n",
        "a_perf", "a_opp_elo", "a_gs", "a_gc", "a_goal_diff", "a_n",
        "is_friendly", "is_neutral",
    ]
    available = [c for c in feature_cols if c in feat_df.columns]
    missing = [c for c in feature_cols if c not in feat_df.columns]
    if missing:
        print(f"  Warning: missing features: {missing}")

    X = feat_df[available].copy().fillna(0)
    y = y[: len(X)]

    # 7. Temporal split
    if "match_date" not in feat_df.columns:
        print("  No match_date column — cannot do temporal split.")
        return

    dates = pd.to_datetime(feat_df["match_date"])
    tourn_code = feat_df["tournament_code"].fillna("")

    # Train: all matches before 2022 (entire pre-WC history)
    train_mask = dates.dt.year <= 2021
    # Val: 2022 World Cup only
    val_mask = (dates.dt.year == 2022) & (tourn_code == "WC")
    # Test: 2023+ matches (2026 qualifiers, friendlies, etc.)
    test_mask = dates >= pd.Timestamp("2023-01-01")

    # Sort chronologically within each split
    train_idx = dates[train_mask].sort_values().index
    val_idx = dates[val_mask].sort_values().index
    test_idx = dates[test_mask].sort_values().index

    X_train = X.loc[train_idx]
    y_train = y[train_idx]
    X_val = X.loc[val_idx]
    y_val = y[val_idx]
    X_test = X.loc[test_idx]
    y_test = y[test_idx]

    print(f"\n  Split sizes:")
    print(f"    Train (≤2021):          {len(X_train):5d}")
    print(f"    Val   (2022 WC):    {len(X_val):5d}")
    print(f"    Test  (2023+):      {len(X_test):5d}")

    if len(X_val) < 10:
        print("  Validation set too small — check tournament_code filter.")
        return

    # Class distribution per split
    for split_name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        counts = np.bincount(y_split, minlength=3)
        print(f"\n  {split_name} class distribution:")
        for i, label in enumerate(OUTCOME_LABELS):
            print(f"    {label:12s}: {counts[i]:4d} ({counts[i]/len(y_split):.1%})")

    # 8. Train multiclass classifier — NO sample weighting
    # Previous ISNS (sqrt inverse freq) weighting over-amplified draws (2x)
    # and away wins (3.76x), causing 60-70% draw predictions vs 25% reality.
    # With 500 draw samples in training, the model can learn naturally.
    print("\n5. Training model (early stopping on 2022 WC validation, no class weighting)...")

    model = lgb.LGBMModel(
        objective="multiclass",
        num_class=3,
        n_estimators=800,
        num_leaves=31,
        learning_rate=0.02,
        subsample=0.75,
        feature_fraction=0.75,
        reg_alpha=1.0,
        reg_lambda=2.0,
        min_child_samples=50,
        random_state=42,
        verbosity=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )

    # 9. Evaluate on all three splits
    def evaluate_split(X, y, label):
        preds = model.predict(X)
        pred_classes = np.argmax(preds, axis=1)

        y_onehot = np.zeros((len(y), 3))
        y_onehot[np.arange(len(y)), y] = 1
        brier = float(np.mean(np.sum((preds - y_onehot) ** 2, axis=1)))

        majority_class = np.argmax(np.bincount(y))
        naive_preds = np.zeros((len(y), 3))
        naive_preds[:, majority_class] = 1
        naive_brier = float(np.mean(np.sum((naive_preds - y_onehot) ** 2, axis=1)))

        acc = np.mean(pred_classes == y)
        print(f"  {label:10s} accuracy: {acc:.1%}  Brier: {brier:.4f} (naive: {naive_brier:.4f})  ",
              end="")
        if brier < naive_brier:
            print(f"✅ +{(naive_brier - brier) / naive_brier:.0%}")
        else:
            print(f"❌ {(naive_brier - brier) / naive_brier:.0%}")
        return preds, pred_classes, brier

    print(f"\n  Evaluation:")
    preds_train, _, _ = evaluate_split(X_train, y_train, "Train")
    preds_val, pred_classes_val, brier_val = evaluate_split(X_val, y_val, "2022 WC")
    preds_test, _, brier_test = evaluate_split(X_test, y_test, "2023+")

    # Per-class accuracy on 2022 WC (validation)
    print(f"\n  Per-class accuracy (2022 WC):")
    for cls_idx, cls_name in enumerate(OUTCOME_LABELS):
        mask = y_val == cls_idx
        if mask.sum() > 0:
            cls_acc = np.mean(pred_classes_val[mask] == cls_idx)
            print(f"    {cls_name:12s}: acc={cls_acc:.1%} n={mask.sum()}")

    # 10. Feature importance
    imp = pd.DataFrame({"feature": available, "importance": model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False)
    print(f"\n  Top features:")
    for _, r in imp.head(8).iterrows():
        print(f"    {r['feature']:12s} {r['importance']}")

    # 11. Calibration from TRAINING set
    print(f"\n6. Saving calibration (from training set, {len(y_train)} matches)...")
    for cls_idx, cls_name in enumerate(["home", "draw", "away"]):
        cal_table = []
        class_preds = preds_train[:, cls_idx]
        class_actual = (y_train == cls_idx).astype(int)
        for lo in np.arange(0, 1.0, 0.05):
            hi = min(lo + 0.05, 1.0)
            mask = (class_preds >= lo) & (class_preds < hi)
            if mask.sum() >= 20:
                cal_table.append({
                    "p_pred_min": float(lo),
                    "p_pred_max": float(hi),
                    "p_actual": float(class_actual[mask].mean()),
                    "n": int(mask.sum()),
                })
        # File naming: "{cls_name}_empirical.json" so EmpiricalCalibrator
        # can find it via calibrate("home", 0, p) lookup.
        cal_path = CALIB_DIR / f"{cls_name}_empirical.json"
        with open(cal_path, "w") as f:
            json.dump({"0": {"bins": cal_table}}, f, indent=2)
        print(f"    Saved {cal_path.name} ({len(cal_table)} bins)")

    # Also save val calibration for reference (57 matches — noisy, reference only)
    print("  Saving val calibration (2022 WC, 57 matches — reference only)...")
    for cls_idx, cls_name in enumerate(["home", "draw", "away"]):
        cal_table = []
        class_preds = preds_val[:, cls_idx]
        class_actual = (y_val == cls_idx).astype(int)
        for lo in np.arange(0, 1.0, 0.05):
            hi = min(lo + 0.05, 1.0)
            mask = (class_preds >= lo) & (class_preds < hi)
            if mask.sum() >= 3:
                cal_table.append({
                    "p_pred_min": float(lo),
                    "p_pred_max": float(hi),
                    "p_actual": float(class_actual[mask].mean()),
                    "n": int(mask.sum()),
                })
        cal_path = CALIB_DIR / f"{cls_name}_val_empirical.json"
        with open(cal_path, "w") as f:
            json.dump({"0": {"bins": cal_table}}, f, indent=2)

    # === Neutral-venue empirical offset (per NotebookLM rec) ===
    # Computes Δ_class = mean(P_model) - actual_rate on the 2022 WC val set
    # (which is 100% neutral-venue matches — the entire WC tournament is at
    # neutral venues). Capped at ±0.15 per class. Applied in scan_wc.py
    # predict_match() when is_neutral=1.
    #
    # This replaces the previous Platt scaling + Elo-diff-aware approach
    # which amplified draw predictions to 60%+ for lopsided matchups
    # because the raw model already over-predicts draws at neutral venues.
    # The empirical offset is the sharp-bettor standard: a single additive
    # shift per class that forces the model's neutral-venue prior to match
    # reality (~33/33/33 in close-Elo matches).
    print(f"\n6.5. Computing neutral-venue empirical offset (2022 WC val)...")

    # Filter to neutral-venue rows only (val set is 100% WC, all neutral,
    # but we filter explicitly for robustness if the split ever changes)
    is_neutral_val = feat_df.loc[val_idx, "is_neutral"].fillna(0).astype(int).values
    neutral_mask = is_neutral_val == 1
    n_neutral = int(neutral_mask.sum())

    if n_neutral < 10:
        print(f"    WARNING: only {n_neutral} neutral-venue val rows — skipping offset")
    else:
        preds_neutral = preds_val[neutral_mask]
        y_neutral = y_val[neutral_mask]

        actual_rates = np.bincount(y_neutral, minlength=3) / n_neutral
        pred_means = preds_neutral.mean(axis=0)
        deltas = pred_means - actual_rates  # positive = over-predicts
        cap = 0.15
        deltas_capped = np.clip(deltas, -cap, cap)
        delta_home, delta_draw, delta_away = deltas_capped

        print(f"    Neutral-venue val rows: {n_neutral}")
        print(f"    Actual rates: H={actual_rates[0]:.1%}  D={actual_rates[1]:.1%}  A={actual_rates[2]:.1%}")
        print(f"    Model means:  H={pred_means[0]:.1%}  D={pred_means[1]:.1%}  A={pred_means[2]:.1%}")
        print(f"    Raw Δ:        H={deltas[0]:+.3f}  D={deltas[1]:+.3f}  A={deltas[2]:+.3f}")
        print(f"    Capped Δ:     H={delta_home:+.3f}  D={delta_draw:+.3f}  A={delta_away:+.3f}  (cap=±{cap})")

        # Apply offset and recompute Brier on neutral-venue val
        preds_offset = preds_neutral.copy()
        preds_offset[:, 0] -= delta_home
        preds_offset[:, 1] -= delta_draw
        preds_offset[:, 2] -= delta_away
        preds_offset = np.maximum(preds_offset, 0.001)
        preds_offset = preds_offset / preds_offset.sum(axis=1, keepdims=True)

        y_onehot_n = np.zeros((n_neutral, 3))
        y_onehot_n[np.arange(n_neutral), y_neutral] = 1
        brier_neutral_before = float(np.mean(np.sum((preds_neutral - y_onehot_n) ** 2, axis=1)))
        brier_neutral_after = float(np.mean(np.sum((preds_offset - y_onehot_n) ** 2, axis=1)))

        pred_classes_neutral = np.argmax(preds_neutral, axis=1)
        pred_classes_offset = np.argmax(preds_offset, axis=1)
        acc_neutral_before = float((pred_classes_neutral == y_neutral).mean())
        acc_neutral_after = float((pred_classes_offset == y_neutral).mean())

        print(f"\n    Neutral-venue Brier: {brier_neutral_before:.4f} → {brier_neutral_after:.4f}  ({(brier_neutral_after - brier_neutral_before):+.4f})")
        print(f"    Neutral-venue Acc:  {acc_neutral_before:.1%} → {acc_neutral_after:.1%}  ({(acc_neutral_after - acc_neutral_before):+.1%})")

        # Save to neutral_offset.json (auto-applied by scan_wc.py on next scan)
        offset_path = CALIB_DIR / "neutral_offset.json"
        offset_data = {
            "applied": True,
            "delta_home": float(delta_home),
            "delta_draw": float(delta_draw),
            "delta_away": float(delta_away),
            "cap": cap,
            "n_val": n_neutral,
            "val_acc_before": acc_neutral_before,
            "val_acc_after": acc_neutral_after,
            "val_brier_before": brier_neutral_before,
            "val_brier_after": brier_neutral_after,
            "actual_rates": {"home": float(actual_rates[0]), "draw": float(actual_rates[1]), "away": float(actual_rates[2])},
            "pred_means": {"home": float(pred_means[0]), "draw": float(pred_means[1]), "away": float(pred_means[2])},
            "raw_deltas": {"home": float(deltas[0]), "draw": float(deltas[1]), "away": float(deltas[2])},
            "computed_at": datetime.now().isoformat(),
            "source": f"train_worldcup.py (2022 WC val, {n_neutral} neutral-venue rows)",
            "notes": "Apply only at is_neutral=1 matches. Renormalized to sum to 1.",
        }
        with open(offset_path, "w") as f:
            json.dump(offset_data, f, indent=2)
        print(f"    Saved: {offset_path.name}")

    # 12. Save model + metadata
    print(f"\n7. Saving model...")
    model_path = MODEL_DIR / "wc_match_outcome.txt"
    model.booster_.save_model(str(model_path))

    meta = {
        "train_date": datetime.now().isoformat(),
        "temporal_split": "pre-2022 train / 2022 WC val / 2023+ test",
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "n_features": int(len(available)),
        "features": available,
        "val_accuracy": float(np.mean(pred_classes_val == y_val)),
        "val_brier": float(brier_val),
        "test_brier": float(brier_test),
        "best_iteration": int(model.best_iteration_ or 0),
        "top_features": imp.head(10)["feature"].tolist(),
    }
    meta_path = MODEL_DIR / "wc_match_outcome.meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Model:  {model_path}")
    print(f"  Meta:   {meta_path}")
    print(f"\n  Done! Model trained on pre-2022 data only.")
    print(f"  Validate on 2022 WC: {meta['val_accuracy']:.1%} accuracy, Brier={brier_val:.4f}")


if __name__ == "__main__":
    train_match_model()
