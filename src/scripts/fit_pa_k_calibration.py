#!/usr/bin/env python3
"""Fit Beta Calibration for the PA outcome model's K class probability.

The PA model (8-class LightGBM) over-predicts K by ~1.3 pp at baseline
(23.8% vs 22.5% actual), contributing to the +11.8% simulator K bias.

This script:
1. Loads the PA outcome model + training dataset
2. Gets test-set predictions (85/15 temporal split)
3. Extracts P(K) from the 8-class output
4. Fits BetaCalibrator on P(K) vs actual K outcomes
5. Saves calibration to models/mlb/f5_pa_k_cal.json

After fitting, the simulator will load this calibration and apply it
to adjust P(K) per PA.
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import lightgbm as lgb
from src.config.settings import PROJECT_ROOT
from src.models.calibrator import BetaCalibrator
from src.mlb.f5_pa_outcome import build_pa_dataset, add_rolling_features
from src.data.mlb_pitching_stats import fetch_multiyear_pitching_stats, merge_pitching_stats_into_pa

MODEL_DIR = PROJECT_ROOT / "models" / "mlb"
K_CAL_PATH = MODEL_DIR / "f5_pa_k_cal.json"


def main():
    print("=" * 65)
    print("  PA MODEL K-CLASS BETA CALIBRATION")
    print("=" * 65)

    # 1. Load model
    print("\n1. Loading PA outcome model...")
    model_path = MODEL_DIR / "f5_pa_outcome.txt"
    meta_path = MODEL_DIR / "f5_pa_outcome.meta.json"
    if not model_path.exists():
        print("  Model not found. Run: python -m src.mlb.f5_pa_outcome")
        return

    model = lgb.Booster(model_file=str(model_path))
    with open(meta_path) as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    outcome_names = meta["outcome_names"]
    print(f"  Model: {meta['n_samples']:,} samples, {len(feature_cols)} features, "
          f"acc={meta['test_accuracy']:.1%}")

    # 2. Build PA dataset (same as training)
    print("\n2. Building PA dataset...")
    pa_df = build_pa_dataset([2024, 2025])
    if pa_df.empty:
        print("  No data!")
        return

    # 3. Merge pitching stats (FIP, K/9, BB/9)
    print("\n3. Merging pitching stats...")
    pitching_df = fetch_multiyear_pitching_stats([2024, 2025], min_ip=30)
    if not pitching_df.empty:
        pa_df = merge_pitching_stats_into_pa(pa_df, pitching_df)

    # 4. Add rolling features
    print("\n4. Adding rolling features...")
    pa_df = add_rolling_features(pa_df)

    # 5. Get model predictions on ALL data (with temporal split)
    print(f"\n5. Computing test-set predictions...")

    # Sort by date
    pa_df = pa_df.sort_values("game_date").reset_index(drop=True)

    X = np.array([pa_df[c].fillna(0).values for c in feature_cols], dtype=float).T
    y = pa_df["outcome_code"].values.astype(int)

    # 85/15 temporal split (same as training)
    split_idx = int(len(pa_df) * 0.85)
    X_test = X[split_idx:]
    y_test = y[split_idx:]

    print(f"  Test set: {len(X_test)} PAs")
    print(f"  Actual K rate in test set: {(y_test == 7).mean():.3f}")

    # Predict in batches to manage memory
    batch_size = 10000
    k_probs = []
    for start in range(0, len(X_test), batch_size):
        end = min(start + batch_size, len(X_test))
        batch = X_test[start:end]
        preds = model.predict(batch)
        k_probs.extend(preds[:, 7].tolist())  # class 7 = K
        if (start + batch_size) % 50000 == 0:
            print(f"    Predicted {start + batch_size}/{len(X_test)}...", flush=True)

    k_probs = np.array(k_probs, dtype=float)
    k_actual = (y_test == 7).astype(int)

    # 6. Analyze calibration
    print(f"\n6. K probability calibration analysis:")
    print(f"   Mean P(K): {k_probs.mean():.3f}")
    print(f"   Actual K rate: {k_actual.mean():.3f}")
    print(f"   Raw bias: {k_probs.mean() - k_actual.mean():+.3f}")

    # Calibration by bin
    print(f"\n   Calibration by confidence bin:")
    for lo in np.arange(0, 0.6, 0.05):
        hi = lo + 0.05
        mask = (k_probs >= lo) & (k_probs < hi)
        if mask.sum() < 100:
            continue
        actual_rate = k_actual[mask].mean()
        print(f"     [{lo:.0%}-{hi:.0%}): pred={k_probs[mask].mean():.3f} "
              f"actual={actual_rate:.3f} n={mask.sum():,}")

    # 7. Fit BetaCal
    print(f"\n7. Fitting Beta Calibration...")
    raw_arr = np.clip(k_probs, 0.001, 0.999)
    out_arr = k_actual

    # Filter near-boundary values for stable fit
    valid = (raw_arr > 0.01) & (raw_arr < 0.99)
    print(f"   Valid predictions: {valid.sum():,}/{len(raw_arr):,}")

    beta_cal = BetaCalibrator()
    beta_cal.fit(raw_arr[valid], out_arr[valid])
    print(f"   BetaCal: a={beta_cal.a:.4f}, b={beta_cal.b:.4f}, c={beta_cal.c:.4f}")

    # Evaluate calibration improvement on ALL test data
    cal_probs = beta_cal.calibrate(raw_arr)
    print(f"   Raw bias: {float(np.mean(raw_arr - out_arr)):+.4f}")
    print(f"   Cal bias: {float(np.mean(cal_probs - out_arr)):+.4f}")

    # Calibration by bin after correction
    print(f"\n   Post-calibration by bin:")
    for lo in np.arange(0, 0.6, 0.05):
        hi = lo + 0.05
        mask = (k_probs >= lo) & (k_probs < hi)
        if mask.sum() < 100:
            continue
        actual_rate = k_actual[mask].mean()
        cal_mean = cal_probs[mask].mean()
        print(f"     [{lo:.0%}-{hi:.0%}): raw={k_probs[mask].mean():.3f} "
              f"cal={cal_mean:.3f} actual={actual_rate:.3f} (n={mask.sum():,})")

    # 8. Save calibration
    beta_cal.save(K_CAL_PATH)
    print(f"\n8. Saved to {K_CAL_PATH}")

    # 9. Estimate impact on simulated K/game bias
    print(f"\n9. Estimated impact on K props:")
    # Average pitcher faces ~25 batters per game
    avg_bf = 25
    raw_k_per_game = k_probs.mean() * avg_bf
    cal_k_per_game = cal_probs.mean() * avg_bf
    actual_k_per_game = k_actual.mean() * avg_bf
    print(f"   Per PA: raw P(K)={k_probs.mean():.3f} cal P(K)={cal_probs.mean():.3f} "
          f"actual={k_actual.mean():.3f}")
    print(f"   Per game (25 BF): raw={raw_k_per_game:.1f} cal={cal_k_per_game:.1f} "
          f"actual={actual_k_per_game:.1f}")
    print(f"   Bias: raw={raw_k_per_game - actual_k_per_game:+.1f} K/game "
          f"({(raw_k_per_game - actual_k_per_game)/actual_k_per_game:+.1%})")
    print(f"   After calibration: {cal_k_per_game - actual_k_per_game:+.1f} K/game "
          f"({(cal_k_per_game - actual_k_per_game)/actual_k_per_game:+.1%})")


if __name__ == "__main__":
    main()
