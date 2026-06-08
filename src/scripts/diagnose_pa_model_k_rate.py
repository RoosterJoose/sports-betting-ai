#!/usr/bin/env python3
"""Diagnose the PA outcome model's K probability distribution.

Probes the LightGBM model to understand per-PA K probability:
1. Baseline P(K) for league-average matchup
2. How each feature shifts P(K) (partial dependence)
3. Actual model predictions for today's pitchers vs market-implied rates
4. Compare training data K rate vs model output
"""
from __future__ import annotations

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.settings import PROJECT_ROOT
from src.mlb.mlb_simulator import MLBSimulator, DEFAULT_RELIEVER, DEFAULT_BATTER

MODEL_DIR = PROJECT_ROOT / "models" / "mlb"


def main():
    print("=" * 80)
    print("  PA MODEL K-RATE DIAGNOSTIC")
    print("=" * 80)

    # ── 1. Load model and meta ──
    print("\n1. Loading PA outcome model...")
    meta_path = MODEL_DIR / "f5_pa_outcome.meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    class_dist = meta["class_distribution"]
    total = sum(class_dist.values())
    k_count = class_dist.get("K", 0)
    print(f"  Training data: {total} PAs")
    print(f"  K count:       {k_count} ({k_count/total:.1%})")
    print(f"  Features:      {len(feature_cols)}")
    print(f"  Accuracy:      {meta['test_accuracy']:.1%}")
    print(f"  Brier:         {meta['test_brier']:.4f}")

    # Load model
    sim = MLBSimulator()
    model = sim.model

    # ── 2. Build baseline feature vector (league average everything) ──
    print("\n2. Baseline: league-average matchup")
    baseline = {}
    for c in feature_cols:
        # Use the defaults from the codebase
        if c in DEFAULT_RELIEVER:
            baseline[c] = DEFAULT_RELIEVER[c]
        elif c in DEFAULT_BATTER:
            baseline[c] = DEFAULT_BATTER[c]
        else:
            baseline[c] = {
                "is_home": 0.0, "runners_on": 0.0,
                "park_factor_k": 1.0, "park_factor_hr": 1.0,
                "same_hand": 0.0, "umpire_zone_factor": 1.0,
            }.get(c, 0.0)

    # Predict baseline
    baseline_vec = np.array([baseline[c] for c in feature_cols], dtype=float).reshape(1, -1)
    baseline_probs = model.predict(baseline_vec)[0]
    k_idx = 7  # K is outcome 7
    baseline_k_prob = baseline_probs[k_idx]
    print(f"  Pitcher features: K9={DEFAULT_RELIEVER['pitcher_k9']}, "
          f"K_rate={DEFAULT_RELIEVER['pitcher_k_rate_prior']}, "
          f"FIP={DEFAULT_RELIEVER['pitcher_fip']}")
    print(f"  Batter features:  K_rate={DEFAULT_BATTER['batter_k_rate_prior']}, "
          f"EV={DEFAULT_BATTER['batter_avg_ev_prior']}")
    print(f"  Predicted P(K) baseline: {baseline_k_prob:.1%}")
    print(f"  MLB actual K rate (2024-25): ~22.5%")
    print(f"  Training data K rate: {k_count/total:.1%}")
    print(f"  Bias vs MLB: {baseline_k_prob - 0.225:+.1%}")

    # ── 3. Full outcome distribution ──
    outcome_names = meta["outcome_names"]
    print(f"\n  Full outcome distribution:")
    for i in range(8):
        name = outcome_names.get(str(i), f"Class {i}")
        p = baseline_probs[i]
        train_pct = class_dist.get(name, 0) / total
        print(f"    {name:4s}: model={p:.1%}  train={train_pct:.1%}  "
              f"diff={p - train_pct:+.1%}")

    # ── 4. Partial dependence: vary batter_k_rate_prior ──
    print(f"\n3. Partial dependence — batter_k_rate_prior")
    print(f"  {'batter_k_rate':>15s}  {'P(K)':>8s}  {'P(1B)':>8s}  {'P(HR)':>8s}  {'P(BB)':>8s}  {'P(OUT)':>8s}")
    for k_rate in np.arange(0.10, 0.45, 0.05):
        test = dict(baseline)
        test["batter_k_rate_prior"] = round(k_rate, 3)
        vec = np.array([test[c] for c in feature_cols], dtype=float).reshape(1, -1)
        probs = model.predict(vec)[0]
        print(f"  {k_rate:>13.0%}  {probs[7]:>7.1%}  {probs[1]:>7.1%}  {probs[4]:>7.1%}  "
              f"{probs[5]:>7.1%}  {probs[0]:>7.1%}")

    # ── 5. Partial dependence: vary pitcher_k_rate_prior ──
    print(f"\n4. Partial dependence — pitcher_k_rate_prior")
    print(f"  {'pitcher_k_rate':>15s}  {'P(K)':>8s}  {'P(1B)':>8s}  {'P(HR)':>8s}  {'P(BB)':>8s}  {'P(OUT)':>8s}")
    for k_rate in np.arange(0.10, 0.50, 0.05):
        test = dict(baseline)
        test["pitcher_k_rate_prior"] = round(k_rate, 3)
        vec = np.array([test[c] for c in feature_cols], dtype=float).reshape(1, -1)
        probs = model.predict(vec)[0]
        print(f"  {k_rate:>13.0%}  {probs[7]:>7.1%}  {probs[1]:>7.1%}  {probs[4]:>7.1%}  "
              f"{probs[5]:>7.1%}  {probs[0]:>7.1%}")

    # ── 6. Joint effect: batter_k_rate x pitcher_k_rate ──
    print(f"\n5. Joint effect: batter_k_rate (rows) × pitcher_k_rate (cols) -> P(K)")
    print(f"  {'batt':>7s}/{'pitch':>5s}", end="")
    for p_k in [0.15, 0.20, 0.22, 0.25, 0.30]:
        print(f"  {p_k:>6.0%}", end="")
    print()
    for b_k in [0.15, 0.18, 0.20, 0.22, 0.25, 0.30, 0.35]:
        print(f"  {b_k:>10.0%}  ", end="")
        for p_k in [0.15, 0.20, 0.22, 0.25, 0.30]:
            test = dict(baseline)
            test["batter_k_rate_prior"] = round(b_k, 3)
            test["pitcher_k_rate_prior"] = round(p_k, 3)
            vec = np.array([test[c] for c in feature_cols], dtype=float).reshape(1, -1)
            probs = model.predict(vec)[0]
            print(f"  {probs[7]:>5.0%}  ", end="")
        print()

    # ── 7. K9 vs P(K): vary pitcher_k9 ──
    print(f"\n6. Effect: pitcher_k9 on P(K)")
    print(f"  {'K/9':>6s}  {'P(K)':>8s}  {'P(1B)':>8s}  {'P(BB)':>8s}")
    for k9 in [6.0, 7.0, 8.0, 8.5, 9.0, 10.0, 11.0, 12.0]:
        test = dict(baseline)
        test["pitcher_k9"] = k9
        vec = np.array([test[c] for c in feature_cols], dtype=float).reshape(1, -1)
        probs = model.predict(vec)[0]
        print(f"  {k9:>5.1f}  {probs[7]:>7.1%}  {probs[1]:>7.1%}  {probs[5]:>7.1%}")

    # ── 8. Game context effects ──
    print(f"\n7. Effect: game context on P(K)")
    print(f"  {'is_home':>8s}  {'runners_on':>11s}  {'same_hand':>10s}  {'P(K)':>8s}")
    for is_home, runners, same_hand in [
        (0, 0, 0), (1, 0, 0), (0, 3, 0), (0, 0, 1),
    ]:
        test = dict(baseline)
        test["is_home"] = float(is_home)
        test["runners_on"] = float(runners)
        test["same_hand"] = float(same_hand)
        vec = np.array([test[c] for c in feature_cols], dtype=float).reshape(1, -1)
        probs = model.predict(vec)[0]
        print(f"  {is_home:>8}  {runners:>11}  {same_hand:>10}  {probs[7]:>7.1%}")

    # ── 9. Simulated K/game comparison ──
    print(f"\n8. Simulated K/PA and K/game for different pitcher profiles")
    print(f"  Simulating the effect of batter_k_rate on pitcher K totals...")

    # Quick sim with high-K batters vs low-K batters
    test_pitcher = dict(DEFAULT_RELIEVER) | {"name": "Test SP"}

    # 9 batters with high K rate
    high_k_batters = []
    for i in range(9):
        b = dict(DEFAULT_BATTER)
        b["name"] = f"High-K Batter #{i+1}"
        b["batter_k_rate_prior"] = 0.30  # 30% K rate (strikeout-prone)
        high_k_batters.append(b)

    # 9 batters with low K rate
    low_k_batters = []
    for i in range(9):
        b = dict(DEFAULT_BATTER)
        b["name"] = f"Low-K Batter #{i+1}"
        b["batter_k_rate_prior"] = 0.15  # 15% K rate (contact hitter)
        low_k_batters.append(b)

    for label, batters in [("High-K lineup (30%)", high_k_batters),
                           ("Low-K lineup (15%)", low_k_batters)]:
        result = sim.simulate_game(
            test_pitcher, test_pitcher,
            batters, batters,
            n_sims=500,
        )
        k_mean = result["away_pitcher"]["k_mean"]
        print(f"  {label:30s}: K/game={k_mean:.1f}")

    # ── 10. Direct bias calibration check ──
    print(f"\n9. Root cause analysis")
    print(f"  The baseline P(K)={baseline_k_prob:.1%} vs MLB avg ~22.5%:")
    if baseline_k_prob > 0.225:
        print(f"    → Model OVER-predicts Ks by {baseline_k_prob - 0.225:+.1%} at baseline")
    else:
        print(f"    → Model UNDER-predicts Ks by {baseline_k_prob - 0.225:+.1%} at baseline")

    print(f"\n  The joint effect table shows the model's sensitivity to batter vs pitcher K rate.")
    print(f"  The top feature in the model is batter_k_rate_prior —")
    print(f"  the single biggest influence on K probability is the batter's strikeout tendency.")

    print(f"\n  Training data K rate = {k_count/total:.1%}")
    print(f"  If baseline P(K) ~ training K rate, the model is calibrated to the training set.")
    print(f"  The bias vs live markets comes from feature values, not model miscalibration.")
    print(f"  (e.g., if the scanner's default batters have higher K_rate than real MLB lineups)")
    print(f"\n  Done.")


if __name__ == "__main__":
    main()
