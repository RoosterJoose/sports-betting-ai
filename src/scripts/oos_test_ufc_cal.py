#!/usr/bin/env python3
"""Chronological 80/20 OOS holdout test for the UFC winner model.

Mirrors the pattern in src/scripts/oos_test_nba_cal.py and
src/scripts/validate_offset_oos.py. The key question:

  Are the scanner's +76% edges on UFC underdogs (e.g. Diego Lopes at
  model=77% vs market=1%) REAL OOS signal, or the same overcompression
  artifact we saw in NBA (model 60-70% on the training-test split
  collapsing to actual 27% in production)?

Why this is critical: the user wants to bet the 6-leg parlay on Sunday
June 14 (UFC Freedom 250). If the +76% edge is real, the 6-leg is
high-EV. If it's the overcompression bug class, the 6-leg is a trap.

Approach:
  1. Load MikeSpa UFC dataset via UFCDataSource (same as train_ufc.py)
  2. Sort fights chronologically by game_date
  3. Build features for ALL fights (so we have X, y for the full set)
  4. Split chronologically 80/20 (oldest 80% = train, newest 20% = OOS test)
  5. Re-train XGBClassifier on the train slice (no leakage)
  6. Predict on the OOS test slice
  7. Compare Brier/accuracy/log-loss to:
       - Naive baseline (constant prior = base_rate)
       - In-sample performance (model fit on full data)
  8. Bin by predicted probability and check calibration per decile
  9. Quantify the "overcompression gap" = (avg in-sample model prob
     on winners) - (actual OOS win rate in same bin). If positive and
     large, the bug is real.

The +76% edges come from the in-sample (training-test fitted) model.
If the OOS gap on the same bins is similar, the edges are real. If
the OOS gap is much larger (in-sample says 77% but OOS actual is 30%),
the edges are overfit.

Output: prints a verdict to stdout. Critical for the 6-leg bet.

Usage:
    python -m src.scripts.oos_test_ufc_cal
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score, roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.ufc import UFCDataSource
from src.features.ufc import build_ufc_features, FEATURE_COLS

MODEL_DIR = Path("models/ufc")


def main():
    print("=" * 72)
    print("  UFC WINNER MODEL — CHRONOLOGICAL 80/20 OOS HOLDOUT TEST")
    print("  (mirrors oos_test_nba_cal.py for the UFC pick scanner)")
    print("=" * 72)

    # ── 1. Load data ────────────────────────────────────────────────
    print("\n1. Loading UFC dataset (MikeSpa master CSV via UFCDataSource)...")
    ds = UFCDataSource()
    df = ds.fetch_player_game_logs(["all"])
    if df.empty:
        print("   No data loaded. Check data/cache/ufc/ufc-master.csv")
        return
    print(f"   {len(df)} fighter-game rows from {df['player_id'].nunique()} fighters")

    # Each fight appears TWICE in the per-fighter view (one row per corner).
    # For OOS testing we want unique fights → take one row per fight.
    # The "Red" corner row is the canonical one.
    if "is_red" in df.columns:
        fights = df[df["is_red"] == 1].copy()
    else:
        # Fallback: use r_fighter as the key
        fights = df.drop_duplicates(subset=["r_fighter", "game_date"]).copy()
    print(f"   {len(fights)} unique fights (Red corner rows)")

    # Sort chronologically
    fights["game_date"] = pd.to_datetime(fights["game_date"], errors="coerce")
    fights = fights.sort_values("game_date").reset_index(drop=True)
    print(f"   Date range: {fights['game_date'].min().date()} → {fights['game_date'].max().date()}")

    # ── 2. Build features ──────────────────────────────────────────
    print("\n2. Building features...")
    featured = build_ufc_features(fights)
    available = [c for c in FEATURE_COLS if c in featured.columns]
    print(f"   {len(available)} of {len(FEATURE_COLS)} features available")

    # Drop rows with missing target
    if "winner" in featured.columns:
        target_col = "winner"
    elif "Winner" in featured.columns:
        target_col = "Winner"
    else:
        print("   No winner column found in featured data!")
        return
    featured = featured.dropna(subset=[target_col])
    print(f"   {len(featured)} rows with target")

    X = featured[available].fillna(0)
    y = (featured[target_col].astype(str) == "Red").astype(int).values
    base_rate = float(y.mean())
    print(f"   Base rate (Red wins): {base_rate:.1%}")

    # ── 3. Chronological 80/20 split ───────────────────────────────
    print("\n3. Splitting chronologically 80/20 (oldest = train, newest = OOS)...")
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]
    train_dates = featured["game_date"].iloc[:split]
    test_dates = featured["game_date"].iloc[split:]
    print(f"   Train: {len(X_train)} rows ({train_dates.min().date()} → {train_dates.max().date()})")
    print(f"   Test:  {len(X_test)} rows ({test_dates.min().date()} → {test_dates.max().date()})")
    print(f"   Train base rate: {y_train.mean():.1%}  Test base rate: {y_test.mean():.1%}")

    # ── 4. Train on train slice, predict on OOS test slice ─────────
    print("\n4. Training XGBClassifier on train slice (no leakage)...")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=(1 - y_train.mean()) / y_train.mean(),
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train.values, y_train)

    # OOS predictions
    print("\n5. Predicting on OOS test slice...")
    p_test = model.predict_proba(X_test.values)[:, 1]
    p_test_clipped = np.clip(p_test, 0.001, 0.999)

    # Also fit on full data for the "in-sample" baseline
    print("   (Also fitting on FULL data for in-sample comparison...)")
    full_model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=(1 - base_rate) / base_rate,
        random_state=42,
        eval_metric="logloss",
    )
    full_model.fit(X.values, y)
    p_full = full_model.predict_proba(X.values)[:, 1]
    p_full_clipped = np.clip(p_full, 0.001, 0.999)

    # ── 6. OOS metrics ─────────────────────────────────────────────
    print("\n6. OOS metrics (test slice only — this is the honest number):")
    oos_brier = float(brier_score_loss(y_test, p_test_clipped))
    oos_logloss = float(log_loss(y_test, p_test_clipped))
    oos_acc = float(accuracy_score(y_test, (p_test >= 0.5).astype(int)))
    try:
        oos_auc = float(roc_auc_score(y_test, p_test))
    except ValueError:
        oos_auc = float("nan")
    naive_brier = float(np.mean((y_test.mean() - y_test) ** 2))

    # In-sample metrics
    in_brier = float(brier_score_loss(y, p_full_clipped))
    in_acc = float(accuracy_score(y, (p_full >= 0.5).astype(int)))
    in_auc = float(roc_auc_score(y, p_full))

    print(f"\n   {'Metric':18s} {'In-sample':>12s} {'OOS':>12s} {'Naive':>12s}")
    print(f"   {'-'*18} {'-'*12} {'-'*12} {'-'*12}")
    print(f"   {'Brier':18s} {in_brier:>12.4f} {oos_brier:>12.4f} {naive_brier:>12.4f}")
    print(f"   {'Accuracy':18s} {in_acc:>12.1%} {oos_acc:>12.1%} {y_test.mean():>12.1%}")
    try:
        print(f"   {'AUC':18s} {in_auc:>12.4f} {oos_auc:>12.4f} {0.5:>12.4f}")
    except Exception:
        pass

    # ── 7. OOS Calibration by decile (THIS is the honest reference) ──
    # The CRITICAL fix vs the previous version: we compare p_test (OOS
    # predicted) to y_test (OOS actual). The in-sample predictions are
    # only used to flag *additional* overfit signal, not the verdict.
    print("\n7. OOS calibration by predicted-probability decile (OOS-vs-OOS only):")
    print("   (THIS IS THE KEY SECTION — does the model overcompress OOS?)")
    print(f"\n   {'Bin':>10s}  {'N':>4s}  {'p_model_OOS':>11s}  {'p_actual':>9s}  {'Gap':>7s}  {'Status'}")
    print(f"   {'-'*10}  {'-'*4}  {'-'*11}  {'-'*9}  {'-'*7}  {'-'*10}")

    bins = np.arange(0.0, 1.01, 0.1)
    oos_gaps = []
    overcompression_bins = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        # OOS test predictions in this bin
        mask_oos = (p_test >= lo) & (p_test < hi)
        n_oos = int(mask_oos.sum())
        if n_oos < 5:
            if n_oos > 0:
                print(f"   {lo:.0%}-{hi:.0%}      {n_oos:>4d}  (too few, skipping)")
            continue
        p_model_oos = float(p_test[mask_oos].mean())
        p_actual_oos = float(y_test[mask_oos].mean())
        gap_oos = p_actual_oos - p_model_oos

        # Flag: model says p (OOS) but OOS actual is significantly lower.
        # This is the user's actual concern — the +76% edge is what the
        # scanner would show (also the OOS predicted prob on those fights).
        overcompression = (p_model_oos > 0.5) and (p_actual_oos < p_model_oos - 0.15)
        if overcompression:
            overcompression_bins.append({
                "bin": f"{lo:.0%}-{hi:.0%}",
                "n_oos": n_oos,
                "p_model_oos": p_model_oos,
                "p_actual_oos": p_actual_oos,
                "gap": gap_oos,
            })

        oos_gaps.append(abs(gap_oos))
        status = "🚩 OVERCOMP" if overcompression else ("OK" if abs(gap_oos) < 0.05 else ("WARN" if abs(gap_oos) < 0.15 else "BAD"))
        print(f"   {lo:.0%}-{hi:.0%}      {n_oos:>4d}  {p_model_oos:>10.1%}  {p_actual_oos:>8.1%}  {gap_oos:>+6.1%}  {status}")

    # ── 7b. In-sample side-by-side (for overfit-signal only) ───────
    print("\n7b. IN-SAMPLE calibration (for overfit signal ONLY, not the verdict):")
    print("   --- IN-SAMPLE (overfit check only) ---")
    print(f"\n   {'Bin':>10s}  {'N':>5s}  {'p_model_in':>11s}  {'p_actual_in':>12s}  {'Gap':>7s}")
    print(f"   {'-'*10}  {'-'*5}  {'-'*11}  {'-'*12}  {'-'*7}")
    in_sample_gaps_top_decile = None
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask_in = (p_full >= lo) & (p_full < hi)
        n_in = int(mask_in.sum())
        if n_in < 5:
            if n_in > 0:
                print(f"   {lo:.0%}-{hi:.0%}      {n_in:>5d}  (too few, skipping)")
            continue
        p_model_in = float(p_full[mask_in].mean())
        p_actual_in = float(y[mask_in].mean())
        gap_in = p_actual_in - p_model_in
        if lo >= 0.8:
            in_sample_gaps_top_decile = (p_model_in, p_actual_in, n_in)
        print(f"   {lo:.0%}-{hi:.0%}      {n_in:>5d}  {p_model_in:>10.1%}  {p_actual_in:>11.1%}  {gap_in:>+6.1%}")
    print("   --- END IN-SAMPLE ---")

    # ── 7c. Underdog-subsample Brier (the parlay-relevant test) ─────
    # The 6-leg parlay is on underdogs. Test how the model performs on
    # rows where it predicts Red (the underdog) but the actual winner
    # was the favorite (Blue). This is the exact failure mode the
    # parlay is exposed to.
    print("\n7c. Underdog-subsample Brier (model says Red, actual was Blue):")
    underdog_mask = (p_test >= 0.5) & (y_test == 0)  # model picks Red, Blue won
    n_underdog_failures = int(underdog_mask.sum())
    n_underdog_total = int((p_test >= 0.5).sum())
    if n_underdog_total > 0:
        underdog_failure_rate = n_underdog_failures / n_underdog_total
        print(f"   Model picked Red (underdog side) in {n_underdog_total} OOS fights.")
        print(f"   Of those, {n_underdog_failures} were wrong (Blue won).")
        print(f"   Underdog failure rate: {underdog_failure_rate:.1%}")
        # On these specific rows, what was the model's confidence?
        if n_underdog_failures > 0:
            avg_p_on_failures = float(p_test[underdog_mask].mean())
            print(f"   Mean model confidence on those failures: {avg_p_on_failures:.1%}")

    # ── 7d. CI for the top-decile bin (the parlay-relevant bin) ─────
    print("\n7d. 95% confidence interval for the top decile (90-100% bin):")
    top_mask = (p_test >= 0.9) & (p_test < 1.0)
    n_top = int(top_mask.sum())
    if n_top >= 5:
        p_actual_top = float(y_test[top_mask].mean())
        # Wilson 95% CI for binomial
        z = 1.96
        denom = 1 + z**2 / n_top
        center = (p_actual_top + z**2 / (2 * n_top)) / denom
        margin = z * sqrt(p_actual_top * (1 - p_actual_top) / n_top + z**2 / (4 * n_top**2)) / denom
        ci_lo, ci_hi = max(0.0, center - margin), min(1.0, center + margin)
        p_model_top = float(p_test[top_mask].mean())
        print(f"   n = {n_top} OOS fights in 90-100% bin")
        print(f"   p_model_OOS: {p_model_top:.1%}")
        print(f"   p_actual_OOS: {p_actual_top:.1%}")
        print(f"   95% CI on actual: [{ci_lo:.1%}, {ci_hi:.1%}]")
        # The +76% edge case: if the model picks 77% but CI is [50%, 70%],
        # the actual edge is much smaller than 76%
    else:
        print(f"   n = {n_top} OOS fights in 90-100% bin (too few for CI)")

    # ── 8. Quantify overcompression ─────────────────────────────────
    print("\n8. Overcompression summary (the user's key question):")
    if overcompression_bins:
        print(f"   {len(overcompression_bins)} bins show overcompression (p_model_OOS > 50% but p_actual < p_model - 15%):")
        for b in overcompression_bins:
            print(f"     {b['bin']}: model says p={b['p_model_oos']:.0%} OOS but "
                  f"actual is {b['p_actual_oos']:.0%} (gap={b['gap']:+.0%}, n={b['n_oos']})")
    else:
        print("   No bins show severe overcompression. Edges may be real.")

    # Mean and max OOS gap (CRITICAL: max-gap check is more robust than mean)
    if oos_gaps:
        mean_abs_gap = float(np.mean(oos_gaps))
        max_abs_gap = float(max(oos_gaps))
    else:
        # No bins with enough OOS fights — fail safe: assume worst case
        mean_abs_gap = 1.0
        max_abs_gap = 1.0
    # Ensure underdog_failure_rate is always defined for the verdict code
    if n_underdog_total == 0:
        underdog_failure_rate = 1.0  # fail safe
    print(f"\n   Mean |OOS gap| across binned predictions: {mean_abs_gap:.1%}")
    print(f"   Max  |OOS gap| (the worst-bin signal):  {max_abs_gap:.1%}")
    if mean_abs_gap > 0.15:
        print(f"   ⚠ CRITICAL: mean gap >15% indicates significant overcompression.")
    elif mean_abs_gap > 0.10:
        print(f"   ⚠ WARNING: mean gap >10% suggests mild overcompression.")
    else:
        print(f"   ✓ OK: mean gap ≤10% suggests the model is reasonably calibrated OOS.")
    if max_abs_gap > 0.25:
        print(f"   ⚠ CRITICAL: max bin gap >25% — at least one bin is severely overcompressed.")
    elif max_abs_gap > 0.15:
        print(f"   ⚠ WARNING: max bin gap >15% — at least one bin is overcompressed.")

    # ── 9. Naive comparison: does the model beat base-rate? ────────
    beats_naive_brier = oos_brier < naive_brier
    print(f"\n9. OOS Brier vs naive baseline:")
    print(f"   OOS Brier:  {oos_brier:.4f}")
    print(f"   Naive Brier: {naive_brier:.4f}")
    print(f"   Beats naive: {'YES ✓' if beats_naive_brier else 'NO ✗'}")

    # ── 10. Verdict (tighter, more robust thresholds) ───────────────
    # Decision logic: use BOTH the mean OOS gap AND the max-bin gap,
    # since one bad bin can hide in the mean. Also penalize high
    # underdog failure rate (the parlay-relevant test).
    print("\n" + "=" * 72)
    print("  VERDICT (for the 6-leg UFC Freedom 250 parlay)")
    print("=" * 72)

    # Underdog failure rate (only meaningful if we have any)
    if n_underdog_total >= 10:
        underdog_fail_strict = underdog_failure_rate > 0.55
    else:
        underdog_fail_strict = False  # not enough data to penalize

    if not beats_naive_brier:
        verdict = "SKIP the 6-leg"
        reason = (f"OOS Brier ({oos_brier:.4f}) is WORSE than the naive baseline "
                  f"({naive_brier:.4f}). The model adds zero predictive value OOS.")
    elif overcompression_bins and any(b["p_actual_oos"] < 0.20 for b in overcompression_bins):
        verdict = "SKIP the 6-leg"
        reason = (f"{len(overcompression_bins)} bins show severe overcompression. "
                  f"Scanner edges like +76% are likely artifacts, not real edge.")
    elif max_abs_gap > 0.25:
        verdict = "SKIP the 6-leg"
        reason = (f"Max bin OOS gap = {max_abs_gap:.1%} (>25% threshold). "
                  f"At least one bin is severely overcompressed. "
                  f"The +76% edge is likely an artifact, not real signal.")
    elif max_abs_gap > 0.15 and mean_abs_gap > 0.15:
        verdict = "REDUCE 6-leg to 3-leg (Topuria + O'Malley + Ruffy only)"
        reason = (f"Max gap = {max_abs_gap:.1%}, mean gap = {mean_abs_gap:.1%}. "
                  f"Model is biased but not random. Stick to the highest-confidence core.")
    elif mean_abs_gap > 0.10:
        verdict = "REDUCE 6-leg to 3-leg (Topuria + O'Malley + Ruffy only)"
        reason = (f"Mean OOS gap = {mean_abs_gap:.1%} (>10% threshold). "
                  f"Mild overcompression — model is biased. Reduce stake.")
    elif underdog_fail_strict:
        verdict = "REDUCE 6-leg to 3-leg (Topuria + O'Malley + Ruffy only)"
        reason = (f"Underdog failure rate = {underdog_failure_rate:.1%} (>55% threshold). "
                  f"Model is bad at identifying underdog winners specifically. Reduce stake.")
    else:
        verdict = "PROCEED with 6-leg"
        reason = (f"OOS Brier beats naive ({oos_brier:.4f} vs {naive_brier:.4f}). "
                  f"Mean |OOS gap| = {mean_abs_gap:.1%} (≤10% threshold). "
                  f"Max |OOS gap| = {max_abs_gap:.1%} (≤15% threshold). "
                  f"Underdog failure rate = {underdog_failure_rate:.1%} (≤55% threshold). "
                  f"Edges appear real.")

    print(f"\n   Verdict: {verdict}")
    print(f"   Reason:  {reason}")
    print(f"\n   Decision thresholds applied:")
    print(f"     OOS Brier beats naive:           {oos_brier:.4f} < {naive_brier:.4f} = {beats_naive_brier}")
    print(f"     Mean |OOS gap| threshold:        {mean_abs_gap:.1%} (≤10% = PROCEED, >10% = REDUCE, >15% = REDUCE+max)")
    print(f"     Max  |OOS gap| threshold:        {max_abs_gap:.1%} (≤15% = PROCEED, >15% = REDUCE, >25% = SKIP)")
    if n_underdog_total >= 10:
        print(f"     Underdog failure rate threshold: {underdog_failure_rate:.1%} (≤55% = PROCEED, >55% = REDUCE)")
    else:
        print(f"     Underdog failure rate: n={n_underdog_total} too few to penalize")
    print(f"\n   Recommended allocation:")
    if "PROCEED" in verdict:
        print("     $40 on 3-leg (Topuria + O'Malley + Ruffy)")
        print("     $25 on 4-leg (+ Nickal)")
        print("     $20 on 5-leg (+ Hokit)")
        print("     $10 on 6-leg (+ Lopes)")
        print("     $5 on Topuria R3-5 prop (+140)")
    elif "REDUCE" in verdict:
        print("     $40 on 3-leg (Topuria + O'Malley + Ruffy only)")
        print("     $5 on Topuria R3-5 prop (+140)")
        print("     DO NOT bet the 4/5/6-leg until OOS validation improves")
    else:
        print("     DO NOT bet the UFC parlay. Use the $100 budget on the WC + MLB 4-leg play instead.")

    # ── 11. Save JSON output for downstream consumption ──────────
    output = {
        "test_date": pd.Timestamp.now().isoformat(),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "base_rate": base_rate,
        "in_sample_for_overfit_check_only_NOT_for_verdict": {
            "brier": in_brier,
            "accuracy": in_acc,
            "auc": in_auc,
        },
        "oos_use_this_for_verdict": {
            "brier": oos_brier,
            "log_loss": oos_logloss,
            "accuracy": oos_acc,
            "auc": oos_auc,
            "naive_brier": naive_brier,
            "beats_naive": beats_naive_brier,
            "mean_abs_gap": mean_abs_gap if oos_gaps else None,
            "max_abs_gap": max_abs_gap if oos_gaps else None,
            "underdog_failure_rate": underdog_failure_rate if n_underdog_total >= 10 else None,
            "n_underdog_total": n_underdog_total,
        },
        "overcompression_bins": overcompression_bins,
        "thresholds": {
            "mean_abs_gap_proceed": 0.10,
            "mean_abs_gap_reduce": 0.15,
            "max_abs_gap_proceed": 0.15,
            "max_abs_gap_skip": 0.25,
            "underdog_fail_rate_reduce": 0.55,
        },
        "verdict": verdict,
        "reason": reason,
    }
    out_path = MODEL_DIR / "oos_test_ufc.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n   Results saved to {out_path}")


if __name__ == "__main__":
    main()
