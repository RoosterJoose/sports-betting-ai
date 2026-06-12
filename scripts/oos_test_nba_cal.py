#!/usr/bin/env python3
"""Proper out-of-sample (OOS) test of the new live-fitted NBA calibrations.

For each stat, splits resolved NBA trades chronologically into:
  - Train: first 80% (oldest)
  - Test:  last 20% (most recent)

Then evaluates Brier score on the holdout for three calibrators:
  1. Identity   — no calibration (baseline)
  2. Old cal    — training-test fitted (from .bak.json backups)
  3. New cal    — refit on the 80% train split (what the live refit would produce)

A real improvement is one where the NEW cal beats the OLD cal on the OOS
holdout (not just in-sample). The in-sample +0.48 Brier improvement from
scripts/refit_nba_beta_cal_live.py was suspect because the calibrator was
fit and measured on the same data. This script measures the proper OOS
improvement.

Usage:
    python scripts/oos_test_nba_cal.py                  # default 80/20 split
    python scripts/oos_test_nba_cal.py --holdout 0.3   # 70/30 split
    python scripts/oos_test_nba_cal.py --min-train 50  # require more train data
"""
import sys, json, argparse, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config.settings import PROJECT_ROOT
from src.utils.trade_tracker import TradeTracker
from src.models.calibrator import BetaCalibrator

MODEL_DIR = PROJECT_ROOT / "models" / "nba"

STAT_MAP = {
    "PTS": "pts",
    "REB": "reb",
    "AST": "ast",
    "BLK": "blk",
    "STL": "stl",
    "3PT": "fg3m",
    "FTM": "ftm",
    "PR":  "pr",
    "PA":  "pa",
    "RA":  "ra",
    "PRA": "pra",
}


def _brier(probs, outcomes):
    return float(np.mean((np.asarray(probs) - np.asarray(outcomes)) ** 2))


def _find_old_cal_path(json_name: str):
    """Find the oldest .bak.json for this stat (the original training-test cal).

    Filename format from refit_nba_beta_cal_live.py with --backup is
    '{stat}_beta_cal.YYYYMMDD_HHMMSS_ffffff.bak.json'. Sorting by filename
    sorts by timestamp, so the first match is the oldest.
    """
    pattern = f"{json_name}_beta_cal.*.bak.json"
    candidates = sorted(MODEL_DIR.glob(pattern))
    if candidates:
        return candidates[0], "bak (oldest)"
    # No .bak.json — fall back to the current file ONLY for informational
    # display; the caller must NOT use its brier as a comparison baseline
    # because the current file is the live-fitted cal (same as "new").
    current = MODEL_DIR / f"{json_name}_beta_cal.json"
    if current.exists():
        return current, "current (no .bak — NOT usable as baseline)"
    return None, None


def _load_pairs_with_timestamp(tt: TradeTracker, model_name: str) -> pd.DataFrame:
    """Load resolved NBA trades for a stat, sorted by timestamp ASC.

    Uses tt._conn directly because TradeTracker.get_raw_pairs() intentionally
    does not expose timestamps (it returns (model_prob, outcome) only).
    """
    return pd.read_sql_query(
        "SELECT model_prob, "
        "CASE WHEN status='won' THEN 1.0 ELSE 0.0 END as outcome, "
        "timestamp "
        "FROM trades "
        "WHERE sport='nba' AND model_name=? AND status IN ('won','lost') "
        "ORDER BY timestamp ASC",
        tt._conn, params=[model_name],
    )


def _evaluate_stat(tt: TradeTracker, model_name: str, json_name: str,
                   holdout_frac: float, min_train: int, min_test: int):
    """Run the OOS test for one stat. Returns a result dict or None if skipped."""
    df = _load_pairs_with_timestamp(tt, model_name)
    if df.empty:
        return {"model_name": model_name, "skipped": True, "reason": "no resolved trades"}

    n_total = len(df)
    n_train = int(n_total * (1 - holdout_frac))
    n_test = n_total - n_train
    if n_train < min_train:
        return {"model_name": model_name, "n_total": n_total, "skipped": True,
                "reason": f"n_train={n_train} < min_train={min_train}"}
    if n_test < min_test:
        return {"model_name": model_name, "n_total": n_total, "skipped": True,
                "reason": f"n_test={n_test} < min_test={min_test}"}

    train_probs = df["model_prob"].values[:n_train].astype(float)
    train_outcomes = df["outcome"].values[:n_train].astype(int)
    test_probs = df["model_prob"].values[n_train:].astype(float)
    test_outcomes = df["outcome"].values[n_train:].astype(int)

    # Degenerate-data guards on the train set
    if np.std(train_probs) == 0:
        return {"model_name": model_name, "n_total": n_total, "skipped": True,
                "reason": "train has constant model_prob"}
    if train_outcomes.sum() == 0 or train_outcomes.sum() == len(train_outcomes):
        return {"model_name": model_name, "n_total": n_total, "skipped": True,
                "reason": "train outcomes are all-0 or all-1"}

    # 1. Identity baseline — no calibration at all
    brier_identity = _brier(test_probs, test_outcomes)

    # 2. Old cal — from .bak.json (the training-test cal that was in production
    #    before any live refit ran). If no .bak exists, we cannot form a
    #    meaningful "old vs new" comparison (the current file IS the live-fit),
    #    so we leave brier_old = None and the verdict will read "n/a".
    old_cal_path, old_cal_src = _find_old_cal_path(json_name)
    brier_old = None
    if old_cal_path is not None and "NOT usable" not in old_cal_src:
        try:
            old_cal = BetaCalibrator.load(old_cal_path)
            brier_old = _brier(old_cal.calibrate(test_probs), test_outcomes)
        except (json.JSONDecodeError, KeyError, ValueError, OSError) as e:
            brier_old = None
            old_cal_src = f"{old_cal_src} (load failed: {type(e).__name__}: {e})"

    # 3. New cal — refit on the 80% train split
    new_cal = BetaCalibrator()
    try:
        new_cal.fit(train_probs, train_outcomes)
    except (ValueError, np.linalg.LinAlgError, RuntimeError) as e:
        return {"model_name": model_name, "n_total": n_total, "skipped": True,
                "reason": f"new cal fit failed: {type(e).__name__}: {e}"}
    brier_new = _brier(new_cal.calibrate(test_probs), test_outcomes)

    # Verdict
    if brier_old is None:
        verdict = "n/a (no old cal)"
    else:
        delta = brier_old - brier_new
        if delta > 0.001:
            verdict = "✅ NEW better OOS"
        elif delta < -0.001:
            verdict = "❌ OLD better OOS"
        else:
            verdict = "≈ TIE"

    return {
        "model_name": model_name,
        "json_name": json_name,
        "n_total": n_total,
        "n_train": n_train,
        "n_test": n_test,
        "train_period": f"{df['timestamp'].iloc[0][:10]} → {df['timestamp'].iloc[n_train-1][:10]}",
        "test_period": f"{df['timestamp'].iloc[n_train][:10]} → {df['timestamp'].iloc[-1][:10]}",
        "brier_identity": round(brier_identity, 4),
        "brier_old": round(brier_old, 4) if brier_old is not None else None,
        "brier_new": round(brier_new, 4),
        "improve_vs_identity": round(brier_identity - brier_new, 4),
        "improve_vs_old": round(brier_old - brier_new, 4) if brier_old is not None else None,
        "new_a": round(new_cal.a, 4),
        "new_b": round(new_cal.b, 4),
        "new_c": round(new_cal.c, 4),
        "old_cal_source": old_cal_src,
        "verdict": verdict,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--holdout", type=float, default=0.20,
                        help="Holdout fraction (default 0.20 = 80/20 split)")
    parser.add_argument("--min-train", type=int, default=30,
                        help="Min resolved trades in train (default 30)")
    parser.add_argument("--min-test", type=int, default=10,
                        help="Min resolved trades in holdout (default 10)")
    args = parser.parse_args()

    print("=" * 78)
    print("  NBA CALIBRATION — OUT-OF-SAMPLE TEST (chronological 80/20 split)")
    print(f"  Holdout: {args.holdout:.0%}  |  Min train: {args.min_train}  |  Min test: {args.min_test}")
    print("=" * 78)

    tt = TradeTracker()

    # Quick data overview
    total_n = tt.count_pairs(sport="nba")
    print(f"\n  Total resolved NBA trades: {total_n}")

    if total_n == 0:
        print("  No resolved NBA trades. Run resolve_paper_trades.py first.")
        return

    print(f"\n  {'Stat':6s} {'Ntr':>4s} {'Nte':>4s} "
          f"{'BrierId':>8s} {'BrierOld':>9s} {'BrierNew':>9s} "
          f"{'ΔOld':>7s}  Verdict")
    print(f"  {'-'*6} {'-'*4} {'-'*4} "
          f"{'-'*8} {'-'*9} {'-'*9} "
          f"{'-'*7}  -------")

    results = []
    for model_name, json_name in STAT_MAP.items():
        result = _evaluate_stat(tt, model_name, json_name,
                                holdout_frac=args.holdout,
                                min_train=args.min_train,
                                min_test=args.min_test)
        if result is None:
            continue
        if result.get("skipped"):
            n_disp = result.get("n_total", 0)
            print(f"  {model_name:6s}      -                       "
                  f"skipped ({result['reason']})")
            continue
        results.append(result)
        b_old = result["brier_old"]
        b_old_disp = f"{b_old:>9.4f}" if b_old is not None else f"{'n/a':>9s}"
        d_old = result["improve_vs_old"]
        d_old_disp = f"{d_old:>+7.4f}" if d_old is not None else f"{'n/a':>7s}"
        print(f"  {result['model_name']:6s} {result['n_train']:>4d} {result['n_test']:>4d} "
              f"{result['brier_identity']:>8.4f} {b_old_disp} {result['brier_new']:>9.4f} "
              f"{d_old_disp}  {result['verdict']}")

    if not results:
        print("\n  No stats had enough data to evaluate.")
        return

    # Summary stats
    n_with_old = [r for r in results if r["brier_old"] is not None]
    n_new_better = sum(1 for r in n_with_old if r["improve_vs_old"] > 0.001)
    n_old_better = sum(1 for r in n_with_old if r["improve_vs_old"] < -0.001)
    n_tie = sum(1 for r in n_with_old if abs(r["improve_vs_old"]) <= 0.001)
    avg_improve_vs_old = (float(np.mean([r["improve_vs_old"] for r in n_with_old]))
                          if n_with_old else None)
    avg_improve_vs_identity = float(np.mean([r["improve_vs_identity"] for r in results]))

    print(f"\n  {'='*70}")
    print(f"  Summary: {len(results)} stats evaluated "
          f"({len(n_with_old)} have old cal for comparison)")
    if n_with_old:
        print(f"    New cal better OOS:    {n_new_better}")
        print(f"    Old cal better OOS:    {n_old_better}")
        print(f"    Tied:                  {n_tie}")
        print(f"    Avg Δ Brier (new-old): {avg_improve_vs_old:+.4f}  "
              f"(positive = new cal better)")
    print(f"    Avg Δ vs identity:     {avg_improve_vs_identity:+.4f}  "
          f"(positive = any calibration helps)")
    print(f"  {'='*70}")

    # Detailed table for the stats we evaluated
    print(f"\n  Per-stat details:\n")
    for r in sorted(results, key=lambda x: -x["n_total"]):
        print(f"  --- {r['model_name']} (N_total={r['n_total']}, "
              f"N_train={r['n_train']}, N_test={r['n_test']}) ---")
        print(f"      Train period: {r['train_period']}")
        print(f"      Test period:  {r['test_period']}")
        print(f"      Old cal src:  {r['old_cal_source']}")
        print(f"      New cal:      a={r['new_a']:+.4f}  b={r['new_b']:+.4f}  c={r['new_c']:+.4f}")
        print(f"      Brier identity: {r['brier_identity']:.4f}")
        if r["brier_old"] is not None:
            print(f"      Brier old cal:  {r['brier_old']:.4f}  (Δ new vs old: {r['improve_vs_old']:+.4f})")
        print(f"      Brier new cal:  {r['brier_new']:.4f}  (Δ new vs identity: {r['improve_vs_identity']:+.4f})")
        print(f"      Verdict:        {r['verdict']}")
        print()


if __name__ == "__main__":
    main()
