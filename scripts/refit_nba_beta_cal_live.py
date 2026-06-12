#!/usr/bin/env python3
"""Refit NBA BetaCalibrator from live trade tracker outcomes.

Uses the actual (model_prob, actual outcome) pairs from resolved trades in
data/trade_tracker.db to fit a fresh BetaCalibrator. Overwrites the existing
training-test fitted models/nba/{stat}_beta_cal.json files with live-fitted
versions.

Why this exists:
  The training-test fitted calibrations overfit to in-distribution data and
  don't generalize to live (potentially OOD) inference — a known cause of
  the 30-80% probability overconfidence bug (predicted 60-70% → actual 27%).
  Live refit uses actual outcomes from production to produce a calibration
  that matches real deployed behavior.

Mirrors the raw-pair query in TradeTracker.get_calibration() (see
src/utils/trade_tracker.py:152-167) but returns the raw (model_prob, outcome)
pairs so BetaCalibrator.fit() can use them directly.

Usage:
    python scripts/refit_nba_beta_cal_live.py                    # refit all
    python scripts/refit_nba_beta_cal_live.py --min-samples 50   # require more data
    python scripts/refit_nba_beta_cal_live.py --dry-run          # compute only
    python scripts/refit_nba_beta_cal_live.py --backup           # save old cal to .bak
"""
import sys, json, argparse, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config.settings import PROJECT_ROOT
from src.utils.trade_tracker import TradeTracker
from src.models.calibrator import BetaCalibrator

MODEL_DIR = PROJECT_ROOT / "models" / "nba"

# Module-level flag for the magnitude guard. Set by main() from --force-save.
# Used by _check_betacal_magnitude() when called from _refit_stat().
_force_save_global = False

# Maps trade_tracker.model_name → JSON filename stem (e.g. "ast_beta_cal.json")
# NOTE: Kalshi's "3PT" market is for 3-point MAKES (FG3M), not attempts
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

# Magnitude guard: BetaCal fits with |a|, |b|, or |c| > this threshold are
# in the "dangerous-magnitude" / overfit regime. Such calibrators systematically
# destroy the model_prob distribution (e.g. NBA pts a=8.06, ast a=-1.62 — live-fitted
# values that crushed/scattered predictions and made the scanner produce negative
# edges on every pick). The guard refuses to write them and preserves the old
# cal file. Override with --force-save.
MAX_PARAM_MAGNITUDE = 3.0


def _check_betacal_magnitude(bc: BetaCalibrator, trade_model_name: str,
                             force_save: bool = False):
    """Return None if calibrator parameters are safe, else return a skip-result dict.

    The magnitude guard prevents overfit calibrations (|a|>3, |b|>3, |c|>3) from
    silently overwriting the old cal file. The scanner would then load these
    dangerous cals and produce junk model_prob distributions (same bug class
    as the NBA pts a=8.06 / ast a=-1.62 overcompression diagnosed June 11).

    Args:
        bc: a fitted BetaCalibrator
        trade_model_name: for logging
        force_save: if True, skip the check (user override via --force-save)

    Returns:
        None if safe to save, or a dict with skipped=True if rejected.
    """
    if force_save:
        return None
    if (abs(bc.a) > MAX_PARAM_MAGNITUDE or
        abs(bc.b) > MAX_PARAM_MAGNITUDE or
        abs(bc.c) > MAX_PARAM_MAGNITUDE):
        msg = (f"DANGEROUS MAGNITUDE: a={bc.a:.4f}, b={bc.b:.4f}, c={bc.c:.4f} "
               f"(max |a|,|b|,|c| \u2264 {MAX_PARAM_MAGNITUDE})")
        print(f"\n  \u26a0\u26a0\u26a0 CRITICAL: {trade_model_name} \u2014 {msg} \u26a0\u26a0\u26a0")
        print(f"    Refusing to save. Previous cal file PRESERVED (not overwritten).")
        print(f"    These parameters are in the overfit regime and would")
        print(f"    systematically destroy the model_prob distribution.")
        print(f"    Override with --force-save to bypass this guard.", file=sys.stderr)
        return {"skipped": True, "reason": msg, "a": round(bc.a, 4),
                "b": round(bc.b, 4), "c": round(bc.c, 4)}
    return None


def _refit_stat(tt: TradeTracker, model_name: str, json_name: str,
                min_samples: int, backup: bool):
    """Refit BetaCalibrator for one stat using live trade outcomes.

    Returns a dict with fit diagnostics, or None if skipped.

    # probs here are POST-calibration — see get_raw_pairs() docstring for full
    # context on why this is a re-calibration on top of the existing one.
    """
    df = tt.get_raw_pairs(sport="nba", model_name=model_name)
    n = len(df)
    if n < min_samples:
        return {"model_name": model_name, "n": n, "skipped": True,
                "reason": f"n={n} < min_samples={min_samples}"}

    probs = df["model_prob"].values.astype(float)
    outcomes = df["outcome"].astype(int).values

    # Degenerate-data guards: identical probs, all-0, or all-1 outcomes
    # will cause the logistic regression inside BetaCalibrator.fit() to fail.
    if np.std(probs) == 0:
        return {"model_name": model_name, "n": n, "skipped": True,
                "reason": "degenerate (constant model_prob)"}
    if outcomes.sum() == 0 or outcomes.sum() == len(outcomes):
        return {"model_name": model_name, "n": n, "skipped": True,
                "reason": "degenerate (all outcomes identical)"}

    # Fit the new calibrator on live data.
    # Catch specific exceptions only — the degenerate-data early returns above
    # cover the common failure modes; a bare `except Exception` would mask
    # real bugs (typos, missing columns) and just print them in a "skipped" line.
    bc = BetaCalibrator()
    try:
        bc.fit(probs, outcomes)
    except (ValueError, np.linalg.LinAlgError, RuntimeError) as e:
        return {"model_name": model_name, "n": n, "skipped": True,
                "reason": f"fit failed: {type(e).__name__}: {e}"}

    # Magnitude guard: refuse to save dangerous-magnitude calibrations.
    magnitude_skip = _check_betacal_magnitude(bc, model_name,
                                              force_save=_force_save_global)
    if magnitude_skip is not None:
        return {"model_name": model_name, "n": n, **magnitude_skip}

    # Diagnostics: Brier before vs after.
    # Note: both are POST-calibration Brier scores — ``probs`` already has
    # the old BetaCal applied, ``cal_probs`` has the new one.
    old_cal_brier = float(np.mean((probs - outcomes) ** 2))
    cal_probs = bc.calibrate(probs)
    new_cal_brier = float(np.mean((cal_probs - outcomes) ** 2))

    # Per-decile calibration table (old-cal prob vs actual outcome)
    deciles = np.linspace(0.1, 0.9, 9)
    table = []
    for lo, hi in zip([0.0] + list(deciles), list(deciles) + [1.01]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() >= 5:
            table.append({
                "lo": round(float(lo), 2), "hi": round(float(hi), 2),
                "n": int(mask.sum()),
                "pred": round(float(probs[mask].mean()), 3),
                "actual": round(float(outcomes[mask].mean()), 3),
            })

    cal_path = MODEL_DIR / f"{json_name}_beta_cal.json"

    # Optionally back up the old calibration.
    # %f (microseconds) prevents collision if the script is run twice
    # within the same second.
    if backup and cal_path.exists():
        backup_path = cal_path.with_suffix(
            f".{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.bak.json"
        )
        backup_path.write_text(cal_path.read_text())
        backup_info = str(backup_path)
    else:
        backup_info = None

    # Save the new calibration
    bc.save(cal_path)

    return {
        "model_name": model_name,
        "json_name": json_name,
        "n": int(n),
        "old_cal_brier": round(old_cal_brier, 4),
        "new_cal_brier": round(new_cal_brier, 4),
        "improvement": round(old_cal_brier - new_cal_brier, 4),
        "a": round(bc.a, 4),
        "b": round(bc.b, 4),
        "c": round(bc.c, 4),
        "saved_to": str(cal_path),
        "backup": backup_info,
        "calibration_table": table,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-samples", type=int, default=30,
                        help="Min resolved trades required to refit (default: 30)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute new calibrations but don't write to disk")
    parser.add_argument("--backup", action="store_true",
                        help="Save existing *_beta_cal.json to timestamped .bak before overwriting")
    parser.add_argument("--force-save", action="store_true",
                        help="Bypass the magnitude guard and save calibrations with "
                             "|a|/|b|/|c| > %.1f anyway (DANGEROUS, rarely needed)" % MAX_PARAM_MAGNITUDE)
    args = parser.parse_args()
    global _force_save_global
    _force_save_global = args.force_save

    print("=" * 72)
    print("  NBA BETA-CALIBRATION LIVE REFIT")
    print(f"  Source: data/trade_tracker.db  |  Min samples: {args.min_samples}")
    if args.dry_run:
        print("  MODE: dry-run (no files will be written)")
    if args.backup:
        print("  MODE: backup existing calibrations before overwriting")
    print("=" * 72)

    tt = TradeTracker()

    # Pre-flight: count available resolved NBA trades
    total_n = tt.count_pairs(sport="nba")
    print(f"\n  Resolved NBA trades in tracker: {total_n}")

    if total_n == 0:
        print("\n  No resolved NBA trades found. Run resolve_paper_trades.py first.")
        return

    print(f"\n  {'Stat':6s} {'N':>4s} {'OldCalBrier':>11s} {'NewCalBrier':>11s} {'Δ':>8s}  Status")
    print(f"  {'-'*6} {'-'*4} {'-'*11} {'-'*11} {'-'*8}  ------")

    results = []
    for model_name, json_name in STAT_MAP.items():
        if args.dry_run:
            # In dry-run, peek at the count but don't save
            n = tt.count_pairs(sport="nba", model_name=model_name)
            print(f"  {model_name:6s} {n:>4d}  (dry-run, not computed)")
            continue

        result = _refit_stat(tt, model_name, json_name,
                             min_samples=args.min_samples, backup=args.backup)
        if result is None:
            print(f"  {model_name:6s}      -                       (unexpected None)")
            continue
        if result.get("skipped"):
            print(f"  {result['model_name']:6s} {result['n']:>4d}                       "
                  f"skipped ({result['reason']})")
            continue
        results.append(result)
        delta_icon = "✅" if result["improvement"] > 0 else ("⚠️" if result["improvement"] == 0 else "❌")
        print(f"  {result['model_name']:6s} {result['n']:>4d} "
              f"{result['old_cal_brier']:>11.4f} {result['new_cal_brier']:>11.4f} "
              f"{result['improvement']:>+8.4f}  {delta_icon} → {result['saved_to']}")
        if result.get("backup"):
            print(f"           ↳ backed up to {result['backup']}")

    if args.dry_run:
        print("\n  Dry-run complete. Re-run without --dry-run to write calibrations.")
        return

    if not results:
        print("\n  No stats had enough data to refit.")
        return

    # Summary
    n_improved = sum(1 for r in results if r["improvement"] > 0)
    n_unchanged = sum(1 for r in results if r["improvement"] == 0)
    n_worse = sum(1 for r in results if r["improvement"] < 0)
    avg_improvement = float(np.mean([r["improvement"] for r in results]))

    print(f"\n  {'='*66}")
    print(f"  Summary: {len(results)} stats refit")
    print(f"    Improved:    {n_improved}")
    print(f"    Unchanged:   {n_unchanged}")
    print(f"    Worse:       {n_worse}")
    print(f"    Avg Δ Brier: {avg_improvement:+.4f} (positive = better re-calibration)")
    print(f"  {'='*66}")

    # Print calibration table for the best/worst fits
    if results:
        print("\n  Per-decile calibration tables (old-cal prob vs actual outcome):\n")
        for r in sorted(results, key=lambda x: -x["n"])[:5]:
            print(f"  --- {r['model_name']} (n={r['n']}) ---")
            print(f"    {'Bucket':>10s}  {'N':>4s}  {'Pred':>5s}  {'Actual':>6s}  {'Gap':>6s}")
            for row in r["calibration_table"]:
                gap = row["actual"] - row["pred"]
                print(f"    {row['lo']:.2f}-{row['hi']:.2f}  {row['n']:>4d}  "
                      f"{row['pred']:>5.0%}  {row['actual']:>6.0%}  {gap:>+6.0%}")
            print()


if __name__ == "__main__":
    main()
