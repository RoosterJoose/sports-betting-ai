#!/usr/bin/env python3
"""Refit WNBA calibrations from live trade tracker outcomes.

Mirrors scripts/refit_mlb_beta_cal_live.py and scripts/refit_nba_beta_cal_live.py
for WNBA. Uses actual (model_prob, actual outcome) pairs from resolved trades
in data/trade_tracker.db to fit a fresh BetaCalibrator for each stat. Replaces
the training-test fitted models/wnba/{stat}_beta_cal.json files with live-fitted
versions.

Why this exists:
  The training-test fitted calibrations overfit to in-distribution data and
  don't generalize to live (potentially OOD) inference. WNBA has 1,501
  resolved trades and 0% historical win rate (same data-quality signature as
  MLB — strongly suggests the trades were resolved before the
  resolve_paper_trades.py result-field fix in commit c65ce9d was applied).
  Live refit uses actual outcomes from production to produce calibrations
  that match real deployed behavior, but only after the data-quality issue
  is fixed.

Mirrors the raw-pair query in TradeTracker.get_calibration() (see
src/utils/trade_tracker.py) but returns the raw (model_prob, outcome) pairs
so BetaCalibrator.fit() can use them directly.

Usage:
    python scripts/refit_wnba_beta_cal_live.py                    # refit all
    python scripts/refit_wnba_beta_cal_live.py --min-samples 50   # require more data
    python scripts/refit_wnba_beta_cal_live.py --dry-run          # compute only
    python scripts/refit_wnba_beta_cal_live.py --backup           # save old cal to .bak
"""
import sys, json, argparse, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import datetime

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config.settings import PROJECT_ROOT
from src.utils.trade_tracker import TradeTracker
from src.models.calibrator import BetaCalibrator

MODEL_DIR = PROJECT_ROOT / "models" / "wnba"
# WNBA scanner (src/scripts/kalshi_wnba_unified.py:81) reads cal files
# from MODEL_DIR root (`{mn}_beta_cal.json`), so write to root too.
CALIB_DIR = MODEL_DIR

# Module-level flag for the magnitude guard. Set by main() from --force-save.
_force_save_global = False

# (trade_tracker.model_name, json_stem, calibrator_type)
# Notes:
#   - The WNBA trade tracker uses "WNBA-{STAT}" prefixed model names
#     (e.g. "WNBA-PTS", "WNBA-3PT") while the scanner uses the bare
#     stat name (e.g. "PTS", "FG3M"). 3PT trades in the tracker map to
#     the FG3M cal file (Kalshi's 3PT market is for 3-point MAKES).
#   - Only stats that currently have resolved trades in the tracker are
#     included; add more here as the scanner / tracker expand.
STAT_MAP = [
    ("WNBA-PTS",  "pts",   "beta"),
    ("WNBA-REB",  "reb",   "beta"),
    ("WNBA-AST",  "ast",   "beta"),
    ("WNBA-3PT",  "fg3m",  "beta"),
]

# Magnitude guard: BetaCal fits with |a|, |b|, or |c| > this threshold are
# in the "dangerous-magnitude" / overfit regime. Such calibrators systematically
# destroy the model_prob distribution (e.g. NBA pts a=8.06, ast a=-1.62 — live-fitted
# values that crushed/scattered predictions). The guard refuses to write them
# and preserves the old cal file. Override with --force-save.
MAX_PARAM_MAGNITUDE = 3.0


def _check_betacal_magnitude(bc: BetaCalibrator, trade_model_name: str,
                             force_save: bool = False):
    """Return None if calibrator parameters are safe, else return a skip-result dict.

    The magnitude guard prevents overfit calibrations (|a|>3, |b|>3, |c|>3) from
    silently overwriting the old cal file.

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


def _refit_one(tt: TradeTracker, trade_model_name: str, json_stem: str,
               cal_type: str, min_samples: int, backup: bool):
    """Refit a single BetaCalibrator for one stat. Returns one result dict."""
    df = tt.get_raw_pairs(sport="wnba", model_name=trade_model_name)
    n = len(df)
    if n < min_samples:
        return {"trade_model_name": trade_model_name, "json_stem": json_stem,
                "cal_type": cal_type, "n": n, "skipped": True,
                "reason": f"n={n} < min_samples={min_samples}"}

    probs = df["model_prob"].values.astype(float)
    outcomes = df["outcome"].astype(int).values
    win_count = int(outcomes.sum())
    win_rate = win_count / n if n else 0.0

    # Degenerate-data guards. For WNBA specifically: if all resolved
    # trades are losses (win_rate=0), no calibrator can be fit — the
    # target is constant. This is a DATA QUALITY issue, not a script
    # bug: model output is being systematically inverted, or the
    # resolution is wrong, or both. Flag loudly.
    if np.std(probs) == 0:
        return {"trade_model_name": trade_model_name, "json_stem": json_stem,
                "cal_type": cal_type, "n": n, "skipped": True,
                "reason": "degenerate (constant model_prob)"}
    if win_count == 0:
        return {"trade_model_name": trade_model_name, "json_stem": json_stem,
                "cal_type": cal_type, "n": n, "skipped": True,
                "reason": f"DATA QUALITY: 0/{n} wins ({win_rate:.0%}). "
                           f"Cannot fit calibrator to all-zero outcomes. "
                           f"Investigate model/resolution before calibrating."}
    if win_count == n:
        return {"trade_model_name": trade_model_name, "json_stem": json_stem,
                "cal_type": cal_type, "n": n, "skipped": True,
                "reason": f"degenerate (all {n} outcomes are wins — trivial calibrator)"}

    # Fit the new calibrator on live data.
    # Catch specific exceptions only — the degenerate-data early returns above
    # cover the common failure modes; a bare `except Exception` would mask
    # real bugs (typos, missing columns) and just print them in a "skipped" line.
    bc = BetaCalibrator()
    try:
        bc.fit(probs, outcomes)
    except (ValueError, np.linalg.LinAlgError, RuntimeError) as e:
        return {"trade_model_name": trade_model_name, "json_stem": json_stem,
                "cal_type": cal_type, "n": n, "skipped": True,
                "reason": f"fit failed: {type(e).__name__}: {e}"}

    # Magnitude guard: refuse to save dangerous-magnitude calibrations.
    magnitude_skip = _check_betacal_magnitude(bc, trade_model_name,
                                              force_save=_force_save_global)
    if magnitude_skip is not None:
        return {"trade_model_name": trade_model_name, "json_stem": json_stem,
                "cal_type": cal_type, "n": n, **magnitude_skip}

    # Brier before/after
    old_cal_brier = float(np.mean((probs - outcomes) ** 2))
    cal_probs = bc(probs)
    new_cal_brier = float(np.mean((cal_probs - outcomes) ** 2))

    cal_path = CALIB_DIR / f"{json_stem}_{cal_type}_cal.json"

    # Optionally back up the old calibration
    backup_info = None
    if backup and cal_path.exists():
        backup_path = cal_path.with_suffix(
            f".{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.bak.json"
        )
        backup_path.write_text(cal_path.read_text())
        backup_info = str(backup_path)

    bc.save(cal_path)

    # Per-decile calibration table
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

    return {
        "trade_model_name": trade_model_name,
        "json_stem": json_stem,
        "cal_type": cal_type,
        "n": int(n),
        "old_cal_brier": round(old_cal_brier, 4),
        "new_cal_brier": round(new_cal_brier, 4),
        "improvement": round(old_cal_brier - new_cal_brier, 4),
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
                        help="Save existing cal JSONs to timestamped .bak before overwriting")
    parser.add_argument("--force-save", action="store_true",
                        help="Bypass the magnitude guard and save calibrations with "
                             "|a|/|b|/|c| > %.1f anyway (DANGEROUS, rarely needed)" % MAX_PARAM_MAGNITUDE)
    args = parser.parse_args()
    global _force_save_global
    _force_save_global = args.force_save

    print("=" * 72)
    print("  WNBA CALIBRATION LIVE REFIT (BetaCal)")
    print(f"  Source: data/trade_tracker.db  |  Min samples: {args.min_samples}")
    print(f"  Output: {CALIB_DIR}")
    if args.dry_run:
        print("  MODE: dry-run (no files will be written)")
    if args.backup:
        print("  MODE: backup existing calibrations before overwriting")
    print("=" * 72)

    tt = TradeTracker()

    # Pre-flight: show resolved counts for ALL WNBA model_names (not just
    # the ones in STAT_MAP) so the user can see other-stats availability.
    # Uses public get_analytics (per-model n/wins/losses/win_rate) so we
    # don't touch the private tt._conn.
    print(f"\n  All WNBA model_name counts in trade tracker:")
    analytics = tt.get_analytics(sport="wnba", min_sample=0)
    if not analytics:
        print("\n  No resolved WNBA trades found. Run resolve_paper_trades.py first.")
        return
    # Sort by n descending for stable display
    analytics = sorted(analytics, key=lambda a: -a["n"])
    in_scope_names = {m[0] for m in STAT_MAP}
    for a in analytics:
        in_scope = "✓ in scope" if a["model_name"] in in_scope_names else "  (not in scope)"
        print(f"    {a['model_name']:12s}  n={a['n']:>4d}  "
              f"wins={a['wins']:>3d}  WR={a['win_rate']:>4.0%}  {in_scope}")

    # Aggregate stats
    total_n = sum(a["n"] for a in analytics)
    total_wins = sum(a["wins"] for a in analytics)
    total_wr = total_wins / total_n if total_n else 0
    print(f"\n  Resolved WNBA trades: {total_n}  Wins: {total_wins}  WR: {total_wr:.1%}")
    if total_wins == 0:
        print(f"\n  ⚠ CRITICAL: 0 wins across all {total_n} resolved WNBA trades.")
        print(f"    No calibrator can be fit to all-zero outcomes.")
        print(f"    Root cause investigation required before calibrating:")
        print(f"      1. Verify resolve_paper_trades.py result-field handling (commit c65ce9d)")
        print(f"      2. Verify the WNBA model isn't systematically inverted")
        print(f"      3. Check if Kalshi settlement prices are being read correctly")
        print(f"    Script will SKIP all refits and report 0 calibrations updated.")
        print()
    elif total_wr < 0.10:
        print(f"  ⚠ WARNING: very low win rate ({total_wr:.1%}). Refits will be unreliable.")

    print(f"\n  {'Stat':12s} {'N':>4s} {'OldBrier':>8s} {'NewBrier':>8s} {'Δ':>7s}  Status")
    print(f"  {'-'*12} {'-'*4} {'-'*8} {'-'*8} {'-'*7}  ------")

    results = []
    for trade_name, json_stem, cal_type in STAT_MAP:
        if args.dry_run:
            n = tt.count_pairs(sport="wnba", model_name=trade_name)
            print(f"  {trade_name:12s} {n:>4d}  (dry-run, not computed)")
            continue

        result = _refit_one(tt, trade_name, json_stem, cal_type,
                            min_samples=args.min_samples, backup=args.backup)
        if result is None:
            print(f"  {trade_name:12s}      -                       (unexpected None)")
            continue
        if result.get("skipped"):
            print(f"  {result['trade_model_name']:12s} {result['n']:>4d}                       "
                  f"skipped ({result['reason']})")
            continue
        results.append(result)
        delta_icon = "✅" if result["improvement"] > 0 else ("⚠️" if result["improvement"] == 0 else "❌")
        print(f"  {result['trade_model_name']:12s} {result['n']:>4d} "
              f"{result['old_cal_brier']:>8.4f} {result['new_cal_brier']:>8.4f} "
              f"{result['improvement']:>+7.4f}  {delta_icon} → {result['saved_to']}")
        if result.get("backup"):
            print(f"           ↳ backed up to {result['backup']}")

    if args.dry_run:
        print("\n  Dry-run complete. Re-run without --dry-run to write calibrations.")
        return

    if not results:
        print("\n  No stats had enough data to refit.")
        print("  (See CRITICAL warning above for WNBA's 0% WR — fix the data first.)")
        return

    # Summary
    n_improved = sum(1 for r in results if r["improvement"] > 0)
    n_unchanged = sum(1 for r in results if r["improvement"] == 0)
    n_worse = sum(1 for r in results if r["improvement"] < 0)
    avg_improvement = float(np.mean([r["improvement"] for r in results]))

    print(f"\n  {'='*66}")
    print(f"  Summary: {len(results)} calibrators refit")
    print(f"    Improved:    {n_improved}")
    print(f"    Unchanged:   {n_unchanged}")
    print(f"    Worse:       {n_worse}")
    print(f"    Avg Δ Brier: {avg_improvement:+.4f} (positive = better re-calibration)")
    print(f"  {'='*66}")

    # Per-decile calibration tables
    if results:
        print("\n  Per-decile calibration tables (old-cal prob vs actual outcome):\n")
        for r in sorted(results, key=lambda x: -x["n"])[:5]:
            print(f"  --- {r['trade_model_name']} (n={r['n']}) ---")
            print(f"    {'Bucket':>10s}  {'N':>4s}  {'Pred':>5s}  {'Actual':>6s}  {'Gap':>6s}")
            for row in r["calibration_table"]:
                gap = row["actual"] - row["pred"]
                print(f"    {row['lo']:.2f}-{row['hi']:.2f}  {row['n']:>4d}  "
                      f"{row['pred']:>5.0%}  {row['actual']:>6.0%}  {gap:>+6.0%}")
            print()


if __name__ == "__main__":
    main()
