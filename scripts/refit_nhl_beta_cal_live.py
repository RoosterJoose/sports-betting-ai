#!/usr/bin/env python3
"""Refit NHL calibrations from live trade tracker outcomes.

Mirrors scripts/refit_mlb_beta_cal_live.py and scripts/refit_nba_beta_cal_live.py
for NHL. Uses actual (model_prob, actual outcome) pairs from resolved trades
in data/trade_tracker.db to fit a fresh BetaCalibrator for each stat. Replaces
the training-test fitted models/nhl/{stat}_beta_cal.json files with live-fitted
versions.

Why this exists:
  The training-test fitted calibrations overfit to in-distribution data and
  don't generalize to live (potentially OOD) inference. NHL currently has
  NO resolved trades in the tracker — this script is forward-looking and
  will refit the moment resolved trades accumulate. The same resolve_paper_trades.py
  result-field bug (commit c65ce9d) that affected NBA/MLB applies to NHL
  too, so once NHL trades start resolving, the same data-quality guard
  (0% WR detection) will fire.

Mirrors the raw-pair query in TradeTracker.get_calibration() (see
src/utils/trade_tracker.py) but returns the raw (model_prob, outcome) pairs
so BetaCalibrator.fit() can use them directly.

Usage:
    python scripts/refit_nhl_beta_cal_live.py                    # refit all
    python scripts/refit_nhl_beta_cal_live.py --min-samples 50   # require more data
    python scripts/refit_nhl_beta_cal_live.py --dry-run          # compute only
    python scripts/refit_nhl_beta_cal_live.py --backup           # save old cal to .bak
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

MODEL_DIR = PROJECT_ROOT / "models" / "nhl"
# NHL scanner (src/scripts/kalshi_nhl_unified.py:114) reads cal files
# from MODEL_DIR root (`{mn}_beta_cal.json`), so write to root too.
CALIB_DIR = MODEL_DIR

# Module-level flag for the magnitude guard. Set by main() from --force-save.
_force_save_global = False

# (trade_tracker.model_name, json_stem, calibrator_type)
# Notes:
#   - NHL model filenames use the literal stat name (uppercase, e.g.
#     "GOALS+ASSISTS" with a literal +). The scanner uses
#     `model_name.lower()` for the cal path, so json_stem mirrors that
#     (lowercase with literal +). On macOS the + is filesystem-safe.
#   - The trade tracker model_name currently mirrors the scanner's
#     model_name (uppercase, no prefix) for NHL — confirmed by inspection
#     of MARKET_TYPES in kalshi_nhl_unified.py:39-90.
#   - All 6 NHL stats are included; if a stat has no resolved trades
#     yet, _refit_one will return a "skipped (DATA QUALITY)" result.
STAT_MAP = [
    ("GOALS",          "goals",          "beta"),
    ("ASSISTS",        "assists",        "beta"),
    ("POINTS",         "points",         "beta"),
    ("SHOTS",          "shots",          "beta"),
    ("PIM",            "pim",            "beta"),
    ("GOALS+ASSISTS",  "goals+assists",  "beta"),
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
    df = tt.get_raw_pairs(sport="nhl", model_name=trade_model_name)
    n = len(df)
    if n < min_samples:
        return {"trade_model_name": trade_model_name, "json_stem": json_stem,
                "cal_type": cal_type, "n": n, "skipped": True,
                "reason": f"n={n} < min_samples={min_samples}"}

    probs = df["model_prob"].values.astype(float)
    outcomes = df["outcome"].astype(int).values
    win_count = int(outcomes.sum())
    win_rate = win_count / n if n else 0.0

    # Degenerate-data guards. For NHL: same data-quality concern as
    # NBA/MLB — if all resolved trades are losses (win_rate=0), no
    # calibrator can be fit. This is a DATA QUALITY issue, not a
    # script bug: model output is being systematically inverted, or
    # the resolution is wrong, or both. Flag loudly.
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
    print("  NHL CALIBRATION LIVE REFIT (BetaCal)")
    print(f"  Source: data/trade_tracker.db  |  Min samples: {args.min_samples}")
    print(f"  Output: {CALIB_DIR}")
    if args.dry_run:
        print("  MODE: dry-run (no files will be written)")
    if args.backup:
        print("  MODE: backup existing calibrations before overwriting")
    print("=" * 72)

    tt = TradeTracker()

    # Pre-flight: show resolved counts for ALL NHL model_names (not just
    # the ones in STAT_MAP) so the user can see other-stats availability.
    # Uses public get_analytics (per-model n/wins/losses/win_rate) so we
    # don't touch the private tt._conn.
    print(f"\n  All NHL model_name counts in trade tracker:")
    analytics = tt.get_analytics(sport="nhl", min_sample=0)
    in_scope_names = {m[0] for m in STAT_MAP}
    if not analytics:
        print("    (no NHL trades in trade tracker yet)")
        # NHL has no trades yet — that's expected. Print scope markers and
        # continue so the dry-run/refit flow still works once trades exist.
        for m in STAT_MAP:
            print(f"    {m[0]:14s}  n={0:>4d}  wins={0:>3d}  WR={0:>4.0%}  ✓ in scope")
    else:
        analytics = sorted(analytics, key=lambda a: -a["n"])
        for a in analytics:
            in_scope = "✓ in scope" if a["model_name"] in in_scope_names else "  (not in scope)"
            print(f"    {a['model_name']:14s}  n={a['n']:>4d}  "
                  f"wins={a['wins']:>3d}  WR={a['win_rate']:>4.0%}  {in_scope}")

    # Aggregate stats
    total_n = sum(a["n"] for a in analytics) if analytics else 0
    total_wins = sum(a["wins"] for a in analytics) if analytics else 0
    total_wr = total_wins / total_n if total_n else 0
    print(f"\n  Resolved NHL trades: {total_n}  Wins: {total_wins}  WR: {total_wr:.1%}")
    if total_n == 0:
        print(f"\n  ℹ No resolved NHL trades yet. Refit will SKIP all stats until trades accumulate.")
        print(f"    (Script is forward-looking — run it again after resolve_paper_trades.py populates results.)")
    elif total_wins == 0:
        print(f"\n  ⚠ CRITICAL: 0 wins across all {total_n} resolved NHL trades.")
        print(f"    No calibrator can be fit to all-zero outcomes.")
        print(f"    Root cause investigation required before calibrating:")
        print(f"      1. Verify resolve_paper_trades.py result-field handling (commit c65ce9d)")
        print(f"      2. Verify the NHL model isn't systematically inverted")
        print(f"      3. Check if Kalshi settlement prices are being read correctly")
        print(f"    Script will SKIP all refits and report 0 calibrations updated.")
    elif total_wr < 0.10:
        print(f"  ⚠ WARNING: very low win rate ({total_wr:.1%}). Refits will be unreliable.")

    print(f"\n  {'Stat':14s} {'N':>4s} {'OldBrier':>8s} {'NewBrier':>8s} {'Δ':>7s}  Status")
    print(f"  {'-'*14} {'-'*4} {'-'*8} {'-'*8} {'-'*7}  ------")

    results = []
    for trade_name, json_stem, cal_type in STAT_MAP:
        if args.dry_run:
            n = tt.count_pairs(sport="nhl", model_name=trade_name)
            print(f"  {trade_name:14s} {n:>4d}  (dry-run, not computed)")
            continue

        result = _refit_one(tt, trade_name, json_stem, cal_type,
                            min_samples=args.min_samples, backup=args.backup)
        if result is None:
            print(f"  {trade_name:14s}      -                       (unexpected None)")
            continue
        if result.get("skipped"):
            print(f"  {result['trade_model_name']:14s} {result['n']:>4d}                       "
                  f"skipped ({result['reason']})")
            continue
        results.append(result)
        delta_icon = "✅" if result["improvement"] > 0 else ("⚠️" if result["improvement"] == 0 else "❌")
        print(f"  {result['trade_model_name']:14s} {result['n']:>4d} "
              f"{result['old_cal_brier']:>8.4f} {result['new_cal_brier']:>8.4f} "
              f"{result['improvement']:>+7.4f}  {delta_icon} → {result['saved_to']}")
        if result.get("backup"):
            print(f"           ↳ backed up to {result['backup']}")

    if args.dry_run:
        print("\n  Dry-run complete. Re-run without --dry-run to write calibrations.")
        return

    if not results:
        print("\n  No stats had enough data to refit.")
        if total_wins == 0 and total_n > 0:
            print("  (See CRITICAL warning above for NHL's 0% WR — fix the data first.)")
        elif total_n == 0:
            print("  (No NHL trades in tracker yet — script is ready for when they accumulate.)")
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
