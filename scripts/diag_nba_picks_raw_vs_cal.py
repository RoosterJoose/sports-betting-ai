#!/usr/bin/env python3
"""Sample today's NBA picks and show raw vs calibrated probs.

For a stratified sample of 10 picks, computes:
  - raw p (distribution-based, no calibrator)
  - cal p (with the new live-fitted calibrator — what's in models/nba/*_beta_cal.json)
  - old-cal p (with the training-test fitted calibrator — from .bak.json)
  - market p (the Kalshi mid)
  - edges vs market for each

A pick is a "calibrator artifact" if the new cal makes it look attractive
(positive edge vs market) while the raw prob doesn't support it (e.g., raw
p=0.50 vs market p=0.20 means the line is bad; but if cal says p=0.30 and
edge=+0.10, the pick gets through but is a value trap).

Usage:
    python scripts/diag_nba_picks_raw_vs_cal.py
    python scripts/diag_nba_picks_raw_vs_cal.py --n 10
    python scripts/diag_nba_picks_raw_vs_cal.py --seed 42
"""
import sys, json, argparse, warnings, random
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.scripts.kalshi_nba_unified import (
    load_features, _load_regressor, _match_player, _p_ge_line,
)
from src.scripts.nba_bet import get_nba_bets
from src.data.kalshi import KalshiClient
from src.models.calibrator import BetaCalibrator
from src.models.distributions import p_ge_stat


def _load_old_cal(json_name: str, model_dir: Path):
    """Load oldest .bak.json (training-test cal) for a stat, or None.

    Caller passes the stat in upper case (e.g. "AST") matching the scanner
    convention, but the on-disk `.bak.json` files are lower-cased
    (e.g. `ast_beta_cal.*.bak.json`). Python's `pathlib.Path.glob` uses
    case-sensitive fnmatch on Unix, so we lowercase the stat here.
    """
    pattern = f"{json_name.lower()}_beta_cal.*.bak.json"
    cands = sorted(model_dir.glob(pattern))
    if cands:
        return BetaCalibrator.load(cands[0])
    return None


def _build_p_raw(latest, model_name, line_val, mrow, model, std, feats):
    """Compute the raw p (no calibrator) for a player+line."""
    feat_dict = {c: float(mrow[c]) if c in mrow.index and not pd.isna(mrow[c]) else 0.0
                 for c in feats}
    X_pred = pd.DataFrame([feat_dict]).fillna(0)
    mu = model.predict(X_pred)[0]
    sigma = max(std, 0.3)
    return float(p_ge_stat(model_name, mu, sigma, line_val)), float(mu)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=10,
                   help="Number of picks to sample (default 10)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for sampling (default 42)")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    MODEL_DIR = Path("models/nba")

    print("=" * 80)
    print("  NBA PICK DIAGNOSTIC — raw vs calibrated probs (calibrator artifact check)")
    print(f"  Sampling {args.n} picks from today's scan")
    print("=" * 80)

    # Get all picks via the public get_nba_bets() (mimics scanner flow)
    print("\n  Loading features + running scan...")
    kc = KalshiClient()
    picks = get_nba_bets(kc=kc, min_edge=0.05)
    print(f"  Total qualifying picks: {len(picks)}")
    if not picks:
        print("  No picks to sample.")
        return

    # Stratified sample: take at least 1 from each stat, then fill the rest
    by_stat = {}
    for pick in picks:
        by_stat.setdefault(pick["type"], []).append(pick)

    sample = []
    # 1 per stat first
    for stat, stat_picks in by_stat.items():
        sample.append(random.choice(stat_picks))
    # Fill the rest randomly
    remaining = [pk for pk in picks if pk not in sample]
    random.shuffle(remaining)
    sample.extend(remaining[:max(0, args.n - len(sample))])
    sample = sample[:args.n]

    # Load features once for the raw-prob recomputation
    latest = load_features()
    if latest is None or latest.empty:
        print("  Could not load features.")
        return

    # Cache models per stat
    model_cache = {}
    print()
    print(f"  {'Stat':5s} {'Player':22s} {'Ln':>3s} "
          f"{'Mkt':>5s} {'Raw':>6s} {'Old':>6s} {'New':>6s}  "
          f"{'eRaw':>6s} {'eOld':>6s} {'eNew':>6s}  Verdict")
    print(f"  {'-'*5} {'-'*22} {'-'*3} "
          f"{'-'*5} {'-'*6} {'-'*6} {'-'*6}  "
          f"{'-'*6} {'-'*6} {'-'*6}  -------")

    artifact_count = 0
    for pick in sample:
        stat = pick["type"]
        pname = pick["player"]
        line = pick["line_val"]
        mkt = pick["market_prob"]
        cal_new = pick["model_prob"]
        edge_new = pick["edge"]

        # Load model + old cal
        if stat not in model_cache:
            model, std, feats, cal_new_obj = _load_regressor(stat)
            if model is None:
                print(f"  {stat:5s} {pname[:22]:22s} {line:>3d}  no model")
                continue
            old_cal = _load_old_cal(stat, MODEL_DIR)
            model_cache[stat] = (model, std, feats, cal_new_obj, old_cal)
        model, std, feats, cal_new_obj, old_cal = model_cache[stat]

        mrow = _match_player(pname, latest)
        if mrow is None:
            print(f"  {stat:5s} {pname[:22]:22s} {line:>3d}  no match")
            continue

        p_raw, mu = _build_p_raw(latest, stat, line, mrow, model, std, feats)
        p_old = old_cal(p_raw) if (old_cal is not None and old_cal._fitted) else None

        edge_raw = p_raw - mkt
        edge_old = (p_old - mkt) if p_old is not None else None
        edge_new_disp = edge_new  # already computed in pick

        # Verdict: is this a calibrator artifact?
        # A pick is a calibrator artifact if the raw model says it's a bad
        # bet (edge_raw < 0) but the new calibrator inflates the prob
        # enough to make it look attractive (edge_new > 0).
        if edge_raw < 0 and edge_new > 0:
            verdict = "🚩 ARTIFACT"
            artifact_count += 1
        elif edge_new > 0.10 and p_raw < 0.30:
            verdict = "⚠ aggressive cal"
        elif p_raw < 0.20 and edge_new > 0.05:
            verdict = "⚠ cal rescued"
        elif p_raw > 0.30 and edge_new > 0.05:
            verdict = "✓ real edge"
        else:
            verdict = "neutral"

        old_disp = f"{p_old:>6.3f}" if p_old is not None else f"{'n/a':>6s}"
        eold_disp = f"{edge_old:>+6.3f}" if edge_old is not None else f"{'n/a':>6s}"
        print(f"  {stat:5s} {pname[:22]:22s} {line:>3d} "
              f"{mkt:>5.2f} {p_raw:>6.3f} {old_disp} {cal_new:>6.3f}  "
              f"{edge_raw:>+6.3f} {eold_disp} {edge_new_disp:>+6.3f}  {verdict}")

    print(f"\n  {'='*76}")
    print(f"  Sampled {len(sample)} picks, flagged {artifact_count} as calibrator artifacts")
    print(f"  {'='*76}")


if __name__ == "__main__":
    main()
