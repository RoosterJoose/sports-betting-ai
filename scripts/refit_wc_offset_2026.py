#!/usr/bin/env python3
"""Refit models/worldcup/offset_oos_2023plus.json with 2026 WC matches.

Spawned by the user's request: "After the first ~10 WC 2026 matches complete
(around June 14-15), re-run src/scripts/tournament_offset_analysis.py and
validate the empirical offset (delta_home/delta_draw/delta_away) against the
2026 sample. If the offset has shifted, regenerate
models/worldcup/offset_oos_2023plus.json with the 2026-included fit."

Why this wrapper exists (vs just re-running tournament_offset_analysis.py):
  - tournament_offset_analysis.py writes to tournament_offset_analysis.json
    (per-tournament stats), NOT offset_oos_2023plus.json (the production file).
  - shrunk_offset_sweep.py writes to neutral_offset.json (different file,
    destructive on every run).
  - Neither has a sample-size guard (n_2026 >= 10) or a shift threshold
    (>5pp in any delta component) — both required to avoid overfitting on
    10 early-tournament matches alone.

Method:
  1. Load 2023+ neutral-venue matches (WC, EC, AC) via fetch_all_matches()
  2. Predict each match with the existing wc_match_outcome model
  3. Compute pooled (model_mean - actual_rate) delta, capped at ±0.15
  4. Compare to current delta in offset_oos_2023plus.json
  5. If n_2026 >= 10 AND any delta component shifts by > 5pp (0.05 absolute):
       - Regenerate offset_oos_2023plus.json with the new pooled delta
       - Save a report to models/worldcup/refit_wc_offset_2026.json
  6. Otherwise: log verdict and exit 0 (no action)

Usage:
    # Dry-run (default): report verdict, don't write
    python -m scripts.refit_wc_offset_2026

    # Commit: regenerate the production file if shift detected
    python -m scripts.refit_wc_offset_2026 --commit

    # Force: skip the n_2026>=10 guard (e.g., for testing with synthetic data)
    python -m scripts.refit_wc_offset_2026 --force

    # Customize the shift threshold (default 0.05 = 5pp)
    python -m scripts.refit_wc_offset_2026 --shift-threshold 0.03
"""
import sys, json, argparse, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.world_cup import fetch_all_matches, compute_elo, build_feature_vector

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "worldcup"
CALIB_DIR = MODEL_DIR / "calibration"
OFFSET_OOS_PATH = MODEL_DIR / "offset_oos_2023plus.json"
REPORT_PATH = MODEL_DIR / "refit_wc_offset_2026.json"

NEUTRAL_TOURNAMENTS = {"WC", "EC", "AC"}
DEFAULT_CAP = 0.15
DEFAULT_SHIFT_THRESHOLD = 0.05  # 5 percentage points
MIN_N_2026 = 10  # sample-size floor for 2026-WC alone
MIN_N_TOTAL = 80  # sample-size floor for the pooled 2023+ set


def _build_team_history(elo_df):
    """Per-team chronological match history (vectorized)."""
    from src.data.world_cup import _elo_expected
    dates = elo_df["match_date"].values
    home = elo_df["home_team"].values
    away = elo_df["away_team"].values
    home_score = elo_df["home_score"].values.astype(int)
    away_score = elo_df["away_score"].values.astype(int)
    elo_home_pre = elo_df["elo_home_pre"].values
    elo_away_pre = elo_df["elo_away_pre"].values

    team_hist = {}
    n = len(elo_df)
    for i in range(n):
        h, a = home[i], away[i]
        if h not in team_hist:
            team_hist[h] = []
        team_hist[h].append((dates[i], 1, int(home_score[i]), int(away_score[i]),
                              float(elo_home_pre[i]), float(elo_away_pre[i])))
        if a not in team_hist:
            team_hist[a] = []
        team_hist[a].append((dates[i], 0, int(home_score[i]), int(away_score[i]),
                              float(elo_away_pre[i]), float(elo_home_pre[i])))
    for t in team_hist:
        team_hist[t].sort(key=lambda r: r[0])
    return team_hist


def _form_for_team(team_hist, team, cutoff_date, default_elo=1500):
    if team not in team_hist:
        return {"perf": 0.0, "opp_elo": default_elo, "gs": 0.0, "gc": 0.0, "n": 0}
    hist = team_hist[team]
    dates_arr = np.array([r[0] for r in hist])
    cutoff_ns = np.datetime64(cutoff_date)
    end = int(np.searchsorted(dates_arr, cutoff_ns, side="left"))
    start = max(0, end - 5)
    window = hist[start:end]
    if not window:
        return {"perf": 0.0, "opp_elo": default_elo, "gs": 0.0, "gc": 0.0, "n": 0}

    from src.data.world_cup import _elo_expected
    perf_sum, opp_elo_sum, gs_sum, gc_sum = 0.0, 0.0, 0.0, 0.0
    for _, is_home, hs, as_, team_elo, opp_elo in window:
        if hs > as_:
            actual = 1.0 if is_home else 0.0
        elif as_ > hs:
            actual = 0.0 if is_home else 1.0
        else:
            actual = 0.5
        expected = _elo_expected(team_elo, opp_elo)
        perf_sum += actual - expected
        opp_elo_sum += opp_elo
        if is_home:
            gs_sum += hs; gc_sum += as_
        else:
            gs_sum += as_; gc_sum += hs
    k = len(window)
    return {
        "perf": perf_sum / k, "opp_elo": opp_elo_sum / k,
        "gs": gs_sum / k, "gc": gc_sum / k, "n": k,
    }


def apply_offset(probs, delta, cap=DEFAULT_CAP):
    p = probs.copy()
    p[0] -= max(-cap, min(cap, delta[0]))
    p[1] -= max(-cap, min(cap, delta[1]))
    p[2] -= max(-cap, min(cap, delta[2]))
    p = np.maximum(p, 0.001)
    return p / p.sum()


def brier_score(probs, y_onehot):
    return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))


def predict_matches(test_df, model, features, elo_df, team_hist):
    """Run the model on each match in test_df; return (probs, actuals)."""
    raw_probs = []
    actuals = []
    for _, match in test_df.iterrows():
        home = match["home_team"]
        away = match["away_team"]
        match_date = match["match_date"]

        pre = elo_df[elo_df["match_date"] < match_date]
        elo_ratings = {}
        if not pre.empty:
            elo_ratings.update(dict(zip(pre["home_team"].values, pre["elo_home_post"].values)))
            elo_ratings.update(dict(zip(pre["away_team"].values, pre["elo_away_post"].values)))
        elo_h = elo_ratings.get(home, 1500)
        elo_a = elo_ratings.get(away, 1500)

        hf = _form_for_team(team_hist, home, match_date, default_elo=elo_h)
        af = _form_for_team(team_hist, away, match_date, default_elo=elo_a)

        x = build_feature_vector(elo_h, elo_a, hf, af, "WC", features)
        probs = model.predict(x)[0]

        hs, as_ = int(match["home_score"]), int(match["away_score"])
        if hs > as_:
            actual = 0
        elif as_ > hs:
            actual = 2
        else:
            actual = 1

        raw_probs.append(probs)
        actuals.append(actual)
    return np.array(raw_probs), np.array(actuals)


def compute_pooled_delta(probs, actuals, cap=DEFAULT_CAP):
    """Capped (model_mean - actual_rate) for [home, draw, away]."""
    actual_rates = np.bincount(actuals, minlength=3) / len(actuals)
    pred_means = probs.mean(axis=0)
    return np.clip(pred_means - actual_rates, -cap, cap), actual_rates, pred_means


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--commit", action="store_true",
                        help="Write to offset_oos_2023plus.json if shift detected (default: dry-run)")
    parser.add_argument("--force", action="store_true",
                        help="Bypass the n_2026>=10 sample-size guard")
    parser.add_argument("--shift-threshold", type=float, default=DEFAULT_SHIFT_THRESHOLD,
                        help=f"Min delta-component shift to trigger regen (default: {DEFAULT_SHIFT_THRESHOLD})")
    args = parser.parse_args()

    print("=" * 72)
    print("  WC OFFSET RE-FIT (2023+ neutral-venue, with 2026 WC sample)")
    print(f"  Mode: {'COMMIT' if args.commit else 'dry-run (no files written)'}")
    print(f"  Shift threshold: {args.shift_threshold:.3f} ({args.shift_threshold*100:.1f}pp)")
    print(f"  Min n_2026 floor: {MIN_N_2026}{' (BYPASSED via --force)' if args.force else ''}")
    print("=" * 72)

    # 1. Load current offset
    if not OFFSET_OOS_PATH.exists():
        print(f"\n  ❌ {OFFSET_OOS_PATH.relative_to(PROJECT_ROOT)} not found.")
        sys.exit(1)
    with open(OFFSET_OOS_PATH) as f:
        current = json.load(f)
    delta_old = np.array([
        current["offset_used"]["delta_home"],
        current["offset_used"]["delta_draw"],
        current["offset_used"]["delta_away"],
    ])
    cap = current["offset_used"].get("cap", DEFAULT_CAP)
    print(f"\n  Current offset: Δ_H={delta_old[0]:+.3f}  Δ_D={delta_old[1]:+.3f}  Δ_A={delta_old[2]:+.3f}")
    print(f"  Current file: n_test={current.get('n_test')}  n_neutral={current.get('n_neutral')}")

    # 2. Load model
    import lightgbm as lgb
    model_path = MODEL_DIR / "wc_match_outcome.txt"
    meta_path = MODEL_DIR / "wc_match_outcome.meta.json"
    if not model_path.exists():
        print(f"\n  ❌ Model not found: {model_path}")
        sys.exit(1)
    model = lgb.Booster(model_file=str(model_path))
    with open(meta_path) as f:
        meta = json.load(f)
    features = meta.get("features", [])

    # 3. Load 2023+ neutral-venue matches
    print("\n  Loading 2023+ neutral-venue matches (WC, EC, AC)...")
    df = fetch_all_matches()
    elo_df = compute_elo(df)
    elo_df = elo_df.merge(
        df[["match_date", "home_team", "away_team", "tournament_code"]],
        on=["match_date", "home_team", "away_team"], how="left",
    )
    test_df = elo_df[
        (elo_df["match_date"] >= pd.Timestamp("2023-01-01"))
        & (elo_df["tournament_code"].isin(NEUTRAL_TOURNAMENTS))
    ].copy().sort_values("match_date").reset_index(drop=True)

    n_2023 = len(test_df)
    n_2026 = int((test_df["match_date"] >= pd.Timestamp("2026-01-01")).sum())
    print(f"  Pooled 2023+ neutral-venue: {n_2023} matches")
    print(f"  2026 subset: {n_2026} matches (floor: {MIN_N_2026})")
    if test_df.empty:
        print("\n  ❌ No 2023+ neutral-venue matches found.")
        sys.exit(1)
    print(f"  By tournament:")
    for tc, n in test_df["tournament_code"].value_counts().items():
        print(f"    {tc}: {n}")
    print(f"  Date range: {test_df.match_date.min().date()} to {test_df.match_date.max().date()}")

    # 4. Predict each match
    print("\n  Predicting...")
    team_hist = _build_team_history(elo_df)
    raw_probs, actuals = predict_matches(test_df, model, features, elo_df, team_hist)
    y_onehot = np.zeros((len(actuals), 3))
    y_onehot[np.arange(len(actuals)), actuals] = 1

    # 5. Compute pooled delta
    delta_new, actual_rates, pred_means = compute_pooled_delta(raw_probs, actuals, cap)
    brier_raw = brier_score(raw_probs, y_onehot)
    probs_new = np.array([apply_offset(p, delta_new, cap) for p in raw_probs])
    brier_new = brier_score(probs_new, y_onehot)
    probs_old = np.array([apply_offset(p, delta_old, cap) for p in raw_probs])
    brier_old = brier_score(probs_old, y_onehot)

    # 6. Per-tournament breakdown
    per_tc = {}
    for tc in test_df["tournament_code"].unique():
        mask = (test_df["tournament_code"] == tc).values
        if mask.sum() < 5:
            continue
        probs_tc = raw_probs[mask]
        actuals_tc = actuals[mask]
        a_rates_tc = np.bincount(actuals_tc, minlength=3) / len(actuals_tc)
        p_means_tc = probs_tc.mean(axis=0)
        deltas_tc = np.clip(p_means_tc - a_rates_tc, -cap, cap)
        per_tc[tc] = {
            "n": int(mask.sum()),
            "actual_rates": a_rates_tc.tolist(),
            "pred_means": p_means_tc.tolist(),
            "raw_delta": (p_means_tc - a_rates_tc).tolist(),
            "capped_delta": deltas_tc.tolist(),
        }

    # 7. Report
    print(f"\n  {'='*68}")
    print(f"  POOLED 2023+ DELTA (cap=±{cap})")
    print(f"  {'='*68}")
    print(f"    actual rates: H={actual_rates[0]:.1%}  D={actual_rates[1]:.1%}  A={actual_rates[2]:.1%}")
    print(f"    pred means  : H={pred_means[0]:.1%}  D={pred_means[1]:.1%}  A={pred_means[2]:.1%}")
    print(f"    raw delta   : H={pred_means[0]-actual_rates[0]:+.3f}  "
          f"D={pred_means[1]-actual_rates[1]:+.3f}  A={pred_means[2]-actual_rates[2]:+.3f}")
    print(f"    capped delta: H={delta_new[0]:+.3f}  D={delta_new[1]:+.3f}  A={delta_new[2]:+.3f}")
    print(f"\n  Brier: raw={brier_raw:.4f}  current={brier_old:.4f}  new={brier_new:.4f}  "
          f"(Δ {brier_new - brier_old:+.4f})")

    print(f"\n  {'='*68}")
    print(f"  SHIFT DETECTION (threshold {args.shift_threshold:.3f})")
    print(f"  {'='*68}")
    deltas_shift = delta_new - delta_old
    print(f"    Δ_H shift: {deltas_shift[0]:+.3f}  ({'⚠️ >threshold' if abs(deltas_shift[0]) > args.shift_threshold else 'OK'})")
    print(f"    Δ_D shift: {deltas_shift[1]:+.3f}  ({'⚠️ >threshold' if abs(deltas_shift[1]) > args.shift_threshold else 'OK'})")
    print(f"    Δ_A shift: {deltas_shift[2]:+.3f}  ({'⚠️ >threshold' if abs(deltas_shift[2]) > args.shift_threshold else 'OK'})")
    any_shift = bool(np.any(np.abs(deltas_shift) > args.shift_threshold))

    if per_tc:
        print(f"\n  Per-tournament pooled delta:")
        for tc, d in per_tc.items():
            cd = d["capped_delta"]
            print(f"    {tc} (n={d['n']}): H={cd[0]:+.3f}  D={cd[1]:+.3f}  A={cd[2]:+.3f}")

    # 8. Verdict
    print(f"\n  {'='*68}")
    print(f"  VERDICT")
    print(f"  {'='*68}")
    sample_size_ok = (n_2026 >= MIN_N_2026) or args.force
    if not sample_size_ok:
        verdict = (f"  ⏳  Waiting for more 2026 data (have {n_2026}, need {MIN_N_2026}). "
                   f"Holding current offset.")
        action = "hold"
    elif not any_shift:
        verdict = (f"  ✅  No significant shift (all components within {args.shift_threshold:.3f} of current). "
                   f"Holding current offset.")
        action = "hold"
    else:
        verdict = (f"  🔄  SIGNIFICANT SHIFT DETECTED in pooled 2023+ delta. "
                   f"Regenerating offset_oos_2023plus.json (mode: {'COMMIT' if args.commit else 'dry-run'}).")
        action = "regenerate"
    print(verdict)

    report = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "mode": "commit" if args.commit else "dry-run",
        "n_total_2023_plus": int(n_2023),
        "n_2026": int(n_2026),
        "n_2026_floor": MIN_N_2026,
        "n_total_floor": MIN_N_TOTAL,
        "current_delta": delta_old.tolist(),
        "new_delta": delta_new.tolist(),
        "delta_shift": deltas_shift.tolist(),
        "shift_threshold": float(args.shift_threshold),
        "any_shift_detected": any_shift,
        "sample_size_ok": bool(sample_size_ok),
        "brier_raw": float(brier_raw),
        "brier_current": float(brier_old),
        "brier_new": float(brier_new),
        "brier_improvement": float(brier_old - brier_new),
        "per_tournament": per_tc,
        "action": action,
        "verdict": verdict.strip(),
    }

    # 9. Regenerate if commit + shift + sample-size ok
    if action == "regenerate":
        if not args.commit:
            print(f"\n  → Re-run with --commit to write to {OFFSET_OOS_PATH.name}.")
        else:
            new_offset = dict(current)
            new_offset["offset_used"] = {
                "delta_home": float(delta_new[0]),
                "delta_draw": float(delta_new[1]),
                "delta_away": float(delta_new[2]),
                "cap": float(cap),
            }
            new_offset["n_test"] = int(n_2023)
            new_offset["n_neutral"] = int(n_2023)
            new_offset["n_non_neutral"] = current.get("n_non_neutral", 0)
            new_offset["neutral_subset"] = {
                "brier_raw": float(brier_raw),
                "brier_offset": float(brier_new),
                "brier_delta": float(brier_new - brier_raw),
                "verdict": "regenerated 2026-included fit" if brier_new < brier_raw else "regenerated but Brier did not improve",
            }
            new_offset["regen_history"] = new_offset.get("regen_history", [])
            new_offset["regen_history"].append({
                "at": report["computed_at"],
                "n_2023_plus": int(n_2023),
                "n_2026": int(n_2026),
                "old_delta": delta_old.tolist(),
                "new_delta": delta_new.tolist(),
                "delta_shift": deltas_shift.tolist(),
                "brier_improvement": float(brier_old - brier_new),
            })
            with open(OFFSET_OOS_PATH, "w") as f:
                json.dump(new_offset, f, indent=2)
            print(f"\n  ✅ Wrote {OFFSET_OOS_PATH.relative_to(PROJECT_ROOT)}")
            print(f"     Δ_H={delta_new[0]:+.3f}  Δ_D={delta_new[1]:+.3f}  Δ_A={delta_new[2]:+.3f}")

    # 10. Save report
    if args.commit or action == "hold":
        with open(REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  📝 Saved report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")

    print(f"\n  {'='*68}")
    print(f"  SUMMARY")
    print(f"  {'='*68}")
    print(f"    n_2023+ pooled: {n_2023}  (2026 subset: {n_2026})")
    print(f"    Current Δ:      H={delta_old[0]:+.3f}  D={delta_old[1]:+.3f}  A={delta_old[2]:+.3f}")
    print(f"    New Δ:          H={delta_new[0]:+.3f}  D={delta_new[1]:+.3f}  A={delta_new[2]:+.3f}")
    print(f"    Action:         {action.upper()}")
    print(f"  {'='*68}\n")


if __name__ == "__main__":
    main()
