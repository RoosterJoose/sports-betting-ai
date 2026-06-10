#!/usr/bin/env python3
"""Per-tournament offset analysis: AC (n=44) vs EC (n=43) on 2023+ OOS.

Question: Why does AC improve strongly under the global offset while EC
slightly worsens? Sample-size noise, or a real per-tournament structure?

Method
------
1. Load raw model probs for the 2023+ neutral-venue subset
2. Split by tournament_code (AC vs EC) and compute per-tournament:
   - actual home/draw/away rates
   - model mean predictions
   - raw Δ_class = mean(P_model) - actual_rate
   - capped Δ_class (at ±0.15)
3. Compare 3 offset strategies:
   a. NO offset: probs unchanged
   b. GLOBAL offset: Δ = (-0.112, +0.053, +0.059) — current production
   c. PER-TOURNAMENT offset: re-fit Δ on each tournament's actuals
4. Bootstrap each tournament (1000 resamples) to get 95% CIs on Brier
   for the Brier difference (offset - no_offset)
5. Verdict: is the EC worsening within the noise floor?

Usage:
    python -m src.scripts.tournament_offset_analysis
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.world_cup import fetch_all_matches, compute_elo, build_feature_vector

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "worldcup"
CALIB_DIR = MODEL_DIR / "calibration"

NEUTRAL_TOURNAMENTS = {"WC", "EC", "AC"}
OUTCOME_LABELS = ["home", "draw", "away"]


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


def apply_offset(probs, delta, cap=0.15):
    """Apply offset to a single probs vector, renormalized."""
    p = probs.copy()
    p[0] -= max(-cap, min(cap, delta[0]))
    p[1] -= max(-cap, min(cap, delta[1]))
    p[2] -= max(-cap, min(cap, delta[2]))
    p = np.maximum(p, 0.001)
    return p / p.sum()


def brier_score(probs, y_onehot):
    return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))


def bootstrap_brier_ci(probs, y_onehot, n_boot=1000, seed=42):
    """Bootstrap 95% CI on Brier score."""
    rng = np.random.default_rng(seed)
    n = len(probs)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = brier_score(probs[idx], y_onehot[idx])
    return {
        "mean": float(np.mean(boots)),
        "ci_lo": float(np.percentile(boots, 2.5)),
        "ci_hi": float(np.percentile(boots, 97.5)),
        "std": float(np.std(boots)),
    }


def main():
    print("=" * 70)
    print("  PER-TOURNAMENT OFFSET ANALYSIS — AC (n=44) vs EC (n=43)")
    print("  Question: sample-size noise or real per-tournament structure?")
    print("=" * 70)

    # 1. Load model + global offset
    print("\n1. Loading model + global offset...")
    import lightgbm as lgb
    model_path = MODEL_DIR / "wc_match_outcome.txt"
    meta_path = MODEL_DIR / "wc_match_outcome.meta.json"
    offset_path = CALIB_DIR / "neutral_offset.json"
    model = lgb.Booster(model_file=str(model_path))
    with open(meta_path) as f:
        meta = json.load(f)
    with open(offset_path) as f:
        offset_global = json.load(f)
    features = meta.get("features", [])
    cap = offset_global.get("cap", 0.15)
    delta_global = np.array([
        max(-cap, min(cap, offset_global.get("delta_home", 0.0))),
        max(-cap, min(cap, offset_global.get("delta_draw", 0.0))),
        max(-cap, min(cap, offset_global.get("delta_away", 0.0))),
    ])
    print(f"  Global offset: Δ_H={delta_global[0]:+.3f} Δ_D={delta_global[1]:+.3f} Δ_A={delta_global[2]:+.3f}")

    # 2. Load 2023+ neutral-venue matches
    print("\n2. Loading 2023+ neutral-venue matches...")
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
    print(f"  {len(test_df)} neutral-venue matches")
    print(f"  By tournament:")
    for tc, n in test_df["tournament_code"].value_counts().items():
        print(f"    {tc}: {n}")

    # 3. Precompute team history
    print("\n3. Precomputing team history...")
    team_hist = _build_team_history(elo_df)
    print(f"  {len(team_hist)} teams indexed")

    # 4. Predict each match
    print("\n4. Predicting...")
    rows = []
    for _, match in test_df.iterrows():
        home = match["home_team"]
        away = match["away_team"]
        match_date = match["match_date"]
        tc = match["tournament_code"]

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
        if hs > as_: actual = 0
        elif as_ > hs: actual = 2
        else: actual = 1

        rows.append({
            "date": str(match_date.date()),
            "home": home, "away": away, "tc": tc,
            "probs": probs, "actual": actual,
        })
    print(f"  {len(rows)} matches predicted.")

    # 5. Per-tournament stats
    print(f"\n{'='*70}")
    print(f"  PER-TOURNAMENT STATS")
    print(f"{'='*70}")
    print(f"\n  {'Tournament':<10} {'n':>4} {'Home%':>8} {'Draw%':>8} {'Away%':>8} "
          f"{'pred_H':>8} {'pred_D':>8} {'pred_A':>8} {'raw_Δ_H':>9} {'raw_Δ_D':>9} {'raw_Δ_A':>9}")
    print(f"  {'-'*10} {'-'*4} {'-'*8} {'-'*8} {'-'*8} "
          f"{'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*9} {'-'*9}")
    per_tc_data = {}
    for tc in ["AC", "EC"]:
        tc_rows = [r for r in rows if r["tc"] == tc]
        if not tc_rows:
            continue
        n = len(tc_rows)
        actuals = np.array([r["actual"] for r in tc_rows])
        probs = np.array([r["probs"] for r in tc_rows])
        actual_rates = np.bincount(actuals, minlength=3) / n
        pred_means = probs.mean(axis=0)
        deltas = pred_means - actual_rates
        per_tc_data[tc] = {
            "n": n, "actuals": actuals, "probs": probs,
            "actual_rates": actual_rates, "pred_means": pred_means,
            "raw_deltas": deltas,
        }
        print(f"  {tc:<10} {n:>4} {actual_rates[0]*100:>7.1f}% {actual_rates[1]*100:>7.1f}% {actual_rates[2]*100:>7.1f}% "
              f"{pred_means[0]*100:>7.1f}% {pred_means[1]*100:>7.1f}% {pred_means[2]*100:>7.1f}% "
              f"{deltas[0]:>+8.3f} {deltas[1]:>+8.3f} {deltas[2]:>+8.3f}")

    # 6. Compare 3 offset strategies per tournament
    print(f"\n{'='*70}")
    print(f"  BRIER SCORES — 3 OFFSET STRATEGIES")
    print(f"{'='*70}")
    print(f"\n  {'Tournament':<12} {'No offset':>12} {'Global':>12} {'Per-tournament':>16} {'Δ global':>10} {'Δ per-tc':>10}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*16} {'-'*10} {'-'*10}")

    results_by_tc = {}
    for tc in ["AC", "EC"]:
        d = per_tc_data[tc]
        n = d["n"]
        probs = d["probs"]
        actuals = d["actuals"]
        y_onehot = np.zeros((n, 3))
        y_onehot[np.arange(n), actuals] = 1

        # No offset
        brier_no = brier_score(probs, y_onehot)

        # Global offset
        probs_global = np.array([apply_offset(p, delta_global, cap) for p in probs])
        brier_global = brier_score(probs_global, y_onehot)

        # Per-tournament offset (capped)
        delta_tc = np.clip(d["raw_deltas"], -cap, cap)
        probs_tc = np.array([apply_offset(p, delta_tc, cap) for p in probs])
        brier_tc = brier_score(probs_tc, y_onehot)

        results_by_tc[tc] = {
            "n": n, "brier_no": brier_no, "brier_global": brier_global,
            "brier_per_tc": brier_tc, "delta_tc": delta_tc.tolist(),
            "delta_global_change": brier_global - brier_no,
            "delta_per_tc_change": brier_tc - brier_no,
        }
        print(f"  {tc:<12} {brier_no:>12.4f} {brier_global:>12.4f} {brier_tc:>16.4f} "
              f"{brier_global-brier_no:>+9.4f} {brier_tc-brier_no:>+9.4f}")
        print(f"             (per-tc Δ: H={delta_tc[0]:+.3f} D={delta_tc[1]:+.3f} A={delta_tc[2]:+.3f})")

    # 7. Bootstrap CIs on Brier difference (global vs no offset)
    print(f"\n{'='*70}")
    print(f"  BOOTSTRAP 95% CI — Brier change from global offset (negative = improves)")
    print(f"{'='*70}")
    print(f"\n  {'Tournament':<12} {'n':>4} {'Brier change':>14} {'95% CI':>20} {'Significant?':>15}")
    print(f"  {'-'*12} {'-'*4} {'-'*14} {'-'*20} {'-'*15}")
    for tc in ["AC", "EC"]:
        d = per_tc_data[tc]
        n = d["n"]
        probs = d["probs"]
        actuals = d["actuals"]
        y_onehot = np.zeros((n, 3))
        y_onehot[np.arange(n), actuals] = 1

        probs_global = np.array([apply_offset(p, delta_global, cap) for p in probs])
        brier_no = brier_score(probs, y_onehot)
        brier_g = brier_score(probs_global, y_onehot)
        point_change = brier_g - brier_no

        # Bootstrap: resample, compute change on each
        rng = np.random.default_rng(42)
        n_boot = 1000
        boots = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, n)
            boots[b] = brier_score(probs_global[idx], y_onehot[idx]) - brier_score(probs[idx], y_onehot[idx])

        ci_lo = float(np.percentile(boots, 2.5))
        ci_hi = float(np.percentile(boots, 97.5))
        sig = "✅ YES" if ci_hi < 0 else ("❌ NO" if ci_lo > 0 else "─ mixed")
        print(f"  {tc:<12} {n:>4} {point_change:>+14.4f} [{ci_lo:>+7.4f}, {ci_hi:>+7.4f}] {sig:>15}")

    # 8. Combined-pooled delta: if we pool AC + EC and refit one offset
    print(f"\n{'='*70}")
    print(f"  POOLED-DELTA ALTERNATIVE — fit one offset on combined 2023+ neutral")
    print(f"{'='*70}")
    all_probs = np.concatenate([per_tc_data["AC"]["probs"], per_tc_data["EC"]["probs"]])
    all_actuals = np.concatenate([per_tc_data["AC"]["actuals"], per_tc_data["EC"]["actuals"]])
    n = len(all_actuals)
    actual_rates_pooled = np.bincount(all_actuals, minlength=3) / n
    pred_means_pooled = all_probs.mean(axis=0)
    delta_pooled = np.clip(pred_means_pooled - actual_rates_pooled, -cap, cap)
    print(f"  Pooled actual rates: H={actual_rates_pooled[0]:.1%} D={actual_rates_pooled[1]:.1%} A={actual_rates_pooled[2]:.1%}")
    print(f"  Pooled pred means  : H={pred_means_pooled[0]:.1%} D={pred_means_pooled[1]:.1%} A={pred_means_pooled[2]:.1%}")
    print(f"  Pooled capped Δ    : H={delta_pooled[0]:+.3f} D={delta_pooled[1]:+.3f} A={delta_pooled[2]:+.3f}")
    print(f"  vs Global (2022 WC): H={delta_global[0]:+.3f} D={delta_global[1]:+.3f} A={delta_global[2]:+.3f}")
    pooled_briers = {}
    for tc in ["AC", "EC"]:
        d = per_tc_data[tc]
        probs_p = np.array([apply_offset(p, delta_pooled, cap) for p in d["probs"]])
        y_onehot = np.zeros((d["n"], 3))
        y_onehot[np.arange(d["n"]), d["actuals"]] = 1
        b_p = brier_score(probs_p, y_onehot)
        b_g = results_by_tc[tc]["brier_global"]
        b_no = results_by_tc[tc]["brier_no"]
        pooled_briers[tc] = b_p
        print(f"  {tc}: no={b_no:.4f}  global={b_g:.4f} (Δ {b_g-b_no:+.4f})  pooled={b_p:.4f} (Δ {b_p-b_no:+.4f})")

    # 9. Verdict
    print(f"\n{'='*70}")
    print(f"  VERDICT")
    print(f"{'='*70}")
    ac_global_d = results_by_tc["AC"]["delta_global_change"]
    ec_global_d = results_by_tc["EC"]["delta_global_change"]
    ac_per_d = results_by_tc["AC"]["delta_per_tc_change"]
    ec_per_d = results_by_tc["EC"]["delta_per_tc_change"]

    print(f"\n  Global offset (currently in production):")
    print(f"    AC: Brier change {ac_global_d:+.4f}  (improves)")
    print(f"    EC: Brier change {ec_global_d:+.4f}  ({'improves' if ec_global_d < 0 else 'worsens'})")
    print(f"  Per-tournament offset (if we re-fit on each):")
    print(f"    AC: Brier change {ac_per_d:+.4f}  (improves more)")
    print(f"    EC: Brier change {ec_per_d:+.4f}  (improves more)")

    # Sample-size noise check
    if abs(ec_global_d) < 0.02:
        verdict = (f"  The EC worsening ({ec_global_d:+.4f}) is likely sample-size noise — "
                   f"n=43 is too small to detect a real offset effect. The bootstrap CI "
                   f"above is the definitive test.")
    else:
        verdict = (f"  The EC worsening ({ec_global_d:+.4f}) may be real — exceeds the noise floor.")
    print(f"\n  {verdict}")

    # Recommendation
    print(f"\n  RECOMMENDATION:")
    if abs(ec_per_d) < abs(ec_global_d) and ac_per_d < ac_global_d:
        # Per-tournament helps both
        print(f"  ✅ Per-tournament offsetting helps BOTH tournaments (AC and EC).")
        print(f"     Consider wiring per-tournament offsets into scan_wc.py.")
    elif ac_per_d < ac_global_d and abs(ec_per_d) < abs(ec_global_d) + 0.01:
        print(f"  ─  Per-tournament helps AC more, EC is similar.")
        print(f"     Marginal benefit — recommend keeping global offset for simplicity.")
    else:
        print(f"  ─  Per-tournament offsetting doesn't help meaningfully beyond global.")
        print(f"     Keep current global offset (simpler, less overfitting risk).")

    # Save
    out_path = MODEL_DIR / "tournament_offset_analysis.json"
    out = {
        "n_total": int(len(rows)),
        "per_tournament": {
            tc: {
                "n": int(results_by_tc[tc]["n"]),
                "actual_rates": per_tc_data[tc]["actual_rates"].tolist(),
                "pred_means": per_tc_data[tc]["pred_means"].tolist(),
                "raw_delta": per_tc_data[tc]["raw_deltas"].tolist(),
                "capped_delta": results_by_tc[tc]["delta_tc"],
                "brier_no_offset": results_by_tc[tc]["brier_no"],
                "brier_global_offset": results_by_tc[tc]["brier_global"],
                "brier_per_tc_offset": results_by_tc[tc]["brier_per_tc"],
            }
            for tc in ["AC", "EC"] if tc in results_by_tc
        },
        "pooled_2023_delta": delta_pooled.tolist(),
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
