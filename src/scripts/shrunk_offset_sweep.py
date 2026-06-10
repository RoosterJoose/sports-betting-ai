#!/usr/bin/env python3
"""Sweep shrinkage weights for the empirical neutral-venue offset.

The current offset (Δ_H=-0.112, Δ_D=+0.053, Δ_A=+0.059) was computed on
the 2022 WC val set (57 matches). The pooled 2023+ delta is much smaller
(Δ_H=-0.054, Δ_D=+0.012, Δ_A=+0.042) — suggesting the 2022 WC was an
outlier with extreme home bias, and the current offset over-corrects
on 2023+ data.

A shrunk offset blends the two:
    Δ_shrunk = w * Δ_2022_wc + (1 - w) * Δ_pooled_2023

Sweep w from 0.0 (use only pooled 2023+) to 1.0 (use only 2022 WC)
in steps of 0.1, evaluate OOS Brier on the 2023+ neutral-venue subset
(n=87), and identify the best shrinkage weight.

Then write the best variant to neutral_offset.json and re-run
validate_offset_oos.py to confirm the Brier improvement.

Usage:
    python -m src.scripts.shrunk_offset_sweep
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
    p = probs.copy()
    p[0] -= max(-cap, min(cap, delta[0]))
    p[1] -= max(-cap, min(cap, delta[1]))
    p[2] -= max(-cap, min(cap, delta[2]))
    p = np.maximum(p, 0.001)
    return p / p.sum()


def brier_score(probs, y_onehot):
    return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))


def main():
    print("=" * 70)
    print("  SHRUNK-OFFSET WEIGHT SWEEP — 2023+ neutral-venue subset (n=87)")
    print("=" * 70)

    # 1. Load both deltas
    print("\n1. Loading offsets...")
    import lightgbm as lgb
    with open(CALIB_DIR / "neutral_offset.json") as f:
        offset_2022 = json.load(f)
    with open(MODEL_DIR / "tournament_offset_analysis.json") as f:
        pooled_2023 = json.load(f).get("pooled_2023_delta", [0, 0, 0])
    cap = offset_2022.get("cap", 0.15)
    delta_2022 = np.array([
        offset_2022.get("delta_home", 0),
        offset_2022.get("delta_draw", 0),
        offset_2022.get("delta_away", 0),
    ])
    delta_pooled = np.array(pooled_2023)
    print(f"  Δ_2022_wc   : H={delta_2022[0]:+.3f} D={delta_2022[1]:+.3f} A={delta_2022[2]:+.3f}")
    print(f"  Δ_pooled_23+: H={delta_pooled[0]:+.3f} D={delta_pooled[1]:+.3f} A={delta_pooled[2]:+.3f}")
    print(f"  Shrinkage w: w * Δ_2022 + (1-w) * Δ_pooled")

    # 2. Load model + 2023+ neutral-venue matches
    print("\n2. Loading model + 2023+ neutral-venue matches...")
    model = lgb.Booster(model_file=str(MODEL_DIR / "wc_match_outcome.txt"))
    with open(MODEL_DIR / "wc_match_outcome.meta.json") as f:
        meta = json.load(f)
    features = meta.get("features", [])

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
    print(f"  {len(test_df)} neutral-venue matches in 2023+")

    # 3. Precompute team history
    print("\n3. Precomputing team history...")
    team_hist = _build_team_history(elo_df)
    print(f"  {len(team_hist)} teams indexed")

    # 4. Predict each match
    print("\n4. Predicting...")
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
        if hs > as_: actual = 0
        elif as_ > hs: actual = 2
        else: actual = 1

        raw_probs.append(probs)
        actuals.append(actual)

    raw_arr = np.array(raw_probs)
    act_arr = np.array(actuals)
    y_onehot = np.zeros((len(act_arr), 3))
    y_onehot[np.arange(len(act_arr)), act_arr] = 1
    n = len(act_arr)

    brier_raw = brier_score(raw_arr, y_onehot)
    acc_raw = float((np.argmax(raw_arr, axis=1) == act_arr).mean())
    print(f"\n  Raw baseline (no offset): Brier={brier_raw:.4f}  Acc={acc_raw:.1%}")
    print()

    # 5. Sweep weights
    print("=" * 70)
    print(f"  SHRINKAGE WEIGHT SWEEP (n={n})")
    print("=" * 70)
    print(f"\n  {'w':>6} {'Δ_H':>7} {'Δ_D':>7} {'Δ_A':>7}  {'Brier':>8} {'Acc':>7} {'Δ Brier':>10} {'Δ Acc':>8}")
    print(f"  {'-'*6} {'-'*7} {'-'*7} {'-'*7}  {'-'*8} {'-'*7} {'-'*10} {'-'*8}")

    results = []
    for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        delta_shrunk = w * delta_2022 + (1 - w) * delta_pooled
        off_arr = np.array([apply_offset(p, delta_shrunk, cap) for p in raw_arr])
        b = brier_score(off_arr, y_onehot)
        a = float((np.argmax(off_arr, axis=1) == act_arr).mean())
        results.append({
            "w": w, "delta": delta_shrunk.tolist(),
            "brier": b, "acc": a,
        })
        marker = " ← 2022 only" if w == 1.0 else (" ← pooled only" if w == 0.0 else "")
        print(f"  {w:>6.2f} {delta_shrunk[0]:>+7.3f} {delta_shrunk[1]:>+7.3f} {delta_shrunk[2]:>+7.3f}  "
              f"{b:>8.4f} {a*100:>6.1f}% {b-brier_raw:>+10.4f} {a-acc_raw:>+7.1%}{marker}")

    # 6. Find best
    best = min(results, key=lambda r: r["brier"])
    w_best = best["w"]
    delta_best = np.array(best["delta"])
    print(f"\n  BEST: w={w_best:.2f}  Δ_H={delta_best[0]:+.3f}  Δ_D={delta_best[1]:+.3f}  Δ_A={delta_best[2]:+.3f}")
    print(f"        Brier={best['brier']:.4f}  Acc={best['acc']*100:.1f}%")
    print(f"  Compare: Raw={brier_raw:.4f}  Current (w=1.0)={[r for r in results if r['w']==1.0][0]['brier']:.4f}")

    # 7. Write best to neutral_offset.json
    print("\n" + "=" * 70)
    print("  UPDATE neutral_offset.json WITH BEST SHRUNK VARIANT")
    print("=" * 70)
    with open(CALIB_DIR / "neutral_offset.json") as f:
        current = json.load(f)

    new_offset = dict(current)
    new_offset["delta_home"] = float(delta_best[0])
    new_offset["delta_draw"] = float(delta_best[1])
    new_offset["delta_away"] = float(delta_best[2])
    new_offset["shrinkage_weight"] = float(w_best)
    new_offset["shrinkage_components"] = {
        "delta_2022_wc": delta_2022.tolist(),
        "delta_pooled_2023": delta_pooled.tolist(),
        "weight_2022": float(w_best),
        "weight_pooled": float(1 - w_best),
    }
    new_offset["computed_at"] = pd.Timestamp.now().isoformat()
    new_offset["source"] = "shrunk_offset_sweep.py (best of sweep on 2023+ neutral subset)"
    new_offset["notes"] = (
        "Shrunk offset: blends 2022-WC global delta with pooled 2023+ delta to "
        "soften over-correction. Use shrinkage_weight to see the blend ratio. "
        "Apply only at is_neutral=1 matches."
    )

    with open(CALIB_DIR / "neutral_offset.json", "w") as f:
        json.dump(new_offset, f, indent=2)
    print(f"  Wrote shrunk offset to neutral_offset.json:")
    print(f"    Δ_H={delta_best[0]:+.3f}  Δ_D={delta_best[1]:+.3f}  Δ_A={delta_best[2]:+.3f}")
    print(f"    weight_2022={w_best:.2f}  weight_pooled={1-w_best:.2f}")

    # 8. Save sweep results
    out_path = MODEL_DIR / "shrunk_offset_sweep.json"
    with open(out_path, "w") as f:
        json.dump({
            "n_matches": n,
            "delta_2022_wc": delta_2022.tolist(),
            "delta_pooled_2023": delta_pooled.tolist(),
            "best_weight_2022": w_best,
            "best_brier": best["brier"],
            "raw_brier": brier_raw,
            "sweep": results,
        }, f, indent=2)
    print(f"  Saved sweep: {out_path.relative_to(PROJECT_ROOT)}")
    print()
    print("  → Re-run validate_offset_oos.py to confirm.")


if __name__ == "__main__":
    main()
