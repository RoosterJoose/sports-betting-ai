#!/usr/bin/env python3
"""Paper-trade simulation: 2023+ neutral-venue subset, raw vs offset.

Answers: does the +0.7% Brier improvement from the empirical offset translate
to positive ROI on simulated bets using current scanner thresholds?

Strategy
--------
- Replays 2023+ matches in chronological order
- For each match, computes raw model probs AND offset-applied probs
- Applies the same thresholds as scan_wc.py:
    * edge_pct > 15% (model_p vs market_p)
    * model_p > 0.15
    * phantom-edge filter: skip if model_p > 3*fair_p AND fair_p < 0.10
    * skip if mkt_p < 0.02 or > 0.90
- For paper-trading on historical data, we DON'T have live Kalshi prices.
  Two market-price scenarios are simulated:
    A. Market = uniform 33/33/33 (no info) — "edge is just model_p - 1/3"
    B. Market = raw model prob (perfectly prices our edge to zero) —
       "offset only helps if it actually shifts resolution"
- Sizing: quarter-Kelly capped at 3% of bankroll per bet
- Payout: even money (no spread, no fees) for cleanest signal
- Starting bankroll: $100, log all trades

Usage:
    python -m src.scripts.paper_trade_wc
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


# Reuse the optimized team-history builder from validate_offset_oos.py
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
    """Form features for one team at a cutoff date."""
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


def simulate(raw_probs_arr, offset_probs_arr, actuals, market_probs_arr, *,
             bankroll_start=100.0, min_edge_pct=15.0, min_model_p=0.15,
             max_stake_pct=0.03, kelly_frac=0.25):
    """Run paper-trading simulation on (raw, offset, actual, market) arrays.

    Returns dict with raw_results, offset_results, plus per-trade log.
    """
    n = len(actuals)

    def _run(probs_all, label):
        bankroll = bankroll_start
        peak = bankroll
        max_dd = 0.0
        trades = []
        for i in range(n):
            probs = probs_all[i]
            mp = market_probs_arr[i]
            actual = actuals[i]

            # Best edge among the 3 outcomes
            best_edge_pct = -1e9
            best_outcome = -1
            best_model_p = 0
            best_market_p = 0
            for k in range(3):
                model_p = probs[k]
                market_p = mp[k]
                if market_p <= 0:
                    continue
                edge_pct = (model_p - market_p) / market_p * 100
                # Phantom-edge filter (from scan_wc.py)
                if model_p > 3.0 * market_p and market_p < 0.10:
                    continue
                # Skip if market is too extreme
                if market_p > 0.90 or market_p < 0.02:
                    continue
                if edge_pct > best_edge_pct:
                    best_edge_pct = edge_pct
                    best_outcome = k
                    best_model_p = model_p
                    best_market_p = market_p

            if best_outcome == -1:
                continue
            if best_edge_pct < min_edge_pct:
                continue
            if best_model_p < min_model_p:
                continue

            # Quarter-Kelly: f* = (bp - q) / b, b = (1 - market_p) / market_p
            # For even-money payout, b = 1, f* = model_p - market_p
            edge = best_model_p - best_market_p
            kelly_full = edge  # for even-money
            stake_pct = min(kelly_full * kelly_frac, max_stake_pct)
            stake = stake_pct * bankroll
            if stake < 0.01:
                continue

            won = int(actual == best_outcome)
            if won:
                bankroll += stake
            else:
                bankroll -= stake

            peak = max(peak, bankroll)
            dd = (peak - bankroll) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

            trades.append({
                "i": i, "outcome_idx": best_outcome, "outcome": OUTCOME_LABELS[best_outcome],
                "model_p": float(best_model_p), "market_p": float(best_market_p),
                "edge_pct": float(best_edge_pct), "stake": float(stake),
                "won": won, "actual": int(actual),
                "actual_label": OUTCOME_LABELS[actual],
                "bankroll": float(bankroll),
            })

        n_trades = len(trades)
        n_wins = sum(1 for t in trades if t["won"])
        wr = n_wins / n_trades if n_trades > 0 else 0
        roi = (bankroll - bankroll_start) / bankroll_start * 100
        return {
            "label": label,
            "n_trades": n_trades, "n_wins": n_wins, "win_rate": wr,
            "bankroll_start": bankroll_start, "bankroll_end": float(bankroll),
            "roi_pct": roi, "max_drawdown_pct": max_dd * 100,
            "trades": trades,
        }

    raw_result = _run(raw_probs_arr, "raw")
    off_result = _run(offset_probs_arr, "offset")
    return {"raw": raw_result, "offset": off_result}


def main():
    print("=" * 70)
    print("  PAPER-TRADING SIM — 2023+ NEUTRAL-VENUE SUBSET")
    print("  Raw vs Offset, current scanner thresholds")
    print("=" * 70)

    # 1. Load model + offset
    print("\n1. Loading model + offset...")
    import lightgbm as lgb
    model_path = MODEL_DIR / "wc_match_outcome.txt"
    meta_path = MODEL_DIR / "wc_match_outcome.meta.json"
    offset_path = CALIB_DIR / "neutral_offset.json"
    model = lgb.Booster(model_file=str(model_path))
    with open(meta_path) as f:
        meta = json.load(f)
    with open(offset_path) as f:
        offset = json.load(f)
    features = meta.get("features", [])
    cap = offset.get("cap", 0.15)
    dh = max(-cap, min(cap, offset.get("delta_home", 0.0)))
    dd = max(-cap, min(cap, offset.get("delta_draw", 0.0)))
    da = max(-cap, min(cap, offset.get("delta_away", 0.0)))
    print(f"  Model: {meta.get('n_features', '?')} features")
    print(f"  Offset: Δ_H={dh:+.3f} Δ_D={dd:+.3f} Δ_A={da:+.3f}")

    # 2. Load 2023+ test data, filter to neutral-venue rows
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
    print(f"  {len(test_df)} neutral-venue matches in 2023+")

    # 3. Precompute team history (one-time cost)
    print("\n3. Precomputing team history...")
    team_hist = _build_team_history(elo_df)
    print(f"  {len(team_hist)} teams indexed")

    # 4. Predict each match
    print("\n4. Predicting...")
    raw_probs = []
    offset_probs = []
    actuals = []
    market_uniform = []
    market_raw = []
    for _, match in test_df.iterrows():
        home = match["home_team"]
        away = match["away_team"]
        match_date = match["match_date"]

        # ELO as of match date (no look-ahead)
        pre = elo_df[elo_df["match_date"] < match_date]
        elo_ratings = {}
        if not pre.empty:
            elo_ratings.update(dict(zip(pre["home_team"].values, pre["elo_home_post"].values)))
            elo_ratings.update(dict(zip(pre["away_team"].values, pre["elo_away_post"].values)))
        elo_h = elo_ratings.get(home, 1500)
        elo_a = elo_ratings.get(away, 1500)

        # Form features
        hf = _form_for_team(team_hist, home, match_date, default_elo=elo_h)
        af = _form_for_team(team_hist, away, match_date, default_elo=elo_a)

        x = build_feature_vector(elo_h, elo_a, hf, af, "WC", features)
        probs = model.predict(x)[0]

        # Offset
        off = probs.copy()
        off[0] -= dh; off[1] -= dd; off[2] -= da
        off = np.maximum(off, 0.001)
        off = off / off.sum()

        hs, as_ = int(match["home_score"]), int(match["away_score"])
        if hs > as_: actual = 0
        elif as_ > hs: actual = 2
        else: actual = 1

        raw_probs.append(probs)
        offset_probs.append(off)
        actuals.append(actual)
        # Two market scenarios
        market_uniform.append(np.array([1/3, 1/3, 1/3]))
        market_raw.append(probs.copy())  # market = model raw (no edge unless offset shifts)

    raw_arr = np.array(raw_probs)
    off_arr = np.array(offset_probs)
    act_arr = np.array(actuals)
    mu_arr = np.array(market_uniform)
    mr_arr = np.array(market_raw)
    print(f"  {len(actuals)} matches predicted.")

    # 5. Simulate under both market scenarios
    print(f"\n{'='*70}")
    print(f"  SCENARIO A: Market = uniform 33/33/33 (no info — edge = model_p - 1/3)")
    print(f"{'='*70}")
    res_a = simulate(raw_arr, off_arr, act_arr, mu_arr)

    print(f"\n  {'Metric':<25} {'Raw':>12} {'Offset':>12} {'Δ':>10}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")
    print(f"  {'Trades':<25} {res_a['raw']['n_trades']:>12} {res_a['offset']['n_trades']:>12} "
          f"{res_a['offset']['n_trades'] - res_a['raw']['n_trades']:>+10}")
    print(f"  {'Wins':<25} {res_a['raw']['n_wins']:>12} {res_a['offset']['n_wins']:>12} "
          f"{res_a['offset']['n_wins'] - res_a['raw']['n_wins']:>+10}")
    print(f"  {'Win rate':<25} {res_a['raw']['win_rate']*100:>11.1f}% {res_a['offset']['win_rate']*100:>11.1f}% "
          f"{(res_a['offset']['win_rate']-res_a['raw']['win_rate'])*100:>+9.1f}pp")
    print(f"  {'Final bankroll':<25} ${res_a['raw']['bankroll_end']:>10.2f} ${res_a['offset']['bankroll_end']:>10.2f} "
          f"${res_a['offset']['bankroll_end']-res_a['raw']['bankroll_end']:>+9.2f}")
    print(f"  {'ROI':<25} {res_a['raw']['roi_pct']:>11.1f}% {res_a['offset']['roi_pct']:>11.1f}% "
          f"{res_a['offset']['roi_pct']-res_a['raw']['roi_pct']:>+9.1f}pp")
    print(f"  {'Max drawdown':<25} {res_a['raw']['max_drawdown_pct']:>11.1f}% {res_a['offset']['max_drawdown_pct']:>11.1f}% "
          f"{res_a['offset']['max_drawdown_pct']-res_a['raw']['max_drawdown_pct']:>+9.1f}pp")

    print(f"\n{'='*70}")
    print(f"  SCENARIO B: Market = raw model prob (offset only helps if it shifts resolution)")
    print(f"{'='*70}")
    res_b = simulate(raw_arr, off_arr, act_arr, mr_arr)

    print(f"\n  {'Metric':<25} {'Raw':>12} {'Offset':>12} {'Δ':>10}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")
    print(f"  {'Trades':<25} {res_b['raw']['n_trades']:>12} {res_b['offset']['n_trades']:>12} "
          f"{res_b['offset']['n_trades']-res_b['raw']['n_trades']:>+10}")
    print(f"  {'Wins':<25} {res_b['raw']['n_wins']:>12} {res_b['offset']['n_wins']:>12} "
          f"{res_b['offset']['n_wins']-res_b['raw']['n_wins']:>+10}")
    print(f"  {'Win rate':<25} {res_b['raw']['win_rate']*100:>11.1f}% {res_b['offset']['win_rate']*100:>11.1f}% "
          f"{(res_b['offset']['win_rate']-res_b['raw']['win_rate'])*100:>+9.1f}pp")
    print(f"  {'Final bankroll':<25} ${res_b['raw']['bankroll_end']:>10.2f} ${res_b['offset']['bankroll_end']:>10.2f} "
          f"${res_b['offset']['bankroll_end']-res_b['raw']['bankroll_end']:>+9.2f}")
    print(f"  {'ROI':<25} {res_b['raw']['roi_pct']:>11.1f}% {res_b['offset']['roi_pct']:>11.1f}% "
          f"{res_b['offset']['roi_pct']-res_b['raw']['roi_pct']:>+9.1f}pp")
    print(f"  {'Max drawdown':<25} {res_b['raw']['max_drawdown_pct']:>11.1f}% {res_b['offset']['max_drawdown_pct']:>11.1f}% "
          f"{res_b['offset']['max_drawdown_pct']-res_b['raw']['max_drawdown_pct']:>+9.1f}pp")

    # 6. Edge-threshold sweep (under scenario A — most informative)
    print(f"\n{'='*70}")
    print(f"  EDGE-THRESHOLD SWEEP (Scenario A: uniform market)")
    print(f"{'='*70}")
    print(f"  {'Min edge':>10} {'Raw trades':>11} {'Raw ROI':>10} {'Off trades':>11} {'Off ROI':>10} {'Δ ROI':>10}")
    print(f"  {'-'*10} {'-'*11} {'-'*10} {'-'*11} {'-'*10} {'-'*10}")
    for min_e in [5, 10, 15, 20, 25, 30, 50, 75, 100]:
        r = simulate(raw_arr, off_arr, act_arr, mu_arr, min_edge_pct=min_e)
        r_raw = r['raw']
        r_off = r['offset']
        print(f"  {f'>{min_e}%':>10} {r_raw['n_trades']:>11} {r_raw['roi_pct']:>9.1f}% "
              f"{r_off['n_trades']:>11} {r_off['roi_pct']:>9.1f}% "
              f"{r_off['roi_pct']-r_raw['roi_pct']:>+9.1f}pp")

    # 7. Verdict
    print(f"\n{'='*70}")
    print(f"  VERDICT — does the +0.7% Brier improvement translate to ROI?")
    print(f"{'='*70}")
    raw_roi_a = res_a["raw"]["roi_pct"]
    off_roi_a = res_a["offset"]["roi_pct"]
    delta_a = off_roi_a - raw_roi_a
    raw_roi_b = res_b["raw"]["roi_pct"]
    off_roi_b = res_b["offset"]["roi_pct"]
    delta_b = off_roi_b - raw_roi_b

    # Decision logic
    print(f"\n  Scenario A (uniform market, current scanner thresholds):")
    print(f"    Raw    ROI: {raw_roi_a:+.1f}%   (n={res_a['raw']['n_trades']} trades, "
          f"WR={res_a['raw']['win_rate']*100:.0f}%)")
    print(f"    Offset ROI: {off_roi_a:+.1f}%   (n={res_a['offset']['n_trades']} trades, "
          f"WR={res_a['offset']['win_rate']*100:.0f}%)")
    print(f"    Δ ROI:      {delta_a:+.1f}pp")
    if delta_a > 0:
        verdict_a = "✅ Offset ADDS ROI"
    elif delta_a > -2:
        verdict_a = "─ Offset is neutral"
    else:
        verdict_a = "❌ Offset HURTS ROI"
    print(f"    {verdict_a}")

    print(f"\n  Scenario B (market = raw model):")
    print(f"    Raw    ROI: {raw_roi_b:+.1f}%   (n={res_b['raw']['n_trades']} trades)")
    print(f"    Offset ROI: {off_roi_b:+.1f}%   (n={res_b['offset']['n_trades']} trades)")
    print(f"    Δ ROI:      {delta_b:+.1f}pp")
    if delta_b > 0:
        verdict_b = "✅ Offset ADDS ROI"
    elif delta_b > -2:
        verdict_b = "─ Offset is neutral"
    else:
        verdict_b = "❌ Offset HURTS ROI"
    print(f"    {verdict_b}")

    # Overall
    print()
    if delta_a > 0 and delta_b > 0:
        print(f"  ✅✅ OVERALL: Offset ADDS ROI in both scenarios — RECOMMEND KEEP")
    elif delta_a > 0 or delta_b > 0:
        print(f"  ─  OVERALL: Offset helps in one scenario — RECOMMEND KEEP (modest benefit)")
    elif delta_a > -2 and delta_b > -2:
        print(f"  ─  OVERALL: Offset is roughly neutral — safe to keep, no clear benefit")
    else:
        print(f"  ❌  OVERALL: Offset HURTS ROI — recommend disable for WC 2026")

    # 8. Save results
    out_path = MODEL_DIR / "paper_trade_2023plus_neutral.json"
    out = {
        "n_matches": int(len(actuals)),
        "scenario_a_uniform_market": {
            "raw": {k: v for k, v in res_a["raw"].items() if k != "trades"},
            "offset": {k: v for k, v in res_a["offset"].items() if k != "trades"},
            "delta_roi_pct": delta_a,
        },
        "scenario_b_market_eq_raw": {
            "raw": {k: v for k, v in res_b["raw"].items() if k != "trades"},
            "offset": {k: v for k, v in res_b["offset"].items() if k != "trades"},
            "delta_roi_pct": delta_b,
        },
        "settings": {
            "min_edge_pct": 15.0, "min_model_p": 0.15,
            "kelly_frac": 0.25, "max_stake_pct": 0.03,
            "bankroll_start": 100.0, "payout": "even money (no spread/fees)",
            "phantom_filter": "model_p > 3*market_p AND market_p < 0.10",
        },
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
