#!/usr/bin/env python3
"""Backtesting framework for F5 and MLB player prop models.

Evaluates historical predictions vs actual outcomes to answer:
- Are our probabilities well-calibrated? (predicted 30% ↔ actual 30%)
- Which models actually beat the naive baseline (predicting the prior)?
- What are the Brier scores and calibration errors?

Note: PnL simulation requires historical Kalshi market prices, which we don't
have yet. Focus on calibration quality as the primary trust metric.

Usage:
    python -m src.scripts.backtest_models --f5       # Backtest F5 model
    python -m src.scripts.backtest_models --props     # Backtest MLB player props (SO, HR, TB, HRR)
    python -m src.scripts.backtest_models --all       # Run all backtests
"""
import sys, json, warnings, re
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from src.config.settings import PROJECT_ROOT

MODEL_DIR = PROJECT_ROOT / "models" / "mlb"
CALIB_DIR = MODEL_DIR / "calibration"
CACHE_DIR = PROJECT_ROOT / "data/cache/mlb"
WC_CACHE = PROJECT_ROOT / "data/cache/world_cup"

# ── helpers ──────────────────────────────────────────────────────────────

def _calibration_summary(preds, actuals, name="model", bins=10):
    """Compare predicted probabilities to actual frequencies."""
    results = []
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        mask = (preds >= lo) & (preds < hi)
        n = int(mask.sum())
        if n >= 5:
            avg_pred = float(preds[mask].mean())
            actual_rate = float(actuals[mask].mean())
            results.append({
                "bin": f"{lo:.0%}-{hi:.0%}",
                "n": n,
                "avg_pred": round(avg_pred, 3),
                "actual_rate": round(actual_rate, 3),
                "error": round(actual_rate - avg_pred, 3),
            })
    return results


def _expected_pnl(preds, market_probs, actuals, kelly_frac=0.25):
    """Simulate quarter-Kelly betting: return total PnL and ROI."""
    edges = preds - market_probs
    bankroll = 100.0
    trades = 0
    wins = 0
    for i in range(len(preds)):
        if edges[i] < 0.05:  # min 5% edge threshold
            continue
        if not (0.10 <= market_probs[i] <= 0.80):
            continue
        price = max(0.01, market_probs[i])
        kelly = min(edges[i] / max(0.001, 1 - market_probs[i]), 0.03) * kelly_frac
        stake = min(kelly * bankroll, bankroll * 0.03)
        contracts = max(1, int(stake / price))
        cost = contracts * price
        if cost > bankroll * 0.05:
            continue
        trades += 1
        if actuals[i]:
            payout = contracts * 1.0  # each contract pays $1
            bankroll += payout - cost
            wins += 1
        else:
            bankroll -= cost
    roi = (bankroll - 100.0) / 100.0 if trades > 0 else 0.0
    return {
        "starting_bankroll": 100.0,
        "ending_bankroll": round(bankroll, 2),
        "total_roi": round(roi, 4),
        "trades": trades,
        "wins": wins,
        "win_rate": round(wins / trades, 3) if trades > 0 else 0,
    }


# ── F5 Backtest ─────────────────────────────────────────────────────────

def backtest_f5():
    """Backtest F5 multiclass model on historical data.
    
    Uses the f5_outcomes.csv (game_pk, f5_away_runs, f5_home_runs, f5_outcome)
    and cached game logs to replay the model on past games.
    """
    print("=" * 65)
    print("  F5 MODEL BACKTEST")
    print("=" * 65)
    
    # Load model
    import lightgbm as lgb
    f5_path = MODEL_DIR / "f5_multiclass_v2.txt"
    meta_path = MODEL_DIR / "f5_multiclass_v2.meta.json"
    if not f5_path.exists():
        print("  No F5 v2 model found.")
        return
    model = lgb.Booster(model_file=str(f5_path))
    with open(meta_path) as f:
        meta = json.load(f)
    feature_cols = meta["features"]
    print(f"  Model: {meta.get('n_features', '?')} features, {meta.get('n_train', 0) + meta.get('n_test', 0)} samples")
    
    # Load F5 outcomes
    f5_path_csv = CACHE_DIR / "f5_outcomes.csv"
    if not f5_path_csv.exists():
        print("  No f5_outcomes.csv found.")
        return
    f5_outcomes = pd.read_csv(f5_path_csv, low_memory=False)
    f5_outcomes.columns = [c.strip().lower().replace(" ", "_") for c in f5_outcomes.columns]
    print(f"  Loaded {len(f5_outcomes)} F5 outcomes")
    
    # Load pitcher features
    from src.features.mlb import MLBFeatureEngineer
    import toml
    from src.config.settings import CONFIG_DIR, SportConfig
    
    cfg = toml.load(CONFIG_DIR / "mlb.toml")
    scfg = SportConfig(name="mlb", display_name="MLB",
                       rolling_windows=cfg["features"]["rolling_windows"],
                       recency_decay=0.001)
    fe = MLBFeatureEngineer(scfg)
    
    cache_files = sorted(CACHE_DIR.glob("game_logs_*.parquet"))
    if not cache_files:
        print("  No cached game logs.")
        return
    
    all_games = pd.concat([pd.read_parquet(f) for f in cache_files], ignore_index=True)
    featured = fe.build_features(all_games)
    pitchers = featured[featured["position"] == "P"].copy() if "position" in featured.columns else featured.copy()
    print(f"  {len(pitchers)} pitcher-game rows")
    
    # Build game team lookup using home_or_away
    teams_hoa = pitchers[["game_pk", "team_abbr", "game_date", "home_or_away"]].drop_duplicates(
        subset=["game_pk", "team_abbr"]
    ).copy()
    teams_hoa["is_home"] = teams_hoa["home_or_away"].str.upper().str[0] == "H"
    game_lookup = {}
    for gpk, grp in teams_hoa.groupby("game_pk"):
        home_teams = grp[grp["is_home"]]["team_abbr"].unique()
        away_teams = grp[~grp["is_home"]]["team_abbr"].unique()
        if len(home_teams) >= 1 and len(away_teams) >= 1:
            game_lookup[gpk] = {
                "away": away_teams[0], "home": home_teams[0],
                "date": grp["game_date"].iloc[0]
            }
    print(f"  Game lookup: {len(game_lookup)} games")
    
    # F5 outcome mapping
    outcome_map = {"AWAY": 0, "HOME": 1, "TIE": 2}
    PARK_FACTOR_K = {
        "SD": 1.08, "SEA": 1.06, "NYM": 1.04, "MIA": 1.03, "CLE": 1.02,
        "OAK": 1.02, "TB": 1.01, "SF": 1.01, "WSH": 1.00, "DET": 1.00,
        "MIL": 0.99, "BAL": 0.99, "KC": 0.99, "MIN": 0.99, "PIT": 0.99,
        "LAA": 0.98, "PHI": 0.98, "CIN": 0.98, "ATL": 0.97, "CHC": 0.97,
        "TEX": 0.97, "BOS": 0.97, "TOR": 0.96, "HOU": 0.96, "STL": 0.96,
        "ARI": 0.95, "NYY": 0.95, "LAD": 0.94, "CWS": 0.93, "COL": 0.88,
    }
    
    # Replay each F5 outcome
    all_preds = []  # list of {away_prob, home_prob, tie_prob, actual_outcome}
    matched, skipped = 0, 0
    for _, game in f5_outcomes.iterrows():
        gpk = game["game_pk"]
        if gpk not in game_lookup:
            skipped += 1
            continue
        info = game_lookup[gpk]
        away_code, home_code = info["away"], info["home"]
        game_date = info["date"]
        gs = game_date - pd.Timedelta(hours=4)
        ge = game_date + pd.Timedelta(hours=3)
        ap = pitchers[(pitchers["game_date"] >= gs) & (pitchers["game_date"] <= ge) &
                      (pitchers["team_abbr"].str.upper() == away_code.upper())].sort_values("game_date")
        hp = pitchers[(pitchers["game_date"] >= gs) & (pitchers["game_date"] <= ge) &
                      (pitchers["team_abbr"].str.upper() == home_code.upper())].sort_values("game_date")
        if ap.empty or hp.empty:
            skipped += 1
            continue
        away_p, home_p = ap.iloc[-1], hp.iloc[-1]
        
        # Build feature vector matching model expectations
        row = {"park_factor_k": PARK_FACTOR_K.get(home_code, 1.0)}
        for col in away_p.index:
            if isinstance(col, str):
                if any(col.endswith(f"_avg_{w}") for w in [7, 14, 30]):
                    row[f"a_{col}"] = away_p[col]
                    row[f"h_{col}"] = home_p[col]
                elif col.endswith("_ewm"):
                    row[f"a_{col}"] = away_p[col]
                    row[f"h_{col}"] = home_p[col]
        
        # Predict
        vec = np.array([row.get(c, 0) for c in feature_cols]).reshape(1, -1).astype(float)
        preds = model.predict(vec)[0]
        
        outcome_str = str(game.get("f5_outcome", "TIE")).strip().upper()
        actual = outcome_map.get(outcome_str, 2)
        
        all_preds.append({
            "game_pk": gpk,
            "away_prob": float(preds[0]),
            "home_prob": float(preds[1]),
            "tie_prob": float(preds[2]),
            "actual": actual,
        })
        matched += 1
    
    print(f"  Matched: {matched}, Skipped: {skipped}")
    
    if not all_preds:
        print("  No predictions generated.")
        return
    
    df = pd.DataFrame(all_preds)
    
    # Per-outcome calibration
    for cls_idx, cls_name in enumerate(["AWAY", "HOME", "TIE"]):
        col = f"{cls_name.lower()}_prob"
        preds = df[col].values
        actuals = (df["actual"].values == cls_idx).astype(int)
        brier = float(np.mean((preds - actuals) ** 2))
        
        # Naive baseline: predict the prior
        prior = actuals.mean()
        naive_brier = float(np.mean((prior - actuals) ** 2))
        
        cal = _calibration_summary(preds, actuals, name=cls_name)
        
        print(f"\n  {cls_name} — Brier: {brier:.4f} (naive: {naive_brier:.4f})")
        if brier < naive_brier:
            print(f"    ✅ Beats naive baseline by {(naive_brier - brier) / naive_brier:.0%}")
        else:
            print(f"    ❌ Worse than naive by {(brier - naive_brier) / naive_brier:.0%}")
        print(f"    Calibration bins:")
        for b in cal[:6]:
            marker = "✅" if abs(b["error"]) < 0.03 else ("⚠️" if abs(b["error"]) < 0.08 else "❌")
            print(f"      {b['bin']:12s} n={b['n']:4d} pred={b['avg_pred']:.0%} actual={b['actual_rate']:.0%} "
                  f"err={b['error']:+.0%} {marker}")
    
    # PnL simulation skipped — requires historical Kalshi market prices


# ── MLB Player Props Backtest ───────────────────────────────────────────

MAX_PLAYERS = 100  # sampled-per-model limit to prevent timeout

def backtest_mlb_props(limit=MAX_PLAYERS):
    """Backtest MLB player prop models (SO, HR, TB, HRR).
    
    For each player-game in the test set, predict P(stat >= line) for
    common lines and compare to actual outcome.
    """
    print("=" * 65)
    print("  MLB PLAYER PROPS BACKTEST")
    print("=" * 65)
    
    from src.scripts.kalshi_mlb_unified import MARKET_TYPES, _load_regressor, _p_ge_line, _recency_check
    from src.execution.mlb_predictor import MLBLinePredictor
    import toml
    from src.config.settings import CONFIG_DIR, SportConfig
    
    # Load features
    cfg = toml.load(CONFIG_DIR / "mlb.toml")
    scfg = SportConfig(name="mlb", display_name="MLB",
                       rolling_windows=cfg["features"]["rolling_windows"],
                       recency_decay=0.001)
    predictor = MLBLinePredictor(scfg)
    predictor.load_data()
    latest = predictor._latest_features
    
    if latest is None or latest.empty:
        print("  No player features.")
        return
    
    # Map model names to the stat columns in game logs
    STAT_COL_MAP = {
        "SO": "so",
        "HR": "hr",
        "TB": "tb",
        "H_R_RBI": "h_r_rbi",
    }
    
    for mt in MARKET_TYPES:
        mname = mt["name"]
        model_name = mt["model_name"]
        pos = mt["position"]
        desc = mt["desc"]
        stat_col = STAT_COL_MAP.get(model_name, model_name.lower())
        
        # Load model
        m, s = _load_regressor(model_name)
        if m is None:
            print(f"\n  {mname:5s} ({desc:20s}): No model — skipping")
            continue
        
        # Get players matching position filter, sample to limit
        if pos == "pitcher":
            players = latest[latest.get("position", "") == "P"].copy()
        elif pos == "hitter":
            players = latest[latest.get("position", "") != "P"].copy()
        else:
            players = latest.copy()
        
        if players.empty:
            print(f"\n  {mname:5s} ({desc:20s}): No players found")
            continue
        
        # Sample to limit to prevent timeout
        if len(players) > limit:
            players = players.sample(n=limit, random_state=42)
        
        # Test a range of common lines
        if model_name == "SO":
            test_lines = [3, 4, 5, 6, 7]
        elif model_name == "HR":
            test_lines = [1]
        elif model_name == "TB":
            test_lines = [1, 2, 3]
        elif model_name == "H_R_RBI":
            test_lines = [1, 2, 3, 4]
        else:
            test_lines = [1, 2]
        
        for line_val in test_lines:
            preds_list = []
            actuals_list = []
            
            for _, row in players.iterrows():
                try:
                    pname = row.get("player_name", "")
                    if not pname:
                        continue
                    
                    # Get model prediction
                    p_yes, mu = _p_ge_line(row, m, s, line_val, stat_name=model_name)
                    
                    # Get actual rate from 2026 game logs
                    actual_rate, _, _ = _recency_check(pname, line_val, stat_col=stat_col)
                    if actual_rate < 0:
                        continue
                    
                    preds_list.append(p_yes)
                    actuals_list.append(actual_rate)
                except Exception:
                    continue
            
            if len(preds_list) < 20:
                continue
            
            preds_arr = np.array(preds_list)
            actuals_arr = np.array(actuals_list)
            brier = float(np.mean((preds_arr - actuals_arr) ** 2))
            prior = float(actuals_arr.mean())
            naive_brier = float(np.mean((prior - actuals_arr) ** 2))
            
            cal = _calibration_summary(preds_arr, actuals_arr, name=f"{mname} {line_val}+")
            
            print(f"\n  {mname:5s} {line_val}+ {desc:15s}: n={len(preds_list):4d} "
                  f"Brier={brier:.4f} (naive={naive_brier:.4f})")
            if brier < naive_brier:
                improvement = (naive_brier - brier) / naive_brier
                print(f"    ✅ Beats naive by {improvement:.0%}")
            else:
                print(f"    ❌ Worse than naive")
            for b in cal[:4]:
                marker = "✅" if abs(b["error"]) < 0.03 else ("⚠️" if abs(b["error"]) < 0.08 else "❌")
                print(f"      {b['bin']:12s} n={b['n']:4d} pred={b['avg_pred']:.0%} actual={b['actual_rate']:.0%} "
                      f"err={b['error']:+.0%} {marker}")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--f5", action="store_true", help="Backtest F5 model")
    parser.add_argument("--props", action="store_true", help="Backtest MLB player props")
    parser.add_argument("--all", action="store_true", help="Run all backtests")
    parser.add_argument("--limit", type=int, default=MAX_PLAYERS, help=f"Max players to sample per model (default: {MAX_PLAYERS})")
    args = parser.parse_args()
    
    run_all = args.all or not (args.f5 or args.props)
    
    if args.f5 or run_all:
        backtest_f5()
        print()
    
    if args.props or run_all:
        backtest_mlb_props(limit=args.limit)
        print()
    
    print("=" * 65)
    print("  Backtest complete")
    print("=" * 65)


if __name__ == "__main__":
    main()
