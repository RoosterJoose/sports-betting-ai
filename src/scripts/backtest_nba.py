#!/usr/bin/env python3
"""Backtest NBA regression models — compare calibrated probability vs actual outcome.

For each scanner stat (PTS, REB, AST, STL, BLK, etc.), loads the trained XGBoost
model + BetaCal, computes P(stat >= line) via distribution-appropriate CDF → BetaCal,
and compares to actual empirical rates across all line values.

Usage:
    python -m src.scripts.backtest_nba
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config.settings import CONFIG_DIR, PROJECT_ROOT
from src.features.nba import NBAFeatureEngineer
from src.models.calibrator import BetaCalibrator
from src.models.distributions import p_ge_stat
import toml

MODEL_DIR = PROJECT_ROOT / "models" / "nba"
CACHE_PATH = PROJECT_ROOT / "data" / "nba_cache" / "game_logs_v14.parquet"

# NBA stat types to backtest
# (stat_name, display_name) — each maps to a model .json file
STAT_DEFS = [
    ("PTS", "PTS"),
    ("REB", "REB"),
    ("AST", "AST"),
    ("STL", "STL"),
    ("BLK", "BLK"),
    ("TOV", "TOV"),
    ("FG3M", "3PT"),
    ("FG3A", "FG3A"),
    ("FGM", "FGM"),
    ("FTM", "FTM"),
    ("FTA", "FTA"),
    ("PR", "PR"),
    ("PA", "PA"),
    ("RA", "RA"),
    ("PRA", "PRA"),
    ("SB", "SB"),
    ("FPTS", "FPTS"),
]


def load_features():
    """Load cached NBA data and build features."""
    if not CACHE_PATH.exists():
        print("No cached NBA data at", CACHE_PATH)
        return None

    cfg = toml.load(CONFIG_DIR / "nba.toml")
    from src.config.settings import SportConfig
    scfg = SportConfig(
        name="nba", display_name="NBA",
        rolling_windows=cfg["features"]["rolling_windows"],
        recency_decay=cfg["features"].get("recency_decay", 0.001),
    )
    fe = NBAFeatureEngineer(scfg)
    all_games = pd.read_parquet(CACHE_PATH)
    featured = fe.build_features(all_games)

    # Merge raw stat columns back (feature engineer strips them)
    raw_stat_cols = ["pts", "reb", "ast", "stl", "blk", "tov",
                     "fg3m", "fg3a", "fgm", "fga", "ftm", "fta", "min"]
    raw_keep = [c for c in raw_stat_cols if c in all_games.columns]
    merge_cols = ["player_id", "game_date"]
    all_games["game_date"] = pd.to_datetime(all_games["game_date"])
    featured["game_date"] = pd.to_datetime(featured["game_date"])
    all_cols = merge_cols + raw_keep
    if "player_name" in all_games.columns:
        all_cols.append("player_name")
    featured = featured.merge(
        all_games[all_cols].drop_duplicates(subset=merge_cols),
        on=merge_cols, how="left"
    )

    # Compute combo stat columns for PRA, PA, PR, RA, SB, FPTS
    if all(c in featured.columns for c in ["pts", "reb", "ast"]):
        featured["pra"] = featured["pts"].fillna(0) + featured["reb"].fillna(0) + featured["ast"].fillna(0)
        featured["pa"] = featured["pts"].fillna(0) + featured["ast"].fillna(0)
        featured["pr"] = featured["pts"].fillna(0) + featured["reb"].fillna(0)
        featured["ra"] = featured["reb"].fillna(0) + featured["ast"].fillna(0)
    if all(c in featured.columns for c in ["stl", "blk"]):
        featured["sb"] = featured["stl"].fillna(0) + featured["blk"].fillna(0)
    if all(c in featured.columns for c in ["pts", "reb", "ast", "stl", "blk", "tov"]):
        featured["fpts"] = (featured["pts"].fillna(0) * 1.0 +
                            featured["reb"].fillna(0) * 1.2 +
                            featured["ast"].fillna(0) * 1.5 +
                            featured["stl"].fillna(0) * 3.0 +
                            featured["blk"].fillna(0) * 3.0 -
                            featured["tov"].fillna(0) * 1.0)
    return featured


def backtest_stat(featured, stat_name, display_name):
    """Backtest a single NBA stat."""
    import xgboost as xgb

    mn = stat_name.lower()
    model_path = MODEL_DIR / f"{mn}.json"
    meta_path = MODEL_DIR / f"{mn}.metrics.json"
    cal_path = MODEL_DIR / f"{mn}_beta_cal.json"

    if not model_path.exists():
        print(f"  {stat_name:6s}: Model not found")
        return

    # Load model
    model = xgb.XGBRegressor()
    model.load_model(str(model_path))

    # Get feature names from model JSON
    try:
        with open(model_path) as f:
            mdata = json.load(f)
        feature_names = mdata.get("learner", {}).get("feature_names", [])
    except Exception:
        feature_names = []

    # Load sigma (residual_std or fallback to MAE)
    sigma = 1.0
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        sigma = meta.get("residual_std", meta.get("mae", 1.0))

    # Load BetaCal
    beta_cal = BetaCalibrator.load(cal_path)
    beta_str = (f"BetaCal: a={beta_cal.a:.3f} b={beta_cal.b:.3f} c={beta_cal.c:.3f}"
                if beta_cal._fitted else "No BetaCal")

    # Target column
    raw_col = stat_name.lower()
    if raw_col not in featured.columns:
        print(f"  {stat_name:6s}: column '{raw_col}' not found")
        return

    df = featured.dropna(subset=[raw_col]).copy()
    if len(df) < 100:
        print(f"  {stat_name:6s}: only {len(df)} rows")
        return

    y = df[raw_col].values
    available = [c for c in feature_names if c in df.columns]
    if not available:
        print(f"  {stat_name:6s}: no matching features")
        return

    X = df[available].copy()
    X = X.fillna(0)

    # Temporal 80/20 split
    if "game_date" in df.columns:
        dates = pd.to_datetime(df["game_date"])
        sort_idx = dates.argsort()
        X = X.iloc[sort_idx]
        y = y[sort_idx]
        split = int(len(X) * 0.8)
        X_test = X.iloc[split:]
        y_test = y[split:]
    else:
        split = int(len(X) * 0.8)
        X_test = X.iloc[split:]
        y_test = y[split:]

    # Predict
    mu = model.predict(X_test.values)

    # Determine line range
    y_mean = float(np.mean(y_test))
    y_max = float(np.max(y_test))
    low = max(1, int(max(y_mean * 0.2, 0)))
    high = min(int(y_max) + 1, max(int(y_mean * 3.0) + 1, int(y_mean) + 8))
    if low >= high:
        print(f"  {stat_name:6s}: no valid line range [{low}, {high})")
        return

    lines = list(range(low, high))

    print(f"\n  {stat_name:6s}: sigma={sigma:.2f}  test_n={len(y_test)}  "
          f"y_mean={y_mean:.1f}  y_max={y_max:.0f}  lines=[{low}..{high})")
    print(f"         {beta_str}")

    results = []
    for line_val in lines:
        # Vectorized: compute P(stat >= line) for all test rows
        mu_clipped = np.maximum(mu, 0)
        p_raw = p_ge_stat(stat_name, mu_clipped, max(sigma, 0.3), line_val)
        p_raw = np.clip(p_raw, 0.001, 0.999)

        if beta_cal._fitted:
            p_cal = beta_cal.calibrate(p_raw)
        else:
            p_cal = p_raw
        p_cal = np.clip(p_cal, 0.001, 0.999)

        actuals = (y_test >= line_val).astype(float)
        p_model = float(np.mean(p_cal))
        p_actual = float(np.mean(actuals))
        n = len(actuals)
        if n < 10:
            continue

        brier = float(np.mean((p_cal - actuals) ** 2))
        naive_brier = float(np.mean((p_actual - actuals) ** 2))
        bias = p_model - p_actual

        results.append({
            "line": line_val, "p_model": round(p_model, 4),
            "p_actual": round(p_actual, 4), "bias": round(bias, 4),
            "brier": round(brier, 4), "naive_brier": round(naive_brier, 4),
            "n": n,
        })

    if not results:
        return

    # Display all lines
    print(f"\n         {'Line':>4s}  {'P_cal':>6s}  {'P_act':>6s}  {'Bias':>7s}  {'Brier':>7s}  {'n':>5s}")
    print(f"         {'-'*4:>4s}  {'-'*6:>6s}  {'-'*6:>6s}  {'-'*7:>7s}  {'-'*7:>7s}  {'-'*5:>5s}")
    for r in results:
        marker = "OK" if abs(r["bias"]) < 0.03 else ("WARN" if abs(r["bias"]) < 0.08 else "BAD")
        print(f"         {r['line']:>4d}+  {r['p_model']:>5.0%}  "
              f"{r['p_actual']:>5.0%}  {r['bias']:>+5.1%}  "
              f"{r['brier']:>6.4f}  {r['n']:>5d}  {marker}")

    # Aggregate
    mean_bias = float(np.mean([r["bias"] for r in results]))
    mean_abs_bias = float(np.mean([abs(r["bias"]) for r in results]))
    mean_brier = float(np.mean([r["brier"] for r in results]))
    beats_naive = sum(1 for r in results if r["brier"] < r["naive_brier"])
    total = len(results)

    print(f"\n         {'-'*55}")
    print(f"         Mean Bias: {mean_bias:+.1%}  Mean |Bias|: {mean_abs_bias:.1%}  "
          f"Mean Brier: {mean_brier:.4f}")
    print(f"         Beats naive baseline: {beats_naive}/{total} ({beats_naive/total:.0%})")


def main():
    print("=" * 70)
    print("  NBA MODEL BACKTEST — Calibrated Probability vs Actual Outcome")
    print("=" * 70)

    print("\n1. Loading features...")
    featured = load_features()
    if featured is None:
        return
    print(f"   {len(featured)} rows")

    print("\n2. Backtesting each stat...")
    for stat_name, display in STAT_DEFS:
        backtest_stat(featured, stat_name, display)

    print(f"\n{'=' * 70}")
    print("  Backtest complete")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
