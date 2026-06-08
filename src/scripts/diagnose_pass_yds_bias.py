#!/usr/bin/env python3
"""Diagnose PASS_YDS model -5.2% under-prediction bias.

Key insight from v1: mu predictions are fine (mean residual = -0.58).
The -5.2% bias comes from the distribution mapping (NB → P(>=line)),
not from mu. This script isolates QB data and compares normal CDF
vs NB vs BetaCal probabilities vs actual outcomes.
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config.settings import CONFIG_DIR, PROJECT_ROOT
from src.features.nfl import NFLFeatureEngineer
from src.models.distributions import p_ge_stat, NB_STATS
from src.models.calibrator import BetaCalibrator
import toml, lightgbm as lgb
from scipy.stats import norm

MODEL_DIR = PROJECT_ROOT / "models" / "nfl"
STAT_NAME = "PASS_YDS"


def main():
    print("Loading PASS_YDS model and data...", flush=True)
    cache_path = PROJECT_ROOT / "data" / "nfl_cache" / "weekly.parquet"
    cfg = toml.load(CONFIG_DIR / "nfl.toml")
    from src.config.settings import SportConfig
    scfg = SportConfig(name="nfl", display_name="NFL",
                       rolling_windows=cfg["features"]["rolling_windows"],
                       recency_decay=cfg["features"].get("recency_decay", 0.001))
    fe = NFLFeatureEngineer(scfg)
    all_games = pd.read_parquet(cache_path)
    if "player_name" not in all_games.columns and "player_display_name" in all_games.columns:
        all_games["player_name"] = all_games["player_display_name"]
    featured = fe.build_features(all_games)

    # Merge raw stats
    stat_cols = ["passing_yards", "passing_tds", "pass_attempts",
                 "position", "player_name", "team_abbr", "season", "week",
                 "spread_line", "total_line"]
    raw_keep = [c for c in stat_cols if c in all_games.columns]
    all_games["game_date"] = pd.to_datetime(all_games["game_date"])
    featured["game_date"] = pd.to_datetime(featured["game_date"])
    if all_games["player_id"].dtype != featured["player_id"].dtype:
        all_games["player_id"] = all_games["player_id"].astype(str)
        featured["player_id"] = featured["player_id"].astype(str)
    featured = featured.merge(all_games[["player_id", "game_date"] + raw_keep], on=["player_id", "game_date"], how="left")

    # Filter to QBs: either position == "QB" or pass_attempts > 0
    if "position" in featured.columns:
        # Normalize position
        featured["position"] = featured["position"].astype(str).str.upper().str.strip()
        qb_mask = featured["position"] == "QB"
    else:
        qb_mask = featured.get("pass_attempts", pd.Series(0)).fillna(0) > 0
    
    print(f"QB rows: {qb_mask.sum()} / {len(featured)}", flush=True)
    
    df = featured[qb_mask].copy()
    target_col = "passing_yards"
    df = df.dropna(subset=[target_col]).copy()
    print(f"QB rows with passing_yards: {len(df)}", flush=True)

    # Load model
    model_path = MODEL_DIR / "lgb_pass_yds.txt"
    meta_path = MODEL_DIR / "lgb_pass_yds.meta.json"
    std_path = MODEL_DIR / "lgb_pass_yds.std.json"
    cal_path = MODEL_DIR / "lgb_pass_yds_beta_cal.json"

    model = lgb.Booster(model_file=str(model_path))
    with open(meta_path) as f:
        meta = json.load(f)
    with open(std_path) as f:
        std_data = json.load(f)

    residual_std = std_data.get("residual_std", meta.get("residual_std", 29.85))
    model_features = model.feature_name()
    beta_cal = BetaCalibrator.load(cal_path)

    print(f"Model: R²={meta['r2']:.3f}, residual_std={residual_std:.3f}")
    print(f"BetaCal: a={beta_cal.a:.3f}, b={beta_cal.b:.3f}, c={beta_cal.c:.3f}, fitted={beta_cal._fitted}")

    # Build feature matrix
    X = df[model_features].copy() if all(c in df.columns for c in model_features) else pd.DataFrame()
    if X.empty:
        # Fall back to available features
        available = [c for c in model_features if c in df.columns]
        X = df[available].copy()
        for c in model_features:
            if c not in X.columns:
                X[c] = 0.0
        X = X[model_features]
    X = X.fillna(0)

    y = df[target_col].values

    # Temporal 80/20 split
    dates = pd.to_datetime(df["game_date"])
    sort_idx = dates.argsort()
    X_sorted = X.iloc[sort_idx]
    y_sorted = y[sort_idx]
    df_sorted = df.iloc[sort_idx]
    split = int(len(X_sorted) * 0.8)
    X_test = X_sorted.iloc[split:]
    y_test = y_sorted[split:]
    df_test = df_sorted.iloc[split:]

    mu = model.predict(X_test.fillna(0))

    n = len(mu)
    print(f"\n=== QB-Only Test Set ===")
    print(f"Size: {n}")
    print(f"y mean: {y_test.mean():.1f}, y max: {y_test.max():.1f}, y std: {y_test.std():.1f}")
    print(f"mu mean: {mu.mean():.1f}, mu std: {mu.std():.1f}")
    print(f"Mean residual: {(mu - y_test).mean():+.3f}")

    # 1. Bias by predicted mu (QB-only bins)
    print(f"\n{'='*60}")
    print(f"1. BIAS BY PREDICTED MU (QB only)")
    print(f"{'='*60}")
    mu_bins = list(range(100, 401, 25))
    for i in range(len(mu_bins) - 1):
        lo, hi = mu_bins[i], mu_bins[i + 1]
        mask = (mu >= lo) & (mu < hi)
        if mask.sum() < 5:
            continue
        bias = mu[mask].mean() - y_test[mask].mean()
        pct = bias / y_test[mask].mean() * 100
        print(f"  mu [{lo:.0f}-{hi:.0f}): n={mask.sum():3d}, mu_avg={mu[mask].mean():.1f}, "
              f"y_avg={y_test[mask].mean():.1f}, bias={bias:+.2f} ({pct:+.1f}%)")

    # 2. Distribution mapping comparison (per-line, per-row)
    print(f"\n{'='*60}")
    print(f"2. DISTRIBUTION MAPPING: NORMAL CDF vs NB vs BetaCal")
    print(f"{'='*60}")
    for line_val in [150, 175, 200, 225, 250, 275, 300, 325, 350]:
        # Per-row normal CDF
        p_norm_arr = np.array([
            1.0 - norm.cdf((line_val - 0.5 - mu[i]) / max(residual_std, 0.3))
            for i in range(n)
        ])
        # Per-row NB
        p_nb_arr = np.array([
            p_ge_stat(STAT_NAME, mu[i], residual_std, line_val)
            for i in range(n)
        ])
        # BetaCal
        if beta_cal._fitted:
            p_beta_arr = np.array([beta_cal(p) for p in p_norm_arr])
        else:
            p_beta_arr = p_norm_arr
        # Actual
        p_act = (y_test >= line_val).mean()

        print(f"  line={line_val:3d}: P_norm={p_norm_arr.mean():.3f} P_nb={p_nb_arr.mean():.3f} "
              f"P_beta={p_beta_arr.mean():.3f} P_act={p_act:.3f}"
              f"  bias_norm={p_norm_arr.mean()-p_act:+.3f} bias_nb={p_nb_arr.mean()-p_act:+.3f} bias_beta={p_beta_arr.mean()-p_act:+.3f}")

    # 3. Bias by player (top QBs)
    print(f"\n{'='*60}")
    print(f"3. BIAS BY QB")
    print(f"{'='*60}")
    if "player_name" in df_test.columns:
        for pname in df_test["player_name"].value_counts().head(15).index:
            mask = df_test["player_name"] == pname
            if mask.sum() < 3:
                continue
            b = mu[mask].mean() - y_test[mask].mean()
            pct = b / y_test[mask].mean() * 100 if y_test[mask].mean() > 0 else 0
            print(f"  {pname:25s}: n={mask.sum():2d}, mu={mu[mask].mean():.1f}, "
                  f"y={y_test[mask].mean():.1f}, bias={b:+.1f} ({pct:+.1f}%)")

    # 4. Is NB actually worse than normal CDF for PASS_YDS?
    print(f"\n{'='*60}")
    print(f"4. WHICH DISTRIBUTION FITS BETTER?")
    print(f"{'='*60}")
    from scipy.stats import ks_2samp
    print(f"  PASS_YDS is a continuous yardage stat — NB is designed for discrete counts.")
    print(f"  Check: does the NB or Normal CDF produce better-calibrated probabilities?")
    print(f"  Look at section 2 above. If P_norm is closer to P_act than P_nb is,")
    print(f"  then the fix is to use normal CDF for PASS_YDS instead of NB.")

    # 5. Fix recommendation
    print(f"\n{'='*60}")
    print(f"5. RECOMMENDED FIX")
    print(f"{'='*60}")
    print(f"  Root cause: distribution mapping (NB → P(>=line)), not mu prediction")
    print(f"  Fix options:")
    print(f"    1. Remove PASS_YDS from NB_STATS → use normal CDF (continuous stat)")
    print(f"    2. Fix BetaCal bounds (force a>0, b>0) for stable fit")
    print(f"    3. Retrain model with additional features (weather, spread, over-under)")
    print(f"  Checking section 2 to decide which is best...")


if __name__ == "__main__":
    main()
