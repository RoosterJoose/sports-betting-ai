#!/usr/bin/env python3
"""Refit PASS_YDS BetaCal on QB-only test predictions.

After removing PASS_YDS from NB_STATS (now uses normal CDF), the
BetaCal needs refitting. Also refits PASS_YDS+TD which inherits bias.
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

def refit_passy(model_name, target_col, stats_to_test):
    print(f"\n{'='*60}")
    print(f"  Refitting {model_name}")
    print(f"{'='*60}")

    # Load model
    mn = model_name.lower()
    model_path = MODEL_DIR / f"lgb_{mn}.txt"
    meta_path = MODEL_DIR / f"lgb_{mn}.meta.json"
    std_path = MODEL_DIR / f"lgb_{mn}.std.json"
    cal_path = MODEL_DIR / f"lgb_{mn}_beta_cal.json"

    if not model_path.exists():
        print(f"  Model not found, skipping")
        return

    model = lgb.Booster(model_file=str(model_path))
    with open(meta_path) as f:
        meta = json.load(f)
    with open(std_path) as f:
        std_data = json.load(f)
    residual_std = std_data.get("residual_std", meta.get("residual_std", 1.0))
    model_features = model.feature_name()

    print(f"  Features: {len(model_features)}, σ={residual_std:.3f}, R²={meta['r2']:.3f}")

    # Load features
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
    stat_cols = ["passing_yards", "passing_tds", "pass_attempts", "position", "player_name"]
    raw_keep = [c for c in stat_cols if c in all_games.columns]
    all_games["game_date"] = pd.to_datetime(all_games["game_date"])
    featured["game_date"] = pd.to_datetime(featured["game_date"])
    if all_games["player_id"].dtype != featured["player_id"].dtype:
        all_games["player_id"] = all_games["player_id"].astype(str)
        featured["player_id"] = featured["player_id"].astype(str)
    featured = featured.merge(all_games[["player_id", "game_date"] + raw_keep], on=["player_id", "game_date"], how="left")

    # Filter to QBs
    if "position" in featured.columns:
        featured["position"] = featured["position"].astype(str).str.upper().str.strip()
        df = featured[featured["position"] == "QB"].copy()
    else:
        df = featured[featured.get("pass_attempts", 0).fillna(0) > 0].copy()

    df = df.dropna(subset=[target_col]).copy()
    print(f"  QB rows: {len(df)}")

    # Build feature matrix
    available = [c for c in model_features if c in df.columns]
    X = df[available].copy()
    for c in model_features:
        if c not in X.columns:
            X[c] = 0.0
    X = X[model_features].fillna(0)

    y = df[target_col].values

    # Temporal 80/20 split
    dates = pd.to_datetime(df["game_date"])
    sort_idx = dates.argsort()
    X = X.iloc[sort_idx]
    y = y[sort_idx]
    split = int(len(X) * 0.8)
    X_test = X.iloc[split:]
    y_test = y[split:]

    mu = model.predict(X_test.fillna(0))
    print(f"  Test size: {len(mu)}, mu mean: {mu.mean():.1f}, y mean: {y_test.mean():.1f}")
    print(f"  Mu bias: {mu.mean() - y_test.mean():+.2f}")

    # For each stat tested, compute raw probabilities and fit BetaCal
    for stat_name, line_min, line_max in stats_to_test:
        print(f"\n  --- {stat_name} ---")

        raw_probs = []
        outcomes = []
        for line_val in range(line_min, line_max + 1, 25):
            # Use the appropriate distribution (now normal for PASS_YDS)
            p_raw = np.array([
                p_ge_stat(stat_name, mu[i], residual_std, line_val)
                for i in range(len(mu))
            ])
            actual = (y_test >= line_val).astype(int)
            raw_probs.extend(p_raw.tolist())
            outcomes.extend(actual.tolist())

        raw_arr = np.array(raw_probs)
        out_arr = np.array(outcomes, dtype=int)

        valid = (raw_arr > 0.01) & (raw_arr < 0.99)
        if valid.sum() < 100:
            print(f"    Only {valid.sum()} valid predictions, skipping")
            continue

        raw_bias = float(np.mean(raw_arr) - np.mean(out_arr))
        print(f"    Raw bias: {raw_bias:+.3f} (n={valid.sum()})")

        # Fit BetaCal
        beta_cal = BetaCalibrator()
        beta_cal.fit(raw_arr[valid], out_arr[valid])

        cal_probs = beta_cal.calibrate(raw_arr[valid])
        cal_bias = float(np.mean(cal_probs - out_arr[valid]))
        print(f"    BetaCal: a={beta_cal.a:.3f}, b={beta_cal.b:.3f}, c={beta_cal.c:.3f}")
        print(f"    Bias: {raw_bias:+.3f} → {cal_bias:+.3f}")

        # Per-line calibration
        print(f"    Per-line:")
        for line_val in range(line_min, line_max + 1, 25):
            p_raw_line = np.array([p_ge_stat(stat_name, mu[i], residual_std, line_val) for i in range(len(mu))])
            p_cal_line = beta_cal.calibrate(p_raw_line)
            p_act_line = (y_test >= line_val).mean()
            print(f"      line={line_val:3d}: P_raw={p_raw_line.mean():.3f} "
                  f"P_cal={p_cal_line.mean():.3f} P_act={p_act_line:.3f}")

        # Save new BetaCal (overwrite old one)
        save_path = MODEL_DIR / f"lgb_{model_name.lower()}_beta_cal.json"
        beta_cal.save(save_path)
        print(f"    Saved to {save_path}")


def main():
    print("Refitting PASS_YDS and PASS_YDS+TD BetaCal")
    print(f"NB_STATS no longer contains PASS_YDS: {'PASS_YDS' not in NB_STATS}")

    refit_passy("PASS_YDS", "passing_yards", [
        ("PASS_YDS", 100, 400),
    ])

    refit_passy("PASS_YDS+TD", "passing_yards", [
        ("PASS_YDS+TD", 100, 400),
    ])

    print(f"\nDone!")


if __name__ == "__main__":
    main()
