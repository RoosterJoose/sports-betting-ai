#!/usr/bin/env python3
"""Fit Beta Calibration for WNBA player stat models.

Loads the trained XGBoost WNBA models, runs a temporal train/test split,
computes raw NB/Poisson probabilities for various lines, then fits
BetaCalibrator to correct systematic bias.

Usage:
    python scripts/fit_wnba_beta_cal.py
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config.settings import CONFIG_DIR, PROJECT_ROOT
from src.models.calibrator import BetaCalibrator
from src.models.distributions import p_ge_stat
import toml

MODEL_DIR = PROJECT_ROOT / "models" / "wnba"
CACHE_PATH = PROJECT_ROOT / "data" / "wnba_cache" / "wnba_games.parquet"

# WNBA stat types to calibrate (models with R2 > 0.20)
STAT_TYPES = [
    ("PTS", "PTS"),
    ("REB", "REB"),
    ("AST", "AST"),
    ("BLK", "BLK"),
    ("FG3M", "3PT"),
    ("PRA", "PRA"),
    ("PR", "PR"),
    ("PA", "PA"),
    ("FPTS", "FPTS"),
]


def load_features():
    """Load WNBA player-level data and build features."""
    from src.features.wnba import WNBAFeatureEngineer
    from src.config.settings import SportConfig

    cfg_path = CONFIG_DIR / "wnba.toml"
    if cfg_path.exists():
        cfg = toml.load(cfg_path)
    else:
        cfg = {"features": {"rolling_windows": [3, 5, 10], "recency_decay": 0.001}}

    scfg = SportConfig(
        name="wnba", display_name="WNBA",
        rolling_windows=cfg["features"]["rolling_windows"],
        recency_decay=cfg["features"].get("recency_decay", 0.001),
    )
    fe = WNBAFeatureEngineer(scfg)

    if not CACHE_PATH.exists():
        print("No cached WNBA data. Run data pipeline first.")
        return None

    all_games = pd.read_parquet(CACHE_PATH)
    print(f"Loaded {len(all_games)} raw rows", flush=True)

    featured = fe.build_features(all_games)
    print(f"Feature engineering: {len(featured)} rows, {len(featured.columns)} cols", flush=True)

    # Merge raw stat columns back (feature engineer strips them)
    raw_stat_cols = ["pts", "reb", "ast", "stl", "blk", "tov", "fg3m", "fg3a", "fgm", "fga", "ftm", "fta", "min"]
    raw_keep = [c for c in raw_stat_cols if c in all_games.columns]
    merge_cols = ["player_id", "game_date"]
    all_games["game_date"] = pd.to_datetime(all_games["game_date"])
    featured["game_date"] = pd.to_datetime(featured["game_date"])
    featured = featured.merge(
        all_games[merge_cols + raw_keep].drop_duplicates(subset=merge_cols),
        on=merge_cols, how="left"
    )

    # Compute combo stat columns
    if all(c in featured.columns for c in ["pts", "reb", "ast"]):
        featured["pra"] = featured["pts"].fillna(0) + featured["reb"].fillna(0) + featured["ast"].fillna(0)
        featured["pa"] = featured["pts"].fillna(0) + featured["ast"].fillna(0)
        featured["pr"] = featured["pts"].fillna(0) + featured["reb"].fillna(0)

    # Compute FPTS (fantasy points)
    if all(c in featured.columns for c in ["pts", "reb", "ast", "stl", "blk", "tov"]):
        featured["fpts"] = (featured["pts"].fillna(0) * 1.0 +
                            featured["reb"].fillna(0) * 1.2 +
                            featured["ast"].fillna(0) * 1.5 +
                            featured["stl"].fillna(0) * 3.0 +
                            featured["blk"].fillna(0) * 3.0 -
                            featured["tov"].fillna(0) * 1.0)

    print(f"  Merged raw stats: {len(raw_keep)} columns back", flush=True)
    return featured


def fit_calibration(stat_name: str, model_display: str, featured: pd.DataFrame):
    """Load model, run temporal test split, fit BetaCal, save results."""
    import xgboost as xgb

    mn = stat_name.lower()
    model_path = MODEL_DIR / f"{mn}.json"
    meta_path = MODEL_DIR / f"{mn}.metrics.json"

    if not model_path.exists():
        print(f"  {stat_name}: no model at {model_path}")
        return

    model = xgb.XGBRegressor()
    model.load_model(str(model_path))

    try:
        with open(model_path) as f:
            mdata = json.load(f)
        feature_names = mdata.get("learner", {}).get("feature_names", [])
    except Exception:
        feature_names = []
        print(f"  {stat_name}: could not extract feature names")
        return

    std = 1.0
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        std = meta.get("residual_std", meta.get("mae", 1.0))

    raw_col = stat_name.lower()
    if raw_col not in featured.columns:
        print(f"  {stat_name}: column '{raw_col}' not found")
        return

    df = featured.dropna(subset=[raw_col]).copy()
    if len(df) < 100:
        print(f"  {stat_name}: only {len(df)} rows")
        return

    available = [c for c in feature_names if c in df.columns]
    if not available:
        print(f"  {stat_name}: no matching features")
        return

    X = df[available].fillna(0).copy()
    y = df[raw_col].values

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

    print(f"  {stat_name:4s}: {len(X_test)} test rows, sigma={std:.3f}", flush=True)

    preds = model.predict(X_test.values)

    y_mean = float(y_test.mean())
    if pd.isna(y_mean) or y_mean <= 0:
        print(f"  {stat_name}: y_mean={y_mean}, skipping")
        return

    max_line = max(1, int(y_mean * 2.5))
    min_line = max(0, int(y_mean * 0.2))
    if max_line <= min_line:
        max_line = min_line + 2

    raw_probs_all = []
    outcomes_all = []

    for line_val in range(min_line, max_line + 1):
        p_model = np.array([
            p_ge_stat(stat_name, max(0, preds[i]), std, line_val)
            for i in range(len(preds))
        ])
        raw_probs_all.extend(p_model.tolist())
        outcomes_all.extend((y_test >= line_val).astype(int).tolist())

    raw_arr = np.array(raw_probs_all)
    out_arr = np.array(outcomes_all, dtype=int)

    valid = (raw_arr > 0.01) & (raw_arr < 0.99)
    if valid.sum() < 100:
        print(f"  {stat_name}: only {valid.sum()} non-trivial predictions, skipping")
        return

    beta_cal = BetaCalibrator()
    beta_cal.fit(raw_arr[valid], out_arr[valid])

    before_bias = float(np.mean(raw_arr[valid] - out_arr[valid]))
    cal_probs = beta_cal.calibrate(raw_arr[valid])
    after_bias = float(np.mean(cal_probs - out_arr[valid]))

    print(f"    BetaCal: bias {before_bias:+.4f} -> {after_bias:+.4f} (n={valid.sum()})")

    cal_path = MODEL_DIR / f"{mn}_beta_cal.json"
    beta_cal.save(cal_path)
    print(f"    Saved {cal_path}")

    diag = {
        "stat": stat_name,
        "n_test": int(len(X_test)),
        "n_calibration_samples": int(valid.sum()),
        "sigma": std,
        "before_bias": round(before_bias, 4),
        "after_bias": round(after_bias, 4),
        "a": beta_cal.a,
        "b": beta_cal.b,
        "c": beta_cal.c,
    }
    diag_path = MODEL_DIR / f"{mn}_calibration_diag.json"
    with open(diag_path, "w") as f:
        json.dump(diag, f, indent=2)


def main():
    print("=" * 65)
    print("  WNBA BETA CALIBRATION FITTER")
    print("=" * 65)

    featured = load_features()
    if featured is None:
        return

    print(f"\nLoaded features: {len(featured)} rows", flush=True)

    for stat_name, display in STAT_TYPES:
        print(f"\nCalibrating {stat_name} ({display})...", flush=True)
        fit_calibration(stat_name, display, featured)

    print(f"\n{'=' * 65}")
    print("  Done. Calibration files saved to models/wnba/")
    for f in sorted(MODEL_DIR.glob("*beta_cal*")):
        print(f"    {f.name}")


if __name__ == "__main__":
    main()
