#!/usr/bin/env python3
"""Fit Beta Calibration for MLB regression models and save to calibration dir.

Loads each trained LGBM regressor, replays on the test set, computes
P(stat >= line_val) via normal CDF for all relevant line values, then fits
a BetaCalibrator on the (p_raw, actual_outcome) pairs.

Saves to: models/mlb/calibration/{stat_name}_beta_cal.json

The scanner (kalshi_mlb_unified.py) already looks for these files and will
use Beta Calibration instead of the Wang Transform fallback.

Usage:
    python -m src.scripts.fit_mlb_beta_cal
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config.settings import CONFIG_DIR, PROJECT_ROOT
from src.features.mlb import MLBFeatureEngineer
from src.models.calibrator import BetaCalibrator
from src.models.distributions import p_ge_stat
import toml, lightgbm as lgb
from scipy.stats import norm

MODEL_DIR = PROJECT_ROOT / "models" / "mlb"
CALIB_DIR = MODEL_DIR / "calibration"
CALIB_DIR.mkdir(parents=True, exist_ok=True)

# Statcast rolling features (same as train_mlb_regression.py)
_STATCAST_FEATS = [
    "barrel_rate_avg5", "barrel_rate_avg15", "barrel_rate_ewm",
    "hard_hit_rate_avg5", "hard_hit_rate_avg15", "hard_hit_rate_ewm",
    "launch_speed_avg_avg5", "launch_speed_avg_avg15", "launch_speed_avg_ewm",
    "launch_angle_avg_avg5", "launch_angle_avg_avg15", "launch_angle_avg_ewm",
    "xwoba_avg5", "xwoba_avg15", "xwoba_ewm",
    "xslg_avg5", "xslg_avg15", "xslg_ewm",
    "xba_avg5", "xba_avg15", "xba_ewm",
    "sweet_spot_rate_avg5", "sweet_spot_rate_avg15", "sweet_spot_rate_ewm",
    "avg_hit_distance_avg5", "avg_hit_distance_avg15", "avg_hit_distance_ewm",
    "max_ev_avg5", "max_ev_avg15", "max_ev_ewm",
]


def load_features():
    """Load cached MLB data and build features (same as train_mlb_regression.py)."""
    cache_dir = PROJECT_ROOT / "data" / "cache" / "mlb"
    cache_files = sorted(cache_dir.glob("game_logs_*.parquet"))
    if not cache_files:
        print("No cached data. Run 'python -m src.main train mlb' first.")
        return None

    cfg = toml.load(CONFIG_DIR / "mlb.toml")
    from src.config.settings import SportConfig
    scfg = SportConfig(
        name="mlb", display_name="MLB",
        rolling_windows=cfg["features"]["rolling_windows"],
        recency_decay=0.001,
    )
    fe = MLBFeatureEngineer(scfg)
    all_games = pd.concat([pd.read_parquet(f) for f in cache_files], ignore_index=True)
    featured = fe.build_features(all_games)
    stat_cols = ["so", "er", "h", "bb", "hr", "tb", "rbi", "sb", "ip", "r",
                 "1b", "2b", "3b", "position", "player_name", "team_abbr", "gs", "bf"]
    raw_keep = [c for c in stat_cols if c in all_games.columns]
    if raw_keep:
        merge_cols = ["player_id", "game_date"]
        all_games["game_date"] = pd.to_datetime(all_games["game_date"])
        featured["game_date"] = pd.to_datetime(featured["game_date"])
        featured = featured.merge(all_games[merge_cols + raw_keep], on=merge_cols, how="left")

    # Merge Statcast features
    statcast_dir = PROJECT_ROOT / "data" / "cache" / "mlb" / "statcast"
    if statcast_dir.exists():
        sc_files = sorted(statcast_dir.glob("statcast_agg_*.parquet"))
        if sc_files:
            sc_data = pd.concat([pd.read_parquet(f) for f in sc_files], ignore_index=True)
            sc_feats = ["game_pk", "player_id"] + [c for c in _STATCAST_FEATS if c in sc_data.columns]
            sc_merge = sc_data[sc_feats].drop_duplicates(subset=["game_pk", "player_id"])
            featured = featured.merge(sc_merge, on=["game_pk", "player_id"], how="left")

    return featured


def fit_beta_cal_for_stat(featured, stat_name, raw_col, pos_filter, compute_fn=None):
    """Load the trained model, replay test set, fit BetaCal, and save.

    Uses the stored feature list from meta.json to ensure exact alignment
    with training, preventing feature drift.
    """
    mn = stat_name.lower()
    model_path = MODEL_DIR / f"lgb_{mn}.txt"
    meta_path = MODEL_DIR / f"lgb_{mn}.meta.json"
    if not model_path.exists() or not meta_path.exists():
        print(f"  {stat_name}: model not found at {model_path}, skipping")
        return

    # Load model and meta — use stored feature list
    model = lgb.Booster(model_file=str(model_path))
    with open(meta_path) as f:
        meta = json.load(f)
    residual_std = meta.get("residual_std", 1.0)
    model_features = meta.get("features", model.feature_name())
    n_train = meta.get("n_train", 0)
    n_test = meta.get("n_test", 0)
    print(f"  Loaded model: {n_train + n_test:,} samples, "
          f"{len(model_features)} features, σ={residual_std:.3f}")

    # Filter by position
    if pos_filter == "pitcher" and "position" in featured.columns:
        df = featured[featured["position"] == "P"].copy()
    elif pos_filter == "hitter" and "position" in featured.columns:
        df = featured[featured["position"] != "P"].copy()
    else:
        df = featured.copy()

    # Compute target column
    target_col = raw_col or f"{stat_name.lower()}_computed"
    if compute_fn is not None:
        df[target_col] = compute_fn(df)
    elif raw_col:
        target_col = raw_col

    if target_col not in df.columns:
        print(f"  {stat_name}: column '{target_col}' not found, skipping")
        return

    df = df.dropna(subset=[target_col]).copy()
    if len(df) < 100:
        print(f"  {stat_name}: only {len(df)} rows, skipping")
        return

    y = df[target_col].values
    available = [c for c in model_features if c in df.columns]
    X = df[available].copy()
    X = X.fillna(X.median())

    # Temporal split (80/20)
    date_col = "game_date" if "game_date" in df.columns else "date"
    dates = pd.to_datetime(df[date_col])
    sort_idx = dates.argsort()
    X = X.iloc[sort_idx]
    y = y[sort_idx]
    dates = dates.iloc[sort_idx]
    split = int(len(X) * 0.8)
    X_test = X.iloc[split:]
    y_test = y[split:]

    # Predict mu for test set
    # Build feature matrix matching model features
    test_feat = pd.DataFrame(index=range(len(X_test)))
    for c in model_features:
        if c in X_test.columns:
            test_feat[c] = X_test[c].values
        else:
            test_feat[c] = 0.0
    mu = model.predict(test_feat.fillna(0))

    # For each line value, compute raw P(>=line) via distribution-appropriate mapping
    # (NB for volume stats like SO/TB/H, Poisson for rare events like HR/SB)
    y_mean = float(np.mean(y_test))
    y_max = int(np.max(y_test))
    line_range = range(max(0, int(y_mean * 0.3)), min(y_max + 1, max(15, y_max + 1)))

    raw_probs = []
    outcomes = []
    for line_val in line_range:
        # Vectorized call: p_ge_stat accepts whole mu array
        p_raw = p_ge_stat(stat_name, mu, max(residual_std, 0.3), line_val)
        actual = (y_test >= line_val).astype(int)
        raw_probs.extend(p_raw.tolist())
        outcomes.extend(actual.tolist())

    raw_arr = np.array(raw_probs, dtype=float)
    out_arr = np.array(outcomes, dtype=int)

    # Show raw calibration
    valid = (raw_arr > 0.01) & (raw_arr < 0.99)
    if valid.sum() < 100:
        print(f"  {stat_name}: only {valid.sum()} valid predictions, skipping BetaCal fit")
        return

    raw_mean = float(np.mean(raw_arr))
    actual_mean = float(np.mean(out_arr))
    raw_bias = raw_mean - actual_mean
    print(f"  Raw: mean P={raw_mean:.3f}, actual rate={actual_mean:.3f}, bias={raw_bias:+.3f}")

    # Fit Beta Calibration
    beta_cal = BetaCalibrator()
    beta_cal.fit(raw_arr[valid], out_arr[valid])

    # Evaluate
    cal_probs = beta_cal.calibrate(raw_arr[valid])
    cal_bias = float(np.mean(cal_probs - out_arr[valid]))
    print(f"  BetaCal({stat_name}): a={beta_cal.a:.3f}, b={beta_cal.b:.3f}, c={beta_cal.c:.3f}")
    print(f"    Bias: {raw_bias:+.3f} → {cal_bias:+.3f}  (n={valid.sum()})")

    # Save
    save_path = CALIB_DIR / f"{stat_name.lower()}_beta_cal.json"
    beta_cal.save(save_path)
    print(f"    Saved to {save_path}")

    # Show per-line calibration with BetaCal (using distribution-appropriate mapping)
    print(f"    Per-line check (line: P_raw, P_cal, P_act, Δ_raw, Δ_cal):")
    for line_val in list(line_range)[::max(1, len(line_range) // 8)]:
        p_raw_line = p_ge_stat(stat_name, mu, max(residual_std, 0.3), line_val)
        p_raw_line = np.clip(p_raw_line, 0.001, 0.999)
        p_cal_line = beta_cal.calibrate(p_raw_line)
        p_act_line = (y_test >= line_val).mean()
        bias_raw_line = float(np.mean(p_raw_line) - p_act_line)
        bias_cal_line = float(np.mean(p_cal_line) - p_act_line)
        print(f"      line={line_val:2d}: P_raw={np.mean(p_raw_line):.3f} "
              f"P_cal={np.mean(p_cal_line):.3f} P_act={p_act_line:.3f} "
              f"Δ_raw={bias_raw_line:+.3f} Δ_cal={bias_cal_line:+.3f}")


def main():
    print("=" * 65)
    print("  MLB Beta Calibration Fitting")
    print("=" * 65)

    print("\nLoading features...", flush=True)
    featured = load_features()
    if featured is None:
        return
    print(f"  {len(featured)} total rows, {len(featured.columns)} columns", flush=True)

    # Stat definitions matching train_mlb_regression.py's STAT_TARGETS for scanner stats
    stat_defs = [
        ("SO", "so", "pitcher", None),
        ("HR", "hr", "hitter", None),
        ("TB", "tb", "hitter", None),
        ("H_R_RBI", None, "hitter", lambda df: df["h"] + df["r"] + df["rbi"]),
        # Also fit for the other scanner market types (for future use)
        ("ER", "er", "pitcher", None),
        ("H", "h", "pitcher", None),
        ("BB", "bb", "pitcher", None),
        ("R", "r", "hitter", None),
        ("RBI", "rbi", "hitter", None),
        ("SB", "sb", "hitter", None),
        ("IP", "ip", "pitcher", None),
    ]

    for stat_name, raw_col, pos_filter, compute_fn in stat_defs:
        print(f"\n--- {stat_name} ---", flush=True)
        fit_beta_cal_for_stat(featured, stat_name, raw_col, pos_filter, compute_fn)

    print(f"\n{'=' * 65}")
    print("  Beta Calibration files saved:")
    for f in sorted(CALIB_DIR.glob("*_beta_cal.json")):
        with open(f) as fh:
            data = json.load(fh)
        print(f"    {f.name}: a={data.get('a', 0):.3f}, b={data.get('b', 0):.3f}, c={data.get('c', 0):.3f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
