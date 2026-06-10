#!/usr/bin/env python3
"""Fit Isotonic Calibration for low-count MLB regression models.

Per NotebookLM: low-count props (mean < 1.0) have severe tail overconfidence
that Isotonic Regression handles better than parametric BetaCal.

Fits IsotonicCalibrator for: IP, R, RBI, HR, SB
Saves to: models/mlb/calibration/{stat}_isotonic_cal.json

The scanner (kalshi_mlb_unified.py) already checks these files and prefers
Isotonic over BetaCal for stats in ISOTONIC_PREFERRED.

Usage:
    python -m src.scripts.fit_mlb_isotonic_cal
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config.settings import CONFIG_DIR, PROJECT_ROOT
from src.features.mlb import MLBFeatureEngineer
from src.models.calibrator import IsotonicCalibrator
from src.models.distributions import p_ge_stat
import toml, lightgbm as lgb

MODEL_DIR = PROJECT_ROOT / "models" / "mlb"
CALIB_DIR = MODEL_DIR / "calibration"
CALIB_DIR.mkdir(parents=True, exist_ok=True)


def load_features():
    """Load cached MLB data and build features (matches fit_mlb_beta_cal.py)."""
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
    return featured


def fit_isotonic_for_stat(featured, stat_name, raw_col, pos_filter, compute_fn=None):
    """Load the trained model, replay test set, fit IsotonicCal, and save."""
    mn = stat_name.lower()
    model_path = MODEL_DIR / f"lgb_{mn}.txt"
    meta_path = MODEL_DIR / f"lgb_{mn}.meta.json"
    if not model_path.exists() or not meta_path.exists():
        print(f"  {stat_name}: model not found, skipping")
        return False

    model = lgb.Booster(model_file=str(model_path))
    with open(meta_path) as f:
        meta = json.load(f)
    residual_std = meta.get("residual_std", 1.0)
    model_features = meta.get("features", model.feature_name())

    # Filter by position
    if pos_filter == "pitcher" and "position" in featured.columns:
        df = featured[featured["position"] == "P"].copy()
    elif pos_filter == "hitter" and "position" in featured.columns:
        df = featured[featured["position"] != "P"].copy()
    else:
        df = featured.copy()

    target_col = raw_col or f"{stat_name.lower()}_computed"
    if compute_fn is not None:
        df[target_col] = compute_fn(df)
    elif raw_col:
        target_col = raw_col

    if target_col not in df.columns:
        print(f"  {stat_name}: column '{target_col}' not found")
        return False

    df = df.dropna(subset=[target_col]).copy()
    if len(df) < 100:
        print(f"  {stat_name}: only {len(df)} rows")
        return False

    y = df[target_col].values
    available = [c for c in model_features if c in df.columns]
    X = df[available].copy().fillna(0)

    # Temporal split (80/20)
    dates = pd.to_datetime(df["game_date"])
    sort_idx = dates.argsort()
    X = X.iloc[sort_idx]
    y = y[sort_idx]
    split = int(len(X) * 0.8)
    X_test = X.iloc[split:]
    y_test = y[split:]

    # Predict
    test_feat = pd.DataFrame(index=range(len(X_test)))
    for c in model_features:
        if c in X_test.columns:
            test_feat[c] = X_test[c].values
        else:
            test_feat[c] = 0.0
    mu = model.predict(test_feat.fillna(0))

    # Build (p_raw, actual) pairs across all line values
    y_mean = float(np.mean(y_test))
    y_max = int(np.max(y_test))
    line_range = range(max(0, int(y_mean * 0.3)), min(y_max + 1, max(15, y_max + 1)))

    raw_probs, outcomes = [], []
    for line_val in line_range:
        p_raw = p_ge_stat(stat_name, mu, max(residual_std, 0.3), line_val)
        actual = (y_test >= line_val).astype(int)
        raw_probs.extend(p_raw.tolist())
        outcomes.extend(actual.tolist())

    raw_arr = np.array(raw_probs, dtype=float)
    out_arr = np.array(outcomes, dtype=int)

    # Filter to non-trivial
    valid = (raw_arr > 0.005) & (raw_arr < 0.995)
    if valid.sum() < 50:
        print(f"  {stat_name}: only {valid.sum()} valid predictions, skipping")
        return False

    # Fit Isotonic
    ic = IsotonicCalibrator()
    ic.fit(raw_arr[valid], out_arr[valid])

    # Evaluate
    raw_bias = float(np.mean(raw_arr[valid] - out_arr[valid]))
    cal_probs = ic.calibrate(raw_arr[valid])
    cal_bias = float(np.mean(cal_probs - out_arr[valid]))
    print(f"  {stat_name:6s}: bias {raw_bias:+.4f} -> {cal_bias:+.4f}  "
          f"n={valid.sum():,}  xs={len(ic._xs):3d}", flush=True)

    # Save
    save_path = CALIB_DIR / f"{mn}_isotonic_cal.json"
    ic.save(save_path)
    return True


def main():
    print("=" * 65)
    print("  MLB ISOTONIC CALIBRATION FITTER")
    print("  Per NotebookLM: better than BetaCal for low-count stats")
    print("=" * 65)

    print("\nLoading features...", flush=True)
    featured = load_features()
    if featured is None:
        return
    print(f"  {len(featured):,} total rows, {len(featured.columns)} cols", flush=True)

    # Low-count stats where Isotonic > BetaCal
    stat_defs = [
        ("HR",  "hr",  "hitter",  None),
        ("IP",  "ip",  "pitcher", None),
        ("R",   "r",   "hitter",  None),
        ("RBI", "rbi", "hitter",  None),
        ("SB",  "sb",  "hitter",  None),
    ]

    n_fitted = 0
    for stat_name, raw_col, pos_filter, compute_fn in stat_defs:
        if fit_isotonic_for_stat(featured, stat_name, raw_col, pos_filter, compute_fn):
            n_fitted += 1

    print(f"\n{'=' * 65}")
    print(f"  Fitted {n_fitted}/{len(stat_defs)} Isotonic calibrators")
    print(f"  Saved to {CALIB_DIR}/")
    for f in sorted(CALIB_DIR.glob("*_isotonic_cal.json")):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name} ({size_kb:.1f} KB)")
    print("=" * 65)


if __name__ == "__main__":
    main()
