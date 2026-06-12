#!/usr/bin/env python3
"""Fit Beta Calibration for MLB player stat models.

Loads the trained XGBoost MLB models, runs a temporal train/test split,
computes raw probabilities for various lines, then fits BetaCalibrator
to correct systematic bias. Mirrors scripts/fit_nba_beta_cal.py but
adapted for MLB stat types and cache.

Usage:
    python scripts/fit_mlb_beta_cal.py
"""
import sys, json, glob, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config.settings import CONFIG_DIR, PROJECT_ROOT
from src.models.calibrator import BetaCalibrator
from src.models.distributions import p_ge_stat
import toml

MODEL_DIR = PROJECT_ROOT / "models" / "mlb"
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "mlb"

# MLB stat types to calibrate
# Combines config/mlb.toml stat_types with the model file naming convention.
# Combo stat file availability (as of 2026-06-09):
#   H+BB       NO model file  (train_mlb_regression.py doesnt train this)
#   SO+BB      NO model file  (train_mlb_regression.py doesnt train this)
#   H+R+RBI    HAS lgb_h_r_rbi.txt (training script name: H_R_RBI)
# The find_model_paths() helper checks both the new LightGBM
# (lgb_<stat>.txt + lgb_<stat>.meta.json) and legacy XGBoost
# (<STAT>.json + <STAT>.metrics.json) formats, so missing models are
# skipped cleanly via the existence check in main().
STAT_TYPES = [
    ("H", "H"),
    ("TB", "TB"),
    ("SO", "SO"),
    ("HR", "HR"),
    ("RBI", "RBI"),
    ("BB", "BB"),
    ("SB", "SB"),
    ("R", "R"),
    ("ER", "ER"),         # added 2026-06-11 — was falling through to Wang
    ("IP", "IP"),         # added 2026-06-11 — was falling through to Wang
    ("2B", "2B"),
    ("1B", "1B"),
    ("3B", "3B"),
    ("H+BB", "H+BB"),
    ("SO+BB", "SO+BB"),
    ("H+R+RBI", "H+R+RBI"),
]


def find_latest_cache() -> Path | None:
    """Find the most recent MLB game_logs parquet in the cache dir."""
    candidates = sorted(CACHE_DIR.glob("game_logs_*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def load_features():
    """Load MLB data and build features using MLBFeatureEngineer."""
    from src.features.mlb import MLBFeatureEngineer
    from src.config.settings import SportConfig

    cfg_path = CONFIG_DIR / "mlb.toml"
    if cfg_path.exists():
        cfg = toml.load(cfg_path)
    else:
        cfg = {"features": {"rolling_windows": [7, 14, 30], "recency_decay": 0.005}}

    scfg = SportConfig(
        name="mlb", display_name="MLB",
        rolling_windows=cfg["features"]["rolling_windows"],
        recency_decay=cfg["features"].get("recency_decay", 0.005),
    )
    fe = MLBFeatureEngineer(scfg)

    cache_path = find_latest_cache()
    if cache_path is None or not cache_path.exists():
        print(f"No cached MLB data at {CACHE_DIR}. Run data pipeline first.")
        return None

    all_games = pd.read_parquet(cache_path)
    print(f"Loaded {len(all_games)} raw rows from {cache_path.name}", flush=True)

    # Ensure player_name exists
    if "player_name" not in all_games.columns:
        all_games["player_name"] = all_games.get("fullName", "")

    featured = fe.build_features(all_games)
    print(f"Feature engineering: {len(featured)} rows, {len(featured.columns)} cols", flush=True)

    # Merge raw stat columns back
    raw_stat_cols = ["h", "ab", "r", "rbi", "hr", "2b", "3b", "bb", "so", "sb", "cs",
                     "tb", "ip", "er", "ha", "outs", "bb_allowed", "1b", "hbp", "sf", "gidp"]
    raw_keep = [c for c in raw_stat_cols if c in all_games.columns]
    merge_cols = ["player_id", "game_date"]
    all_games["game_date"] = pd.to_datetime(all_games["game_date"])
    featured["game_date"] = pd.to_datetime(featured["game_date"])
    featured = featured.merge(
        all_games[merge_cols + raw_keep + ["player_name"]].drop_duplicates(subset=merge_cols),
        on=merge_cols, how="left"
    )

    # Compute combo stat columns
    if all(c in featured.columns for c in ["h", "bb"]):
        featured["h+bb"] = featured["h"].fillna(0) + featured["bb"].fillna(0)
        featured["h+r+rbi"] = featured["h"].fillna(0) + featured.get("r", 0).fillna(0) + featured["rbi"].fillna(0)
        featured["so+bb"] = featured.get("so", 0).fillna(0) + featured["bb"].fillna(0)
    # Total bases (TB): 1B + 2*2B + 3*3B + 4*HR. We have 1B = h - 2b - 3b - hr.
    if all(c in featured.columns for c in ["h", "2b", "3b", "hr"]):
        featured["1b"] = featured["h"] - featured["2b"] - featured["3b"] - featured["hr"]
        featured["tb"] = (
            featured["1b"].fillna(0) * 1
            + featured["2b"].fillna(0) * 2
            + featured["3b"].fillna(0) * 3
            + featured["hr"].fillna(0) * 4
        )
    # OUTS for IP calc: IP = outs / 3 (for the test set we just use outs as the count)
    if "outs" in featured.columns:
        featured["outs"] = featured["outs"].fillna(0)
    # BB_ALLOWED (for pitchers) — alias of bb
    if "bb" in featured.columns:
        featured["bb_allowed"] = featured["bb"]

    print(f"  Merged raw stats: {len(raw_keep)} columns back, computed combos", flush=True)
    return featured


def find_model_paths(stat_name: str):
    """Find MLB model + metadata for a stat, supporting both formats.

    Returns (model_path, meta_path, format_kind) or (None, None, None) if
    no model is on disk. The current training script
    (src/scripts/train_mlb_regression.py) writes LightGBM models as
    `lgb_<stat>.txt` + `lgb_<stat>.meta.json`. An older XGBoost-based
    training system wrote `<STAT>.json` + `<STAT>.metrics.json` (uppercase
    with literal + signs for combos). We try the new format first, then
    fall back to the legacy format.

    Combo stats H+BB and SO+BB are NOT in the current training script's
    STAT_TARGETS, so they have no model file and return None here.
    """
    # New format: lgb_<lowercase_stat>.txt, with + replaced by _
    mn = stat_name.lower().replace("+", "_")
    new_model = MODEL_DIR / f"lgb_{mn}.txt"
    new_meta = MODEL_DIR / f"lgb_{mn}.meta.json"
    if new_model.exists():
        return new_model, new_meta, "lgb"

    # Legacy format: <STAT>.json (uppercase, + signs kept)
    legacy_model = MODEL_DIR / f"{stat_name}.json"
    legacy_meta = MODEL_DIR / f"{stat_name}.metrics.json"
    if legacy_model.exists():
        return legacy_model, legacy_meta, "xgb"

    return None, None, None


def fit_calibration(stat_name: str, model_display: str, featured: pd.DataFrame):
    """Load model, run temporal test split, fit BetaCal, save results."""
    import xgboost as xgb
    import lightgbm as lgb

    model_path, meta_path, fmt = find_model_paths(stat_name)
    if model_path is None:
        print(f"  {stat_name}: no model file (new lgb_*.txt or legacy .json)")
        return

    # Load model and extract feature names
    if fmt == "lgb":
        booster = lgb.Booster(model_file=str(model_path))
        feature_names = booster.feature_name()
        model = booster  # use booster uniformly for prediction below
    else:  # xgb
        model = xgb.XGBRegressor()
        model.load_model(str(model_path))
        try:
            with open(model_path) as f:
                mdata = json.load(f)
            feature_names = mdata.get("learner", {}).get("feature_names", [])
        except Exception:
            print(f"  {stat_name}: could not extract feature names from {model_path}")
            return

    # Load residual std from meta
    std = 1.0
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        std = meta.get("residual_std", meta.get("mae", 1.0))

    # Canonical name for cal/diag file outputs (always lowercase + underscores)
    mn = stat_name.lower().replace("+", "_")

    # Get raw stat column name. Combo stats live in the dataframe as
    # 'h+r+rbi', 'h+bb', 'so+bb' (with plus signs, see the feature
    # engineering above), so we keep the `+` for column lookup while
    # using `mn` (with underscores) for the file path. This was a bug
    # prior to 2026-06-11 — H+R+RBI was silently skipped because
    # `raw_col = mn = "h_r_rbi"` didn't match the actual column.
    raw_col = stat_name.lower()
    if raw_col not in featured.columns:
        print(f"  {stat_name}: column '{raw_col}' not found in features")
        return

    # Filter rows with non-null target
    df = featured.dropna(subset=[raw_col]).copy()
    if len(df) < 100:
        print(f"  {stat_name}: only {len(df)} rows")
        return

    # Build feature matrix
    available = [c for c in feature_names if c in df.columns]
    if not available:
        print(f"  {stat_name}: no matching features in data")
        return

    X = df[available].fillna(0).copy()
    y = df[raw_col].values

    # Temporal split (80/20 by date)
    if "game_date" in df.columns:
        dates = pd.to_datetime(df["game_date"])
        sort_idx = dates.argsort()
        X = X.iloc[sort_idx]
        y = y[sort_idx]
        dates = dates.iloc[sort_idx]
        split = int(len(X) * 0.8)
        X_test = X.iloc[split:]
        y_test = y[split:]
    else:
        split = int(len(X) * 0.8)
        X_test = X.iloc[split:]
        y_test = y[split:]

    print(f"  {stat_name:8s}: {len(X_test)} test rows, σ={std:.3f}", flush=True)

    # Predict on test set
    preds = model.predict(X_test.values)

    # Compute raw probabilities for a range of relevant lines
    y_mean = float(y_test.mean())
    if pd.isna(y_mean) or y_mean <= 0:
        print(f"  {stat_name}: y_mean={y_mean}, skipping")
        return

    # For count stats: line range from 0 to ceil(y_mean * 3.5).
    # The legacy y_mean * 2.5 under-shoots Kalshi market lines for
    # combo stats like H+R+RBI where the typical market line is 3-7
    # but y_mean is only ~2.5. Bumped 2026-06-11.
    max_line = max(1, int(np.ceil(y_mean * 3.5)))
    min_line = 0
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

    # Filter out trivial predictions
    valid = (raw_arr > 0.01) & (raw_arr < 0.99)
    if valid.sum() < 100:
        print(f"  {stat_name}: only {valid.sum()} non-trivial predictions, skipping calibration")
        return

    # Fit BetaCal
    beta_cal = BetaCalibrator()
    beta_cal.fit(raw_arr[valid], out_arr[valid])

    # Evaluate
    before_bias = float(np.mean(raw_arr[valid] - out_arr[valid]))
    cal_probs = beta_cal.calibrate(raw_arr[valid])
    after_bias = float(np.mean(cal_probs - out_arr[valid]))

    # Calibration by bins
    bins = np.linspace(0, 1, 11)
    bin_map = {i: [] for i in range(10)}
    valid_arr = raw_arr[valid]
    for i, p in enumerate(valid_arr):
        bin_idx = min(9, max(0, int(p * 10))) if p < 0.99 else 9
        bin_map[bin_idx].append(i)

    raw_cal = []
    cal_cal = []
    for bin_idx in range(10):
        indices = bin_map[bin_idx]
        if len(indices) < 5:
            continue
        raw_bin = raw_arr[valid][indices]
        cal_bin = cal_probs[indices]
        out_bin = out_arr[valid][indices]
        raw_cal.append(abs(float(np.mean(raw_bin) - np.mean(out_bin))))
        cal_cal.append(abs(float(np.mean(cal_bin) - np.mean(out_bin))))

    print(f"    BetaCal: bias {before_bias:+.4f} → {after_bias:+.4f} (n={valid.sum()})", flush=True)
    if raw_cal and cal_cal:
        avg_raw = float(np.mean(raw_cal))
        avg_cal = float(np.mean(cal_cal))
        print(f"    Cal error: {avg_raw:.4f} → {avg_cal:.4f} "
              f"({'✅ improved' if avg_cal < avg_raw else '❌ worse'})")

    # Save
    cal_path = MODEL_DIR / f"{mn}_beta_cal.json"
    beta_cal.save(cal_path)
    print(f"    Saved {cal_path}")

    diag = {
        "stat": stat_name,
        "n_test": int(len(X_test)),
        "n_calibration_samples": int(valid.sum()),
        "residual_std": std,
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
    print("  MLB BETA CALIBRATION FITTER")
    print("=" * 65)

    featured = load_features()
    if featured is None:
        return

    print(f"\nLoaded features: {len(featured)} rows", flush=True)

    n_skipped = 0
    n_done = 0
    for stat_name, display in STAT_TYPES:
        # Use find_model_paths() so we pick up both new lgb_*.txt and legacy .json
        # files. STAT_TYPES lists all 3 combo stats; H+BB and SO+BB are skipped
        # here because the current training script (train_mlb_regression.py)
        # does not train them. Once models for those stats are added to
        # STAT_TARGETS, this script will pick them up automatically.
        model_path, _, _ = find_model_paths(stat_name)
        if model_path is None:
            print(f"\nCalibrating {stat_name} ({display})... SKIPPED (no model file)", flush=True)
            n_skipped += 1
            continue
        print(f"\nCalibrating {stat_name} ({display})...", flush=True)
        fit_calibration(stat_name, display, featured)
        n_done += 1

    print(f"\n{'=' * 65}")
    print(f"  Done. {n_done} calibrated, {n_skipped} skipped (no model).")
    print(f"  Calibration files saved to models/mlb/")

    # Summary
    print(f"\n  {'Stat':10s} {'a':>8s} {'b':>8s} {'c':>8s} {'N_cal':>6s} {'Bias':>8s}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*8}")
    for stat_name, _ in STAT_TYPES:
        mn = stat_name.lower().replace("+", "_")
        diag_path = MODEL_DIR / f"{mn}_calibration_diag.json"
        if diag_path.exists():
            with open(diag_path) as f:
                d = json.load(f)
            print(f"  {stat_name:10s} {d['a']:>8.3f} {d['b']:>8.3f} {d['c']:>8.3f} "
                  f"{d['n_calibration_samples']:6d} {d['after_bias']:>+7.3f}")


if __name__ == "__main__":
    main()
