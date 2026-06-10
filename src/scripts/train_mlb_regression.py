#!/usr/bin/env python3
"""Train MLB player stat regressors with enhanced features.

Trains LGBMRegressor models for SO, ER, H, BB, HR, TB, RBI, R, H_R_RBI etc.

Enhancements over v1:
- Park factors for HR and TB
- Cross-stat features (ISO, contact_rate, bb_rate, hr_per_h, xbh_per_h)
- Pitcher rate features (k_rate, babip)
- Better hyperparameters with L1 regularization
- Empirical calibration saved post-training
- Feature importance tracking

Usage:
    python -m src.scripts.train_mlb_regression
"""
import sys, json, warnings, os
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config.settings import CONFIG_DIR, PROJECT_ROOT
from src.features.mlb import MLBFeatureEngineer
import toml, lightgbm as lgb

# Statcast data directory
STATCAST_DIR = PROJECT_ROOT / "data" / "cache" / "mlb" / "statcast"

MODEL_DIR = PROJECT_ROOT / "models" / "mlb"
CALIB_DIR = MODEL_DIR / "calibration"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CALIB_DIR.mkdir(parents=True, exist_ok=True)

# All stat types: (model_name, raw_col, position_filter, compute_fn)
STAT_TARGETS = [
    # Position-specific regressors (used by Kalshi)
    ("SO",  "so",   "pitcher", None),
    ("ER",  "er",   "pitcher", None),
    ("H",   "h",    "pitcher", None),
    ("BB",  "bb",   "pitcher", None),
    ("HR",  "hr",   "hitter",  None),
    ("TB",  "tb",   "hitter",  None),
    ("RBI", "rbi",  "hitter",  None),
    ("SB",  "sb",   "hitter",  None),
    ("IP",  "ip",   "pitcher", None),
    ("R",   "r",    "hitter",  None),
    ("H_R_RBI", None, "hitter", lambda df: df["h"] + df["r"] + df["rbi"]),
    # All-position regressors (used by PrizePicks)
    ("ALL_SO",  "so",   "all", None),
    ("ALL_ER",  "er",   "all", None),
    ("ALL_H",   "h",    "all", None),
    ("ALL_BB",  "bb",   "all", None),
    ("ALL_HR",  "hr",   "all", None),
    ("ALL_TB",  "tb",   "all", None),
    ("ALL_RBI", "rbi",  "all", None),
    ("ALL_SB",  "sb",   "all", None),
    ("ALL_IP",  "ip",   "all", None),
    ("ALL_R",   "r",    "all", None),
    ("ALL_H_R_RBI", None, "all", lambda df: df["h"] + df["r"] + df["rbi"]),
    # Newly added (used by PrizePicks Singles/Doubles/Triples/Fantasy Score)
    # Hitter Fantasy Score (PrizePicks standard):
    # 1*1B + 2*2B + 3*3B + 4*HR + 1*RBI + 1*R + 1*BB + 1*HBP + 2*SB - 1*CS - 0.5*SO
    # Pitcher Fantasy Score (PrizePicks standard):
    # 3*IP + 1*SO - 1*H - 1*BB - 2*ER + 2*W - 2*L
    ("ALL_1B",     "1b", "hitter",  None),
    ("ALL_2B",     "2b", "hitter",  None),
    ("ALL_3B",     "3b", "hitter",  None),
    ("ALL_H_FPTS", None, "hitter",
        lambda df: (df["1b"] + 2*df["2b"] + 3*df["3b"] + 4*df["hr"]
                   + df["rbi"] + df["r"] + df["bb"] + df["hbp"]
                   + 2*df["sb"] - df["cs"] - 0.5*df["so"])),
    ("ALL_P_FPTS", None, "pitcher",
        lambda df: (3*df["ip"] + df["so"] - df["h"] - df["bb"]
                    - 2*df["er"] + 2*df["w"] - 2*df["l"])),
]

# Statcast features that improve hitter models
STATCAST_FEATURES = [
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
    """Load cached MLB data and build features."""
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
    # Merge raw stat columns back for use as targets
    stat_cols = ["so", "er", "h", "bb", "hr", "tb", "rbi", "sb", "ip", "r", "1b", "2b", "3b",
                 "hbp", "cs", "w", "l", "sv", "position", "player_name", "team_abbr", "gs", "bf"]
    raw_keep = [c for c in stat_cols if c in all_games.columns]
    if not raw_keep:
        return featured
    merge_cols = ["player_id", "game_date"] if "game_date" in all_games.columns else ["player_id"]
    all_games[merge_cols[1]] = pd.to_datetime(all_games[merge_cols[1]])
    featured[merge_cols[1]] = pd.to_datetime(featured[merge_cols[1]])
    featured = featured.merge(all_games[merge_cols + raw_keep], on=merge_cols, how="left")

    # Merge Statcast features for hitters
    if STATCAST_DIR.exists():
        sc_files = sorted(STATCAST_DIR.glob("statcast_agg_*.parquet"))
        if sc_files:
            sc_data = pd.concat([pd.read_parquet(f) for f in sc_files], ignore_index=True)
            # Select only rolling features (not raw per-game stats which would leak)
            sc_feats = ["game_pk", "player_id"] + [c for c in STATCAST_FEATURES if c in sc_data.columns]
            # Drop duplicates on merge keys
            sc_merge = sc_data[sc_feats].drop_duplicates(subset=["game_pk", "player_id"])
            featured = featured.merge(sc_merge, on=["game_pk", "player_id"], how="left")
            n_matched = featured[["barrel_rate_avg5"]].notna().sum().iloc[0] if "barrel_rate_avg5" in featured.columns else 0
            print(f"  Merged Statcast features: {n_matched}/{len(featured)} rows matched", flush=True)

    return featured


def train_regressor(featured, stat_name, raw_col, pos_filter="pitcher", compute_fn=None):
    """Train LGBMRegressor predicting raw stat value from lagged/engineered features."""
    import re
    lagged_pattern = re.compile(r'.*_avg_\d+$')
    extra_feats = {"days_rest", "b2b", "four_in_six", "park_factor_k", "park_factor_hr", "park_factor_tb",
                    "opp_k_pct", "player_is_lefty", "opp_catcher_framing"}
    # Add Statcast features if available
    if "barrel_rate_avg5" in featured.columns:
        for sc_f in STATCAST_FEATURES:
            if sc_f in featured.columns:
                extra_feats.add(sc_f)
    feature_cols = [c for c in featured.columns
                    if (lagged_pattern.match(c)
                        or c.endswith("_ewm")
                        or c.endswith(("_streak", "_consistency"))
                        or c in extra_feats)
                    and featured[c].dtype in ("float64", "int64", "float32", "int32")]

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
        if compute_fn.__code__.co_varnames and "df" in compute_fn.__code__.co_varnames:
            df[target_col] = compute_fn(df)
        else:
            try:
                df[target_col] = df.apply(compute_fn, axis=1)
            except Exception:
                df[target_col] = compute_fn(df)
    elif raw_col:
        target_col = raw_col

    if target_col not in df.columns:
        print(f"  {stat_name}: column '{target_col}' not found, skipping")
        return None

    df = df.dropna(subset=[target_col]).copy()

    if len(df) < 100:
        print(f"  {stat_name}: only {len(df)} rows, skipping")
        return None

    y = df[target_col].values

    # Per NotebookLM: low-count props (mean<1) get severe tail overconfidence.
    # Log-transform target during training to stabilize variance, then
    # back-transform via expm1 at inference (handled in the scanner/predictor).
    LOG_TRANSFORM_STATS = {"HR", "SB", "STL", "BLK", "TOV"}
    if stat_name.upper() in LOG_TRANSFORM_STATS:
        y_raw = y.copy()
        y = np.log1p(np.maximum(y, 0))
        print(f"  Log-transform applied (y+1 -> log) for {stat_name}", flush=True)

    available = [c for c in feature_cols if c in df.columns]
    print(f"  Features: {len(available)}")
    X = df[available].copy()
    X = X.fillna(X.median())

    # Temporal split
    date_col = "game_date" if "game_date" in df.columns else "date"
    dates = pd.to_datetime(df[date_col])
    sort_idx = dates.argsort()
    X = X.iloc[sort_idx]
    y = y[sort_idx]
    dates = dates.iloc[sort_idx]

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]

    # Enhanced hyperparameters: more trees, lower LR, L1/L2 regularization
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        num_leaves=31,
        learning_rate=0.02,
        subsample=0.8,
        feature_fraction=0.7,
        reg_alpha=0.5,
        reg_lambda=1.0,
        min_child_samples=20,
        random_state=42,
        verbosity=-1,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              eval_metric='l2',
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

    # Evaluate
    preds = model.predict(X_test)
    residuals = y_test - preds
    residual_std = float(np.std(residuals))
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    r2 = float(1 - np.sum(residuals ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

    print(f"  {stat_name:3s}: MAE={mae:.3f}, RMSE={rmse:.3f}, R\u00b2={r2:.3f}, σ_res={residual_std:.3f}")
    print(f"    Best iteration: {model.best_iteration_}")

    # Calibration check
    print(f"    Calibration check (on {len(y_test)} test rows):")
    from scipy.stats import norm
    y_mean = y_test.mean()
    cal_bins = []
    for line_shift in range(-3, 7):
        line_val = int(np.round(y_mean + line_shift))
        line_val = max(0, min(15, line_val))
        p_model = 1.0 - norm.cdf((line_val - 0.5 - preds) / residual_std)
        p_model_mean = float(np.mean(p_model))
        p_actual = float((y_test >= line_val).mean())
        bias = p_model_mean - p_actual
        cal_bins.append({
            "line": line_val,
            "p_model": round(p_model_mean, 4),
            "p_actual": round(p_actual, 4),
            "bias": round(bias, 4),
            "n": int(len(y_test)),
        })
        print(f"      line=μ{line_shift:+d}={line_val:d}: P_model={p_model_mean:.1%}, P_actual={p_actual:.1%}, bias={bias:+.1%}")

    # Feature importance
    imp = pd.DataFrame({"feature": available, "importance": model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False)
    top5 = imp.head(5)["feature"].tolist()
    print(f"    Top features: {top5}")

    # Save model
    model.booster_.save_model(str(MODEL_DIR / f"lgb_{stat_name.lower()}.txt"))

    # Save calibration bins (EmpiricalCalibrator format)
    # NOTE: Single-bin-per-line calibrations (p_pred_min=0, p_pred_max=1)
    # destroy model signal by mapping all predictions to the prior.
    # The proper multi-bin calibration is built separately by
    # build_calibration.py.  Only save the legacy format for reference.
    cal_path_legacy = CALIB_DIR / f"lgb_{stat_name.lower()}.calibration.json"
    with open(cal_path_legacy, "w") as f:
        json.dump(cal_bins, f, indent=2)

    # Save metadata
    meta = {
        "stat": stat_name,
        "type": "regressor",
        "residual_std": residual_std,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "best_iteration": int(model.best_iteration_ or 0),
        "n_features": len(available),
        "features": available,
        "top_features": top5,
    }
    with open(MODEL_DIR / f"lgb_{stat_name.lower()}.meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return model


def main():
    print("Loading MLB features...", flush=True)
    featured = load_features()
    if featured is None:
        return
    print(f"  {len(featured)} total rows", flush=True)

    for stat_name, raw_col, pos_filter, compute_fn in STAT_TARGETS:
        print(f"\nTraining {stat_name}...", flush=True)
        train_regressor(featured, stat_name, raw_col, pos_filter, compute_fn)

    print(f"\nDone. Models saved to {MODEL_DIR}/")
    for f in sorted(MODEL_DIR.glob("lgb_*")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
