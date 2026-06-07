#!/usr/bin/env python3
"""Train NFL player stat regressors with LGBM.

Trains LGBMRegressor models for passing, rushing, receiving, and TD stats
using nfl_data_py weekly data with enhanced features (DvP, opponent quality,
availability, target share, rolling stats).

Usage:
    python -m src.scripts.train_nfl_regression
"""
import sys, json, warnings, os
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config.settings import CONFIG_DIR, PROJECT_ROOT
from src.features.nfl import NFLFeatureEngineer
import toml, lightgbm as lgb

MODEL_DIR = PROJECT_ROOT / "models" / "nfl"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

STAT_TARGETS = [
    ("PASS_YDS",    "passing_yards",    "all", None),
    ("PASS_TD",     "passing_tds",      "all", None),
    ("PASS_ATT",    "pass_attempts",    "all", None),
    ("INT",         "interceptions",    "all", None),
    ("PASS_YDS+TD", None,               "all", lambda df: df["passing_yards"] + df["passing_tds"] * 10),
    ("RUSH_YDS",    "rushing_yards",    "all", None),
    ("REC",         "receptions",       "all", None),
    ("REC_YDS",     "receiving_yards",  "all", None),
    ("RUSH+REC_YDS", None,              "all", lambda df: df["rushing_yards"] + df["receiving_yards"]),
    ("TD",          "touchdowns",       "all", None),
]


def load_features():
    cache_path = PROJECT_ROOT / "data" / "nfl_cache" / "weekly.parquet"
    if not cache_path.exists():
        print("No cached NFL data. Run 'python -m src.data.nfl' to fetch first.")
        return None

    cfg_path = CONFIG_DIR / "nfl.toml"
    if cfg_path.exists():
        cfg = toml.load(cfg_path)
    else:
        cfg = {"features": {"rolling_windows": [3, 5, 7], "recency_decay": 0.001}}

    from src.config.settings import SportConfig
    scfg = SportConfig(
        name="nfl", display_name="NFL",
        rolling_windows=cfg["features"]["rolling_windows"],
        recency_decay=cfg["features"].get("recency_decay", 0.001),
    )
    fe = NFLFeatureEngineer(scfg)
    all_games = pd.read_parquet(cache_path)
    print(f"Loaded {len(all_games)} rows from {cache_path}", flush=True)

    if "player_name" not in all_games.columns and "player_display_name" in all_games.columns:
        all_games["player_name"] = all_games["player_display_name"]

    featured = fe.build_features(all_games)
    print(f"Feature engineering: {len(featured)} rows, {len(featured.columns)} cols", flush=True)

    stat_cols = ["passing_yards", "passing_tds", "passing_air_yards", "interceptions",
                 "rushing_yards", "rushing_tds", "carries",
                 "receiving_yards", "receiving_tds", "receptions", "targets",
                 "touchdowns", "fantasy_points", "pass_attempts", "rush_attempts",
                 "completions", "games_started", "availability",
                 "position", "player_name", "team_abbr", "recent_team",
                 "opponent_team", "headshot_url"]
    raw_keep = [c for c in stat_cols if c in all_games.columns]
    if raw_keep:
        merge_cols = ["player_id", "game_date"]
        all_games["game_date"] = pd.to_datetime(all_games["game_date"])
        featured["game_date"] = pd.to_datetime(featured["game_date"])
        if all_games["player_id"].dtype != featured["player_id"].dtype:
            all_games["player_id"] = all_games["player_id"].astype(str)
            featured["player_id"] = featured["player_id"].astype(str)
        featured = featured.merge(all_games[merge_cols + raw_keep], on=merge_cols, how="left")

    return featured


def train_regressor(featured, stat_name, raw_col, pos_filter="all", compute_fn=None):
    import re
    lagged_pattern = re.compile(r".*_avg_\d+$")
    extra_feats = {"days_rest", "b2b", "four_in_six",
                   "was_available", "availability_avg_3", "availability_avg_5",
                   "target_share", "wopr", "racr", "air_yards_share",
                   "dvp_fp_avg_3", "dvp_fp_avg_5", "dvp_rec_yds_avg_3", "dvp_rush_yds_avg_3",
                   "team_pass_yds_avg_3", "team_pass_yds_avg_5", "team_rush_yds_avg_3",
                   "def_pass_yds_allowed_avg_3", "def_pass_yds_allowed_avg_5",
                   "def_rush_yds_allowed_avg_3", "def_rec_yds_allowed_avg_3",
                   "def_fp_allowed_avg_3", "def_fp_allowed_avg_5"}
    feature_cols = [c for c in featured.columns
                    if (lagged_pattern.match(c)
                        or c.endswith("_ewm")
                        or c.endswith(("_streak", "_consistency"))
                        or c in extra_feats)
                    and featured[c].dtype in ("float64", "int64", "float32", "int32")]

    df = featured.copy()

    target_col = raw_col or f"{stat_name.lower()}_computed"
    if compute_fn is not None:
        try:
            df[target_col] = compute_fn(df)
        except Exception as e:
            print(f"  {stat_name}: compute_fn failed: {e}")
            return None
    elif raw_col:
        target_col = raw_col

    if target_col not in df.columns:
        print(f"  {stat_name}: column '{target_col}' not found")
        return None

    df = df.dropna(subset=[target_col]).copy()
    if len(df) < 100:
        print(f"  {stat_name}: only {len(df)} rows")
        return None

    y = df[target_col].values
    available = [c for c in feature_cols if c in df.columns]
    print(f"  Features: {len(available)}")
    X = df[available].copy()
    X = X.fillna(X.median())

    date_col = "game_date" if "game_date" in df.columns else None
    if date_col:
        dates = pd.to_datetime(df[date_col])
        sort_idx = dates.argsort()
        X = X.iloc[sort_idx]; y = y[sort_idx]; dates = dates.iloc[sort_idx]
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y[:split], y[split:]
    else:
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y[:split], y[split:]

    model = lgb.LGBMRegressor(
        n_estimators=1000, num_leaves=31, learning_rate=0.02,
        subsample=0.8, feature_fraction=0.7,
        reg_alpha=0.5, reg_lambda=1.0,
        min_child_samples=20, random_state=42, verbosity=-1,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              eval_metric='l2',
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

    preds = model.predict(X_test)
    residuals = y_test - preds
    residual_std = float(np.std(residuals))
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    r2 = float(1 - np.sum(residuals ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

    print(f"  {stat_name:12s}: MAE={mae:.3f}, RMSE={rmse:.3f}, R\u00b2={r2:.3f}, \u03c3_res={residual_std:.3f}")
    print(f"    Best iteration: {model.best_iteration_}")

    # Calibration check
    from scipy.stats import norm
    y_mean = y_test.mean()
    cal_bins = []
    max_line = max(1, int(y_mean * 2.5))
    min_line = max(0, int(y_mean * 0.2))
    for line_val in range(min_line, max_line + 1):
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
        if line_val in (min_line, int(y_mean), max_line) or abs(bias) > 0.05:
            print(f"      line={line_val:2d}: P_model={p_model_mean:.1%}, P_actual={p_actual:.1%}, bias={bias:+.1%}")

    # Feature importance
    imp = pd.DataFrame({"feature": available, "importance": model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False)
    top10 = imp.head(10)["feature"].tolist()
    print(f"    Top features: {top10[:5]}")

    # Save model
    model.booster_.save_model(str(MODEL_DIR / f"lgb_{stat_name.lower()}.txt"))

    # Save importance
    imp.to_csv(MODEL_DIR / f"lgb_{stat_name.lower()}_importance.csv", index=False)

    # Calibration saved inline in meta
    meta = {
        "stat": stat_name,
        "type": "regressor",
        "framework": "lightgbm",
        "residual_std": residual_std,
        "mae": mae, "rmse": rmse, "r2": r2,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "best_iteration": int(model.best_iteration_ or 0),
        "n_features": len(available),
        "features": available,
        "top_features": top10,
    }
    with open(MODEL_DIR / f"lgb_{stat_name.lower()}.meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    with open(MODEL_DIR / f"lgb_{stat_name.lower()}.std.json", "w") as f:
        json.dump({"residual_std": residual_std}, f, indent=2)

    return model


def main():
    print("Loading NFL features...", flush=True)
    featured = load_features()
    if featured is None:
        return
    print(f"  {len(featured)} total rows", flush=True)
    print(f"  Seasons: {featured['season'].min()}-{featured['season'].max()}", flush=True)

    for stat_name, raw_col, pos_filter, compute_fn in STAT_TARGETS:
        print(f"\nTraining {stat_name}...", flush=True)
        train_regressor(featured, stat_name, raw_col, pos_filter, compute_fn)

    print(f"\nDone. Models saved to {MODEL_DIR}/")
    for f in sorted(MODEL_DIR.glob("lgb_*")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
