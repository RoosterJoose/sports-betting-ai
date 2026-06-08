#!/usr/bin/env python3
"""Retrain PASS_YDS (and PASS_YDS+TD) on QB-only data with better hyperparams.

Original model trained on ALL players, causing -13 yard mu bias for QBs
(mu=180.3 vs y=193.4) because non-QB rows with 0 passing yards drag the
predictions down.

Changes from original:
  - pos_filter='QB' instead of 'all'
  - Learning rate: 0.01 (was 0.02)
  - n_estimators: 1500 (was 1000)
  - Drop 'days_rest' from features (was #1 by importance — suspicious)
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config.settings import CONFIG_DIR, PROJECT_ROOT
from src.features.nfl import NFLFeatureEngineer
from src.models.calibrator import BetaCalibrator
from src.models.distributions import p_ge_stat
import toml, lightgbm as lgb

MODEL_DIR = PROJECT_ROOT / "models" / "nfl"


def load_features():
    cache_path = PROJECT_ROOT / "data" / "nfl_cache" / "weekly.parquet"
    if not cache_path.exists():
        print("No cached NFL data. Run 'python -m src.data.nfl' first.")
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


def get_feature_cols(featured: pd.DataFrame) -> list[str]:
    """Get feature columns, excluding 'days_rest'."""
    import re
    lagged_pattern = re.compile(r".*_avg_\d+$")
    extra_feats = {"b2b", "four_in_six",
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
    # Explicitly drop days_rest — it was the #1 feature by importance but is
    # likely a spurious signal (overfitting to schedule quirks)
    feature_cols = [c for c in feature_cols if c != "days_rest"]
    return feature_cols


def train_qb_passy(featured: pd.DataFrame, stat_name: str, compute_fn=None):
    """Train PASS_YDS regressor on QB-only data."""
    import re

    # ── QB-only filter ──────────────────────────────────────────────
    df = featured.copy()
    if "position" in df.columns:
        df = df[df["position"].str.upper() == "QB"].copy()
    elif "position_group" in df.columns:
        df = df[df["position_group"].str.upper() == "QB"].copy()
    else:
        # Fallback: filter by pass_attempts > 0
        if "pass_attempts" in df.columns:
            df = df[df["pass_attempts"] > 0].copy()
    print(f"  QB rows: {len(df)}", flush=True)

    if len(df) < 500:
        print(f"  Only {len(df)} QB rows — aborting.")
        return None

    # ── Target column ───────────────────────────────────────────────
    target_col = "passing_yards"
    if compute_fn is not None:
        target_col = f"{stat_name.lower()}_computed"
        try:
            df[target_col] = compute_fn(df)
        except Exception as e:
            print(f"  compute_fn failed: {e}")
            return None
    elif target_col not in df.columns:
        print(f"  Column '{target_col}' not found")
        return None

    df = df.dropna(subset=[target_col]).copy()
    print(f"  After dropna: {len(df)} rows", flush=True)

    y = df[target_col].values
    y_mean = y.mean()
    print(f"  PASS_YDS mean (QB only): {y_mean:.1f}", flush=True)

    # ── Features (days_rest excluded) ───────────────────────────────
    feature_cols = get_feature_cols(df)
    print(f"  Features ({len(feature_cols)}):", flush=True)

    available = [c for c in feature_cols if c in df.columns]
    X = df[available].copy()
    X = X.fillna(X.median())

    date_col = "game_date" if "game_date" in df.columns else None
    if date_col:
        dates = pd.to_datetime(df[date_col])
        sort_idx = dates.argsort()
        X = X.iloc[sort_idx]
        y = y[sort_idx]
        dates = dates.iloc[sort_idx]
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y[:split], y[split:]
    else:
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y[:split], y[split:]

    print(f"  Train: {len(X_train)}, Test: {len(X_test)}", flush=True)

    # ── Train with QB-specific hyperparams ──────────────────────────
    model = lgb.LGBMRegressor(
        n_estimators=1500,       # up from 1000
        num_leaves=31,
        learning_rate=0.01,      # down from 0.02
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
              eval_metric="l2",
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

    preds = model.predict(X_test)

    residuals = y_test - preds
    residual_std = float(np.std(residuals))
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    r2 = float(1 - np.sum(residuals ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

    train_mean = y_train.mean()
    test_mean = y_test.mean()
    pred_mean = preds.mean()

    print(f"")
    print(f"  {'Metric':30s} {'Value':>10s}")
    print(f"  {'─'*40}")
    print(f"  {'Train mean':30s} {train_mean:>10.1f}")
    print(f"  {'Test mean':30s} {test_mean:>10.1f}")
    print(f"  {'Pred mean (QB test)':30s} {pred_mean:>10.1f}")
    print(f"  {'Mu bias (pred - actual)':30s} {(pred_mean - test_mean):>+10.1f}")
    print(f"  {'MAE':30s} {mae:>10.3f}")
    print(f"  {'RMSE':30s} {rmse:>10.3f}")
    print(f"  {'R²':30s} {r2:>10.3f}")
    print(f"  {'σ_res':30s} {residual_std:>10.3f}")
    print(f"  {'Best iteration':30s} {model.best_iteration_}", flush=True)

    # ── Calibration check ───────────────────────────────────────────
    y_test_mean = y_test.mean()
    y_test_max = y_test.max()
    cal_bins = []
    max_line = min(int(y_test_max) + 1, max(15, int(y_test_mean * 3.0) + 1))
    min_line = max(1, int(y_test_mean * 0.3))
    raw_probs_all = []
    outcomes_all = []
    print(f"")
    print(f"  Per-line calibration:")
    print(f"  {'Line':>5s} {'P_model':>8s} {'P_act':>8s} {'Bias':>8s} {'n':>6s}")
    print(f"  {'─'*35}")
    for line_val in range(min_line, max_line + 1):
        p_model = np.array([p_ge_stat(stat_name, preds[i], residual_std, line_val)
                            for i in range(len(preds))])
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
        raw_probs_all.extend(p_model.tolist())
        outcomes_all.extend((y_test >= line_val).astype(int).tolist())
        if bias > 0.02 or bias < -0.02 or line_val % 5 == 0:
            print(f"  {line_val:5d} {p_model_mean:>7.1%} {p_actual:>7.1%} {bias:>+7.1%} {len(y_test):6d}")

    # ── Fit BetaCal on QB test-set predictions ──────────────────────
    raw_arr = np.array(raw_probs_all)
    out_arr = np.array(outcomes_all, dtype=int)
    valid = (raw_arr > 0.01) & (raw_arr < 0.99)
    if valid.sum() > 100:
        beta_cal = BetaCalibrator()
        beta_cal.fit(raw_arr[valid], out_arr[valid])
        cal_probs = beta_cal.calibrate(raw_arr[valid])
        before_bias = float(np.mean(raw_arr[valid] - out_arr[valid]))
        after_bias = float(np.mean(cal_probs - out_arr[valid]))
        print(f"\n  BetaCal: a={beta_cal.a:.3f}, b={beta_cal.b:.3f}, c={beta_cal.c:.3f}")
        print(f"  Bias: {before_bias:+.3f} → {after_bias:+.3f} (n={valid.sum()})")
        beta_cal.save(MODEL_DIR / f"lgb_{stat_name.lower()}_beta_cal.json")
    else:
        print(f"\n  BetaCal skipped ({valid.sum()} valid)")

    # ── Feature importance ──────────────────────────────────────────
    imp = pd.DataFrame({"feature": available, "importance": model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False)
    top10 = imp.head(10)["feature"].tolist()
    print(f"\n  Top features:")
    for _, r in imp.head(10).iterrows():
        print(f"    {r['feature']:45s} {r['importance']:.1f}")

    # ── Save model ──────────────────────────────────────────────────
    model.booster_.save_model(str(MODEL_DIR / f"lgb_{stat_name.lower()}.txt"))
    imp.to_csv(MODEL_DIR / f"lgb_{stat_name.lower()}_importance.csv", index=False)

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
        "pos_filter": "QB",
        "learning_rate": 0.01,
        "n_estimators": 1500,
        "days_rest_excluded": True,
    }
    with open(MODEL_DIR / f"lgb_{stat_name.lower()}.meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(MODEL_DIR / f"lgb_{stat_name.lower()}.std.json", "w") as f:
        json.dump({"residual_std": residual_std}, f, indent=2)

    print(f"\n  Saved {MODEL_DIR / f'lgb_{stat_name.lower()}.txt'}")
    return model


def main():
    print("=" * 65)
    print("  NFL QB-ONLY PASS_YDS RETRAINING")
    print("  Changes: QB filter, LR=0.01, n_est=1500, drop days_rest")
    print("=" * 65)

    print("\n1. Loading features...", flush=True)
    featured = load_features()
    if featured is None:
        return
    print(f"  Total rows: {len(featured)}", flush=True)
    print(f"  Seasons: {featured['season'].min()}-{featured['season'].max()}", flush=True)

    # Count QBs
    if "position" in featured.columns:
        qb_count = (featured["position"].str.upper() == "QB").sum()
        print(f"  QB rows: {qb_count}", flush=True)

    # ── PASS_YDS ────────────────────────────────────────────────────────
    print(f"\n2. Training PASS_YDS (QB-only)...", flush=True)
    train_qb_passy(featured, "PASS_YDS")

    # ── PASS_YDS+TD ────────────────────────────────────────────────────
    print(f"\n3. Training PASS_YDS+TD (QB-only)...", flush=True)
    train_qb_passy(featured, "PASS_YDS+TD",
                    compute_fn=lambda df: df["passing_yards"] + df["passing_tds"] * 10)

    print(f"\n{'='*65}")
    print("  Done! Models saved:")
    for f in sorted(MODEL_DIR.glob("lgb_pass_yds*")):
        print(f"    {f.name}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
