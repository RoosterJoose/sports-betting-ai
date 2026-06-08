#!/usr/bin/env python3
"""Backtest NFL models — compare calibrated probability vs actual outcome for each stat line.

For each stat, loads the trained LGBM model + BetaCal, predicts P(stat >= line)
using NB/Poisson mapping (batched), and compares to the actual empirical rate.

Usage:
    python -m src.scripts.backtest_nfl
"""
from __future__ import annotations

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.settings import CONFIG_DIR, PROJECT_ROOT
from src.features.nfl import NFLFeatureEngineer
from src.models.calibrator import BetaCalibrator
from src.models.distributions import p_ge_stat
import lightgbm as lgb
import toml

MODEL_DIR = PROJECT_ROOT / "models" / "nfl"

# ── Stat targets (matching train_nfl_regression.py) ──────────────
STAT_TARGETS: list[tuple[str, str | None, callable]] = [
    ("PASS_YDS",     "passing_yards",     None),
    ("PASS_TD",      "passing_tds",       None),
    ("PASS_ATT",     "pass_attempts",     None),
    ("INT",          "interceptions",     None),
    ("PASS_YDS+TD",  None,                lambda df: df["passing_yards"] + df["passing_tds"] * 10),
    ("RUSH_YDS",     "rushing_yards",     None),
    ("REC",          "receptions",        None),
    ("REC_YDS",      "receiving_yards",   None),
    ("RUSH+REC_YDS", None,                lambda df: df["rushing_yards"] + df["receiving_yards"]),
    ("TD",           "touchdowns",        None),
]

LOG_TRANSFORM_STATS = {"TD", "PASS_TD", "INT"}


def load_features():
    """Load NFL features using the same pipeline as training."""
    cache_path = PROJECT_ROOT / "data" / "nfl_cache" / "weekly.parquet"
    if not cache_path.exists():
        print(f"  No cached data at {cache_path}")
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
    print(f"  Loaded {len(all_games)} rows from {cache_path}", flush=True)

    if "player_name" not in all_games.columns and "player_display_name" in all_games.columns:
        all_games["player_name"] = all_games["player_display_name"]

    featured = fe.build_features(all_games)

    # Keep raw stat cols for computing targets & actuals
    stat_cols = ["passing_yards", "passing_tds", "passing_air_yards", "interceptions",
                 "rushing_yards", "rushing_tds", "carries",
                 "receiving_yards", "receiving_tds", "receptions", "targets",
                 "touchdowns", "fantasy_points", "pass_attempts", "rush_attempts",
                 "completions", "games_started", "availability",
                 "position", "player_name", "team_abbr", "recent_team",
                 "opponent_team"]
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


def _load_nfl_model(stat_name: str):
    """Load LGBM model + std + BetaCal for a stat."""
    mn = stat_name.lower()
    model_path = MODEL_DIR / f"lgb_{mn}.txt"
    std_path = MODEL_DIR / f"lgb_{mn}.std.json"
    cal_path = MODEL_DIR / f"lgb_{mn}_beta_cal.json"

    if not model_path.exists():
        return None, None, [], None

    model = lgb.Booster(model_file=str(model_path))
    true_features = model.feature_name()

    std = 1.0
    if std_path.exists():
        with open(std_path) as f:
            std = json.load(f).get("residual_std", 1.0)

    beta_cal = BetaCalibrator.load(cal_path)
    return model, float(std), true_features, beta_cal


def _build_predictions_batch(df: pd.DataFrame, model, true_features, std: float,
                              stat_name: str, beta_cal, lines: list[int]):
    """Batch compute P(stat >= line) for all rows and all lines.

    Returns dict[line_val, (p_cal_arr, actual_arr)].
    """
    # Build feature matrix once
    X_list = []
    valid_idx = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        feat_dict = {}
        ok = True
        for c in true_features:
            if c in row.index and pd.notna(row[c]):
                feat_dict[c] = float(row[c])
            else:
                feat_dict[c] = 0.0
        X_list.append(feat_dict)
        valid_idx.append(idx)

    if not X_list:
        return {}

    X_pred = pd.DataFrame(X_list).fillna(0)

    # Predict mu for all rows in one batch
    mu_all = model.predict(X_pred)
    if stat_name in LOG_TRANSFORM_STATS:
        mu_all = np.maximum(0, mu_all)
        mu_all = np.expm1(mu_all)
    sigma = max(std, 0.3)

    results = {}
    target_col = None
    for line_val in lines:
        # Vectorized p_ge_stat for all rows
        p_raw_all = np.array([p_ge_stat(stat_name, mu, sigma, line_val)
                              for mu in mu_all])

        # Apply Beta Calibration
        if beta_cal is not None and beta_cal._fitted:
            p_cal_all = beta_cal(p_raw_all)
        else:
            p_cal_all = p_raw_all

        # Cap at [0.001, 0.999]
        p_cal_all = np.clip(p_cal_all, 0.001, 0.999)

        results[line_val] = p_cal_all

    return results


def backtest_stat(featured: pd.DataFrame, stat_name: str, raw_col: str | None,
                  compute_fn):
    """Backtest a single stat: compute P_model vs P_actual for all line values."""
    model, std, true_features, beta_cal = _load_nfl_model(stat_name)
    if model is None:
        print(f"  {stat_name:12s}: No trained model found")
        return

    # Prepare target column
    target_col = raw_col or f"{stat_name.lower()}_computed"
    df = featured.copy()

    if compute_fn is not None:
        try:
            df[target_col] = compute_fn(df)
        except Exception as e:
            print(f"  {stat_name}: compute_fn failed: {e}")
            return
    elif raw_col:
        if raw_col not in df.columns:
            print(f"  {stat_name}: column '{raw_col}' not found")
            return
        target_col = raw_col
    else:
        print(f"  {stat_name}: no target column specified")
        return

    df = df.dropna(subset=[target_col]).copy()
    if len(df) < 100:
        print(f"  {stat_name}: only {len(df)} rows")
        return

    y = df[target_col].values

    # Time-based split (same 80/20 as training)
    date_col = "game_date" if "game_date" in df.columns else None
    if date_col:
        dates = pd.to_datetime(df[date_col])
        sort_idx = dates.argsort()
        df = df.iloc[sort_idx].reset_index(drop=True)
        y = y[sort_idx]
        split = int(len(df) * 0.8)
        df_test = df.iloc[split:].copy()
        y_test = y[split:]
    else:
        split = int(len(df) * 0.8)
        df_test = df.iloc[split:].copy()
        y_test = y[split:]

    # Determine lines to test
    y_mean = float(np.mean(y_test))
    y_max = float(np.max(y_test))
    y_min = float(np.min(y_test))
    low = max(1, int(max(y_mean * 0.2, y_min)))
    high = min(int(y_max) + 1, max(int(y_mean * 3.0) + 1, int(y_mean) + 5))
    if low >= high:
        print(f"  {stat_name}: no valid line range [{low}, {high})")
        return

    lines = list(range(low, high))

    print(f"\n  {stat_name:12s}: σ_res={std:.3f}  test_n={len(df_test)}  "
          f"y_mean={y_mean:.2f}  y_max={y_max:.0f}  lines=[{low}..{high})")
    if beta_cal and beta_cal._fitted:
        print(f"               BetaCal: a={beta_cal.a:.3f}, b={beta_cal.b:.3f}, c={beta_cal.c:.3f}")

    # ── Batched predictions ──────────────────────────────────────
    print(f"               Predicting for {len(lines)} lines × {len(df_test)} rows...", flush=True)
    results_dict = _build_predictions_batch(
        df_test, model, true_features, std, stat_name, beta_cal, lines
    )
    if not results_dict:
        print(f"               No valid predictions")
        return

    # ── Build per-line results ───────────────────────────────────
    results = []
    for line_val in lines:
        p_cal_all = results_dict[line_val]
        actuals = (y_test >= line_val).astype(float)
        p_model = float(np.mean(p_cal_all))
        p_actual = float(np.mean(actuals))
        n = len(actuals)

        if n < 10:
            continue

        brier = float(np.mean((p_cal_all - actuals) ** 2))
        prior = p_actual
        naive_brier = float(np.mean((prior - actuals) ** 2))

        bias = p_model - p_actual

        results.append({
            "line": line_val, "p_model": round(p_model, 4),
            "p_actual": round(p_actual, 4), "bias": round(bias, 4),
            "brier": round(brier, 4), "naive_brier": round(naive_brier, 4),
            "n": n,
        })

    if not results:
        print(f"               No valid lines after filtering")
        return

    # ── Display: key lines + extremes with bias ──────────────────

    # Pick lines to show: any line where bias > 3%, plus at least every 5th line
    key_lines = set()
    for r in results:
        if abs(r["bias"]) > 0.03:
            key_lines.add(r["line"])
        if r["line"] % 5 == 0:
            key_lines.add(r["line"])

    # Always show ~mean, mean*1.5, mean*2
    for mult in [0.5, 1.0, 1.5, 2.0, 2.5]:
        key_lines.add(int(y_mean * mult))

    key_lines = sorted(k for k in key_lines if low <= k < high)

    # If too many, keep the most biased and the most common lines
    if len(key_lines) > 15:
        biased = sorted(results, key=lambda r: abs(r["bias"]), reverse=True)
        keep = set()
        for r in biased[:8]:
            keep.add(r["line"])
        for mult in [1.0, 1.5]:
            keep.add(int(y_mean * mult))
        key_lines = sorted(k for k in keep if low <= k < high)

    sepa = "─"
    cols = f"{'Line':>5s}  {'P_cal':>7s}  {'P_act':>7s}  {'Bias':>9s}  {'Brier':>9s}  {'n':>5s}"
    sep_line = f"{sepa*5:>5s}  {sepa*7:>7s}  {sepa*7:>7s}  {sepa*9:>9s}  {sepa*9:>9s}  {sepa*5:>5s}"

    print(f"\n               {cols}")
    print(f"               {sep_line}")

    for r in results:
        if r["line"] not in key_lines:
            continue
        marker = "✅" if abs(r["bias"]) < 0.03 else ("⚠️" if abs(r["bias"]) < 0.08 else "❌")
        print(f"               {r['line']:>5d}+  {r['p_model']:>6.0%}  "
              f"{r['p_actual']:>6.0%}  {r['bias']:>+7.1%}  "
              f"{r['brier']:>8.4f}  {r['n']:>5d}  {marker}")

    # ── Aggregate ────────────────────────────────────────────────
    mean_bias = float(np.mean([r["bias"] for r in results]))
    mean_abs_bias = float(np.mean([abs(r["bias"]) for r in results]))
    mean_brier = float(np.mean([r["brier"] for r in results]))
    beats_naive = sum(1 for r in results if r["brier"] < r["naive_brier"])
    total = len(results)

    print(f"\n               {sepa*65}")
    print(f"               Mean Bias: {mean_bias:+.1%}  Mean |Bias|: {mean_abs_bias:.1%}  "
          f"Mean Brier: {mean_brier:.4f}")
    print(f"               Beats naive baseline: {beats_naive}/{total} ({beats_naive/total:.0%})")

    # Also report calibration by tercile
    if len(results) >= 6:
        tercile = len(results) // 3
        for label, slice_res in [("Low lines    ", results[:tercile]),
                                  ("Mid lines    ", results[tercile:2*tercile]),
                                  ("High lines   ", results[2*tercile:])]:
            if not slice_res:
                continue
            mb = float(np.mean([r["bias"] for r in slice_res]))
            mab = float(np.mean([abs(r["bias"]) for r in slice_res]))
            lo = slice_res[0]["line"]
            hi = slice_res[-1]["line"]
            print(f"               {label} [line {lo}..{hi}]: bias={mb:+.1%}  |bias|={mab:.1%}")


def main():
    print("=" * 80)
    print("  NFL MODEL BACKTEST — Calibrated Probability vs Actual Outcome")
    print("=" * 80)

    print("\n1. Loading features...")
    featured = load_features()
    if featured is None:
        return
    print(f"   {len(featured)} rows, {featured['season'].min()}-{featured['season'].max()}", flush=True)

    print("\n2. Backtesting each stat...")
    for stat_name, raw_col, compute_fn in STAT_TARGETS:
        backtest_stat(featured, stat_name, raw_col, compute_fn)

    print(f"\n{'='*80}")
    print("  Backtest complete")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
