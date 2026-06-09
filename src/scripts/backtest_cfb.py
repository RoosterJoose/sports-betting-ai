#!/usr/bin/env python3
"""Backtest CFB models — compare calibrated probability vs actual outcome.

For each CFB target type (win classifier, spread_margin regressor,
total_points regressor), loads the trained XGBoost model, computes
P(stat >= line) via normal CDF with residual sigma, and compares to
actual empirical rates across all line values.

Pattern follows backtest_nba.py: temporal 80/20 split, Brier vs naive,
per-line beats-naive verdict, calibration bins.

Usage:
    python -m src.scripts.backtest_cfb
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config.settings import PROJECT_ROOT
from src.features.cfb import CFBFeatureEngineer
import toml

MODEL_DIR = PROJECT_ROOT / "models" / "cfb"
CACHE_PATH = PROJECT_ROOT / "data" / "cfb_cache" / "game_logs.parquet"

# ── target definitions ──────────────────────────────────────────────────
# Each target has:
#   target_type: "classifier" or "regressor"
#   target_col: column name in featured data
#   display: display name
#   line_range: (low, high) for regressor backtesting

TARGETS = [
    {"target_type": "classifier", "target_col": "win", "display": "Win"},
    {"target_type": "regressor", "target_col": "spread_margin",
     "display": "Spread Margin", "line_low": -28, "line_high": 29},
    {"target_type": "regressor", "target_col": "total_points",
     "display": "Total Points", "line_low": 30, "line_high": 81},
]


def load_features():
    """Load cached CFB data and build features."""
    if not CACHE_PATH.exists():
        print("No cached CFB data at", CACHE_PATH)
        return None

    cfg = toml.load(PROJECT_ROOT / "config" / "cfb.toml")
    from src.config.settings import SportConfig
    scfg = SportConfig(
        name="cfb", display_name="CFB",
        rolling_windows=cfg["features"]["rolling_windows"],
        recency_decay=cfg["features"].get("recency_decay", 0.001),
    )
    fe = CFBFeatureEngineer(scfg)
    all_games = pd.read_parquet(CACHE_PATH)
    featured = fe.build_features(all_games)

    # Ensure target columns are present (feature engineer may strip them)
    for col in ["win", "spread_margin", "total_points", "points_for", "points_against"]:
        if col not in featured.columns and col in all_games.columns:
            featured[col] = all_games[col].values
    # Recompute if raw columns are present
    if "points_for" in featured.columns and "points_against" in featured.columns:
        if "spread_margin" not in featured.columns or featured["spread_margin"].isna().all():
            featured["spread_margin"] = featured["points_for"] - featured["points_against"]
        if "total_points" not in featured.columns or featured["total_points"].isna().all():
            featured["total_points"] = featured["points_for"] + featured["points_against"]

    return featured


def backtest_classifier(featured, target_info):
    """Backtest a binary classifier (win)."""
    import xgboost as xgb

    target_col = target_info["target_col"]
    display = target_info["display"]
    model_path = MODEL_DIR / f"{target_col}.json"
    meta_path = MODEL_DIR / f"{target_col}.meta.json"
    cal_path = MODEL_DIR / f"{target_col}_calibration.json"

    if not model_path.exists():
        print(f"  {display:15s}: Model not found at {model_path}")
        return

    model = xgb.XGBClassifier()
    model.load_model(str(model_path))

    with open(meta_path) as f:
        meta = json.load(f)
    feature_names = meta.get("features", [])

    if target_col not in featured.columns:
        print(f"  {display:15s}: column '{target_col}' not found")
        return

    df = featured.dropna(subset=[target_col]).copy()
    if len(df) < 100:
        print(f"  {display:15s}: only {len(df)} rows")
        return

    y = df[target_col].values.astype(int)
    base_rate = float(np.mean(y))

    available = [c for c in feature_names if c in df.columns]
    if not available:
        print(f"  {display:15s}: no matching features")
        return
    if len(available) != len(feature_names):
        print(f"  {display:15s}: feature mismatch — model expects {len(feature_names)}, data has {len(available)}")
        return

    X = df[available].copy().fillna(0)

    # Temporal 80/20 split
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

    # Predict
    preds = model.predict_proba(X_test.values)[:, 1]
    preds = np.clip(preds, 0.001, 0.999)

    brier = float(np.mean((preds - y_test) ** 2))
    naive_brier = float(base_rate * (1 - base_rate))
    acc = float(np.mean((preds >= 0.5).astype(int) == y_test))

    test_n = len(y_test)
    print(f"\n  {display:15s}: test_n={test_n}  base_rate={base_rate:.1%}  "
          f"OOF_acc={acc:.1%}")
    print(f"         Brier={brier:.4f}  naive_brier={naive_brier:.4f}  "
          f"{'✅ beats by ' + str(int((naive_brier-brier)/naive_brier*100)) + '%' if brier < naive_brier else '❌ worse than naive'}")

    # Calibration by bin
    print(f"\n         {'Bin':>10s}  {'n':>6s}  {'Pred':>6s}  {'Actual':>6s}  {'Err':>7s}")
    print(f"         {'-'*10}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*7}")
    bins = np.arange(0, 1.05, 0.05)
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (preds >= lo) & (preds < hi)
        n = int(mask.sum())
        if n >= 10:
            avg_pred = float(preds[mask].mean())
            actual_rate = float(y_test[mask].mean())
            err = actual_rate - avg_pred
            marker = "✅" if abs(err) < 0.03 else ("⚠️" if abs(err) < 0.08 else "❌")
            print(f"         {lo:.0%}-{hi:.0%}  {n:>6d}  {avg_pred:>5.1%}  "
                  f"{actual_rate:>5.1%}  {err:>+5.1%}  {marker}")

    return brier < naive_brier


def backtest_regressor(featured, target_info):
    """Backtest a regressor (spread_margin or total_points).

    Computes P(stat >= line) via normal CDF using model prediction as mu
    and residual_std as sigma, then compares to actual rates.
    """
    import xgboost as xgb

    target_col = target_info["target_col"]
    display = target_info["display"]
    model_path = MODEL_DIR / f"{target_col}.json"
    meta_path = MODEL_DIR / f"{target_col}.meta.json"

    if not model_path.exists():
        print(f"  {display:15s}: Model not found at {model_path}")
        return

    model = xgb.XGBRegressor()
    model.load_model(str(model_path))

    with open(meta_path) as f:
        meta = json.load(f)
    feature_names = meta.get("features", [])
    sigma = meta.get("residual_std", meta.get("target_std", 1.0))
    # Cap sigma — CFB has high variance, but don't let it explode
    sigma = min(sigma, 30.0)

    if target_col not in featured.columns:
        print(f"  {display:15s}: column '{target_col}' not found")
        return

    df = featured.dropna(subset=[target_col]).copy()
    if len(df) < 100:
        print(f"  {display:15s}: only {len(df)} rows")
        return

    y = df[target_col].values

    available = [c for c in feature_names if c in df.columns]
    if not available:
        print(f"  {display:15s}: no matching features")
        return
    if len(available) != len(feature_names):
        print(f"  {display:15s}: feature mismatch — model expects {len(feature_names)}, data has {len(available)}")
        return

    X = df[available].copy().fillna(0)

    # Temporal 80/20 split
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

    # Predict
    mu = model.predict(X_test.values)
    mu = np.maximum(mu, target_info.get("line_low", -30))

    y_mean = float(np.mean(y_test))
    y_std = float(np.std(y_test))
    low = target_info.get("line_low", int(y_mean - 2 * y_std))
    high = target_info.get("line_high", int(y_mean + 2 * y_std))
    if low >= high:
        low, high = int(y_mean - 15), int(y_mean + 15)

    # Step by 3 for spread, 5 for total points
    step = 3 if target_col == "spread_margin" else 5
    lines = list(range(low, high, step))
    if not lines:
        return

    print(f"\n  {display:15s}: sigma={sigma:.1f}  test_n={len(y_test)}  "
          f"y_mean={y_mean:.1f}  y_std={y_std:.1f}  lines=[{low}..{high}) step={step}")

    results = []
    for line_val in lines:
        # P(stat >= line) via normal CDF
        p_raw = 1.0 - np.array([_norm_cdf(line_val, m, max(sigma, 0.3)) for m in mu])
        p_raw = np.clip(p_raw, 0.001, 0.999)

        actuals = (y_test >= line_val).astype(float)
        p_model = float(np.mean(p_raw))
        p_actual = float(np.mean(actuals))
        n = len(actuals)
        if n < 10:
            continue

        brier = float(np.mean((p_raw - actuals) ** 2))
        naive_brier = float(np.mean((p_actual - actuals) ** 2))
        bias = p_model - p_actual

        results.append({
            "line": line_val, "p_model": round(p_model, 4),
            "p_actual": round(p_actual, 4), "bias": round(bias, 4),
            "brier": round(brier, 4), "naive_brier": round(naive_brier, 4),
            "n": n,
        })

    if not results:
        return

    # Display all lines
    print(f"\n         {'Line':>5s}  {'P_cal':>6s}  {'P_act':>6s}  {'Bias':>7s}  "
          f"{'Brier':>7s}  {'n':>5s}")
    print(f"         {'-'*5}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*5}")
    for r in results:
        marker = "✅" if abs(r["bias"]) < 0.03 else ("⚠️" if abs(r["bias"]) < 0.08 else "❌")
        print(f"         {r['line']:>+5d}  {r['p_model']:>5.0%}  "
              f"{r['p_actual']:>5.0%}  {r['bias']:>+6.1%}  "
              f"{r['brier']:>6.4f}  {r['n']:>5d}  {marker}")

    # Aggregate
    mean_bias = float(np.mean([r["bias"] for r in results]))
    mean_abs_bias = float(np.mean([abs(r["bias"]) for r in results]))
    mean_brier = float(np.mean([r["brier"] for r in results]))
    beats_naive = sum(1 for r in results if r["brier"] < r["naive_brier"])
    total = len(results)

    print(f"\n         {'-'*55}")
    print(f"         Mean Bias: {mean_bias:+.1%}  Mean |Bias|: {mean_abs_bias:.1%}  "
          f"Mean Brier: {mean_brier:.4f}")
    print(f"         Beats naive baseline: {beats_naive}/{total} ({beats_naive/total:.0%})")
    return beats_naive == total


def _norm_cdf(x, mu, sigma):
    """Normal CDF: P(X <= x) given N(mu, sigma)."""
    from math import erf, sqrt
    return 0.5 * (1.0 + erf((x - mu) / (sigma * sqrt(2.0))))


def main():
    print("=" * 70)
    print("  CFB MODEL BACKTEST — Calibrated Probability vs Actual Outcome")
    print("=" * 70)

    print("\n1. Loading features...")
    featured = load_features()
    if featured is None or featured.empty:
        print("  No CFB data. Check CFBD_API_KEY and run train_cfb_models.py")
        return
    print(f"   {len(featured)} rows, {featured['team'].nunique() if 'team' in featured.columns else '?'} teams")

    print("\n2. Backtesting each target...")
    beats_count = 0
    tested_count = 0
    for target_info in TARGETS:
        if target_info["target_type"] == "classifier":
            result = backtest_classifier(featured, target_info)
        else:
            result = backtest_regressor(featured, target_info)
        if result is not None:
            tested_count += 1
            if result:
                beats_count += 1

    print(f"\n{'=' * 70}")
    if tested_count > 0:
        print(f"  Beats naive: {beats_count}/{tested_count}")
    print(f"  Backtest complete")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
