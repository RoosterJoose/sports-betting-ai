#!/usr/bin/env python3
"""Backtest MLB regression models — compare calibrated probability vs actual outcome.

For each scanner stat (SO, HR, TB, H_R_RBI), loads the trained LGBM model +
BetaCal, computes P(stat >= line) via normal CDF → BetaCal, and compares
to actual empirical rates across all line values.

Usage:
    python -m src.scripts.backtest_mlb
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

# Scanner market types: (stat_name, raw_col, pos_filter, compute_fn)
STAT_DEFS = [
    ("SO",      "so",   "pitcher", None),
    ("HR",      "hr",   "hitter",  None),
    ("TB",      "tb",   "hitter",  None),
    ("H_R_RBI", None,   "hitter",  lambda df: df["h"] + df["r"] + df["rbi"]),
    ("IP",      "ip",   "pitcher", None),
    ("H",       "h",    "pitcher", None),
    ("BB",      "bb",   "pitcher", None),
    ("ER",      "er",   "pitcher", None),
    ("R",       "r",    "hitter",  None),
    ("RBI",     "rbi",  "hitter",  None),
    ("SB",      "sb",   "hitter",  None),
]


def load_features():
    """Load cached MLB data and build features (same as train_mlb_regression.py)."""
    cache_dir = PROJECT_ROOT / "data" / "cache" / "mlb"
    cache_files = sorted(cache_dir.glob("game_logs_*.parquet"))
    if not cache_files:
        print("No cached data.")
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


def backtest_stat(featured, stat_name, raw_col, pos_filter, compute_fn):
    """Backtest a single MLB stat: compute P_model vs P_actual for all line values."""
    mn = stat_name.lower()
    model_path = MODEL_DIR / f"lgb_{mn}.txt"
    meta_path = MODEL_DIR / f"lgb_{mn}.meta.json"
    cal_path = CALIB_DIR / f"{mn}_beta_cal.json"

    if not model_path.exists() or not meta_path.exists():
        print(f"  {stat_name:8s}: Model not found")
        return

    # Load model + BetaCal
    model = lgb.Booster(model_file=str(model_path))
    with open(meta_path) as f:
        meta = json.load(f)
    residual_std = meta.get("residual_std", 1.0)
    model_features = meta.get("features", model.feature_name())
    beta_cal = BetaCalibrator.load(cal_path)
    beta_str = f"BetaCal: a={beta_cal.a:.3f}, b={beta_cal.b:.3f}, c={beta_cal.c:.3f}" if beta_cal._fitted else "No BetaCal"

    # Filter by position
    df = featured.copy()
    if pos_filter == "pitcher" and "position" in featured.columns:
        df = featured[featured["position"] == "P"].copy()
    elif pos_filter == "hitter" and "position" in featured.columns:
        df = featured[featured["position"] != "P"].copy()

    # Target column
    target_col = raw_col or f"{stat_name.lower()}_computed"
    if compute_fn is not None:
        df[target_col] = compute_fn(df)
    elif raw_col:
        target_col = raw_col
    if target_col not in df.columns:
        print(f"  {stat_name}: column '{target_col}' not found")
        return

    df = df.dropna(subset=[target_col]).copy()
    if len(df) < 100:
        print(f"  {stat_name}: only {len(df)} rows")
        return

    y = df[target_col].values
    available = [c for c in model_features if c in df.columns]
    X = df[available].copy()
    X = X.fillna(X.median())

    # Temporal 80/20 split
    dates = pd.to_datetime(df["game_date"])
    sort_idx = dates.argsort()
    X = X.iloc[sort_idx]
    y = y[sort_idx]
    df_sorted = df.iloc[sort_idx]
    split = int(len(X) * 0.8)
    X_test = X.iloc[split:]
    y_test = y[split:]
    df_test = df_sorted.iloc[split:]

    # Predict mu
    test_feat = pd.DataFrame(index=range(len(X_test)))
    for c in model_features:
        test_feat[c] = X_test[c].values if c in X_test.columns else 0.0
    mu = model.predict(test_feat.fillna(0))

    # Determine line range
    y_mean = float(np.mean(y_test))
    y_max = float(np.max(y_test))
    low = max(1, int(max(y_mean * 0.2, 0)))
    high = min(int(y_max) + 1, max(int(y_mean * 3.0) + 1, int(y_mean) + 5))
    if low >= high:
        print(f"  {stat_name}: no valid line range [{low}, {high})")
        return

    lines = list(range(low, high))

    print(f"\n  {stat_name:8s}: σ={residual_std:.3f}  test_n={len(df_test)}  "
          f"y_mean={y_mean:.2f}  y_max={y_max:.0f}  lines=[{low}..{high})")
    print(f"           {beta_str}")

    # Compute P(>=line) for each line using distribution-appropriate mapping
    # Use vectorized p_ge_stat (pass whole mu array) for performance
    results = []
    for line_val in lines:
        p_raw = p_ge_stat(stat_name, mu, max(residual_std, 0.3), line_val)
        p_raw = np.clip(p_raw, 0.001, 0.999)

        # BetaCal
        if beta_cal._fitted:
            p_cal = beta_cal(p_raw)
        else:
            p_cal = p_raw
        p_cal = np.clip(p_cal, 0.001, 0.999)

        actuals = (y_test >= line_val).astype(float)
        p_model = float(np.mean(p_cal))
        p_actual = float(np.mean(actuals))
        n = len(actuals)
        if n < 10:
            continue

        brier = float(np.mean((p_cal - actuals) ** 2))
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
        return

    # Display key lines
    key_lines = set()
    for r in results:
        if abs(r["bias"]) > 0.03:
            key_lines.add(r["line"])
        if r["line"] % 1 == 0:  # every line since MLB lines are small ints
            key_lines.add(r["line"])
    key_lines = sorted(k for k in key_lines if low <= k < high)

    if len(key_lines) > 12:
        biased = sorted(results, key=lambda r: abs(r["bias"]), reverse=True)
        keep = set(r["line"] for r in biased[:8])
        key_lines = sorted(k for k in keep if low <= k < high)

    print(f"\n           {'Line':>4s}  {'P_cal':>6s}  {'P_act':>6s}  {'Bias':>7s}  {'Brier':>7s}  {'n':>4s}")
    print(f"           {'─'*4:>4s}  {'─'*6:>6s}  {'─'*6:>6s}  {'─'*7:>7s}  {'─'*7:>7s}  {'─'*4:>4s}")
    for r in results:
        if r["line"] not in key_lines:
            continue
        marker = "✅" if abs(r["bias"]) < 0.03 else ("⚠️" if abs(r["bias"]) < 0.08 else "❌")
        print(f"           {r['line']:>4d}+  {r['p_model']:>5.0%}  "
              f"{r['p_actual']:>5.0%}  {r['bias']:>+5.1%}  "
              f"{r['brier']:>6.4f}  {r['n']:>4d}  {marker}")

    # Aggregate
    mean_bias = float(np.mean([r["bias"] for r in results]))
    mean_abs_bias = float(np.mean([abs(r["bias"]) for r in results]))
    mean_brier = float(np.mean([r["brier"] for r in results]))
    beats_naive = sum(1 for r in results if r["brier"] < r["naive_brier"])
    total = len(results)

    print(f"\n           {'─'*55}")
    print(f"           Mean Bias: {mean_bias:+.1%}  Mean |Bias|: {mean_abs_bias:.1%}  "
          f"Mean Brier: {mean_brier:.4f}")
    print(f"           Beats naive baseline: {beats_naive}/{total} ({beats_naive/total:.0%})")

    # Tercile calibration
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
            print(f"           {label} [line {lo}..{hi}]: bias={mb:+.1%}  |bias|={mab:.1%}")


def main():
    print("=" * 70)
    print("  MLB MODEL BACKTEST — Calibrated Probability vs Actual Outcome")
    print("=" * 70)

    print("\n1. Loading features...")
    featured = load_features()
    if featured is None:
        return
    print(f"   {len(featured)} rows")

    print("\n2. Backtesting each stat...")
    for stat_name, raw_col, pos_filter, compute_fn in STAT_DEFS:
        backtest_stat(featured, stat_name, raw_col, pos_filter, compute_fn)

    print(f"\n{'='*70}")
    print("  Backtest complete")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
