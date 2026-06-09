#!/usr/bin/env python3
"""Build multi-bin empirical calibration for MLB player prop models — vectorized.

For each stat model, loads features, runs temporal 80/20 split, and for each
test-set row computes P(stat >= line) via normal CDF.  Then buckets the raw
predicted probabilities into 20 bins and records the actual rate per bin.

Replaces old single-bin _empirical.json files with proper multi-bin tables.
The calibration is then used by _p_ge_line instead of the Wang transform.

Usage:
    python -m src.scripts.build_calibration
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.settings import CONFIG_DIR, PROJECT_ROOT
from src.features.mlb import MLBFeatureEngineer
from src.models.calibrator import BetaCalibrator
import toml
import lightgbm as lgb

WANG_LAMBDA = 0.30  # fallback (only used when neither empirical nor Beta exists)

MODEL_DIR = PROJECT_ROOT / "models" / "mlb"
CALIB_DIR = MODEL_DIR / "calibration"
CALIB_DIR.mkdir(parents=True, exist_ok=True)

N_BINS = 20

# (model_name, pos_filter, line_values)
TARGETS = [
    ("SO",      "pitcher", [3, 4, 5, 6, 7]),
    ("HR",      "hitter",  [1]),
    ("TB",      "hitter",  [1, 2, 3]),
    ("H_R_RBI", "hitter",  [1, 2, 3, 4]),
    ("IP",      "pitcher", [14, 15, 16, 17, 18]),      # outs ~4.5-6 IP
    ("H",       "pitcher", [3, 4, 5, 6, 7]),
    ("BB",      "pitcher", [1, 2, 3]),
    ("ER",      "pitcher", [1, 2, 3, 4]),
    ("R",       "hitter",  [1]),
    ("RBI",     "hitter",  [1]),
    ("SB",      "hitter",  [1]),
]


def _compute_stat(df: pd.DataFrame, model_name: str) -> np.ndarray:
    """Return the stat value per row as a numpy array."""
    if model_name == "SO":
        return df["so"].values if "so" in df.columns else np.zeros(len(df))
    if model_name == "HR":
        return df["hr"].values if "hr" in df.columns else np.zeros(len(df))
    if model_name == "TB":
        one = df.get("1b", pd.Series(0, index=df.index)).values
        two = df.get("2b", pd.Series(0, index=df.index)).values * 2
        three = df.get("3b", pd.Series(0, index=df.index)).values * 3
        four = df.get("hr", pd.Series(0, index=df.index)).values * 4
        return one + two + three + four
    if model_name == "H_R_RBI":
        h = df.get("h", pd.Series(0, index=df.index)).values
        r = df.get("r", pd.Series(0, index=df.index)).values
        rbi = df.get("rbi", pd.Series(0, index=df.index)).values
        return h + r + rbi
    # Generic: use column matching model_name lowercased
    col = model_name.lower()
    if col in df.columns:
        return df[col].values
    return np.zeros(len(df))


def build_all():
    print(f"Building calibration  {datetime.now().strftime('%H:%M:%S')}")
    print(f"Targets: {len(TARGETS)} models\n")

    # ── Load features once ────────────────────────────────────────────
    cfg = toml.load(CONFIG_DIR / "mlb.toml")
    from src.config.settings import SportConfig

    scfg = SportConfig(
        name="mlb", display_name="MLB",
        rolling_windows=cfg["features"]["rolling_windows"],
        recency_decay=0.001,
    )
    fe = MLBFeatureEngineer(scfg)
    cache_dir = PROJECT_ROOT / "data/cache/mlb"
    cache_files = sorted(cache_dir.glob("game_logs_*.parquet"))
    if not cache_files:
        print("No cached data found.")
        return

    print("Loading and building features...", end=" ", flush=True)
    all_games = pd.concat([pd.read_parquet(f) for f in cache_files], ignore_index=True)
    featured = fe.build_features(all_games)
    print(f"{len(featured)} rows", flush=True)

    # Merge raw stat columns for target computation
    raw_cols = ["so", "er", "h", "bb", "hr", "tb", "rbi", "sb", "ip", "r",
                 "1b", "2b", "3b", "position", "player_name", "team_abbr", "gs"]
    avail_raw = [c for c in raw_cols if c in all_games.columns]
    if "game_date" in all_games.columns and "player_id" in featured.columns:
        all_games["game_date"] = pd.to_datetime(all_games["game_date"])
        featured["game_date"] = pd.to_datetime(featured["game_date"])
        featured = featured.merge(
            all_games[["player_id", "game_date"] + avail_raw],
            on=["player_id", "game_date"], how="left",
        )

    # Ensure game_date is datetime for splitting
    if "game_date" in featured.columns:
        featured["game_date"] = pd.to_datetime(featured["game_date"])
    print(f"  {len(featured.columns)} columns after merge", flush=True)

    # ── Build for each target ─────────────────────────────────────────
    for model_name, pos_filter, line_values in TARGETS:
        print(f"\n{'='*60}")
        print(f"  {model_name}")
        print(f"{'='*60}")

        # ── Load model ────────────────────────────────────────────────
        lgb_path = MODEL_DIR / f"lgb_{model_name.lower()}.txt"
        meta_path = MODEL_DIR / f"lgb_{model_name.lower()}.meta.json"
        if not lgb_path.exists():
            print(f"  No model at {lgb_path}")
            continue

        model = lgb.Booster(model_file=str(lgb_path))
        with open(meta_path) as f:
            meta = json.load(f)
        residual_std = meta.get("residual_std", 1.0)
        model_feats = model.feature_name()
        print(f"  Model: {len(model_feats)} features, σ_res={residual_std:.3f}")

        # ── Filter by position ────────────────────────────────────────
        if pos_filter == "pitcher" and "position" in featured.columns:
            df = featured[featured["position"] == "P"].copy()
        elif pos_filter == "hitter" and "position" in featured.columns:
            df = featured[featured["position"] != "P"].copy()
        else:
            df = featured.copy()

        # Compute target stat
        stat_vals = _compute_stat(df, model_name)
        df["_stat"] = stat_vals
        df = df.dropna(subset=["_stat"]).copy()
        if len(df) < 500:
            print(f"  Only {len(df)} rows — skipping")
            continue
        print(f"  {len(df)} rows after position filter")

        # ── Temporal split (80/20) ────────────────────────────────────
        if "game_date" in df.columns:
            dates = pd.to_datetime(df["game_date"])
            sort_idx = np.argsort(dates.values)
        else:
            sort_idx = np.arange(len(df))
        df = df.iloc[sort_idx].reset_index(drop=True)
        split = int(len(df) * 0.8)
        test_df = df.iloc[split:].reset_index(drop=True)
        print(f"  Test set: {len(test_df)} rows", flush=True)

        # ── Build feature matrix matching model's expected columns ─────
        # One large numpy array for the entire test set
        n_test = len(test_df)
        X = np.zeros((n_test, len(model_feats)), dtype=np.float64)
        for j, col in enumerate(model_feats):
            if col in test_df.columns:
                vals = pd.to_numeric(test_df[col], errors="coerce").values
                X[:, j] = np.nan_to_num(vals, nan=0.0)

        # ── Predict mu for all test rows ──────────────────────────────
        print(f"  Predicting...", end=" ", flush=True)
        mus = model.predict(X)
        print(f"done", flush=True)

        # ── For each line value ───────────────────────────────────────
        cal_data = {}
        raw_probs_list = []  # collected across line values for BetaCal fitting
        outcomes_list = []
        for line_val in line_values:
            sigma = max(residual_std, 0.3)
            # Vectorized: p_raw = P(stat >= line) via normal CDF
            z = (line_val - 0.5 - mus) / sigma
            p_raw_vec = _norm.cdf(-z)  # = 1 - norm.cdf(z)
            np.clip(p_raw_vec, 0.001, 0.999, out=p_raw_vec)

            # Vectorized: actual = 1 if stat >= line
            actual_vec = (test_df["_stat"].values >= line_val).astype(np.float64)

            # Accumulate for Beta Calibration fitting
            raw_probs_list.append(p_raw_vec)
            outcomes_list.append(actual_vec)

            # Bucket into N_BINS bins
            bins = []
            for i in range(N_BINS):
                lo, hi = i / N_BINS, (i + 1) / N_BINS
                mask = (p_raw_vec >= lo) & (p_raw_vec < hi)
                n = int(mask.sum())
                if n >= 5:
                    avg_pred = float(p_raw_vec[mask].mean())
                    actual_rate = float(actual_vec[mask].mean())
                    bins.append({
                        "p_pred_min": round(lo, 3),
                        "p_pred_max": round(hi, 3),
                        "p_actual": round(actual_rate, 4),
                        "p_avg_pred": round(avg_pred, 4),
                        "n": n,
                    })

            # Overall Brier
            brier = float(np.mean((p_raw_vec - actual_vec) ** 2))
            naive = float(np.mean((actual_vec.mean() - actual_vec) ** 2))

            # Print summary
            print(f"  line={line_val:2d}+: n={n_test:6d}  "
                  f"Brier={brier:.4f}  naive={naive:.4f}  "
                  f"σ_res={sigma:.2f}", flush=True)

            # Show bins with >5% error
            for b in bins:
                if abs(b["p_actual"] - b["p_avg_pred"]) > 0.05:
                    print(f"    [{b['p_pred_min']:.1f},{b['p_pred_max']:.1f})  "
                          f"pred={b['p_avg_pred']:.0%}  actual={b['p_actual']:.0%}  "
                          f"n={b['n']}")

            cal_data[str(line_val)] = {"bins": bins}

        # ── Fit Beta Calibration ──────────────────────────────────────
        # Concatenate all (raw_prob, outcome) pairs across all line values
        all_raw = np.concatenate(raw_probs_list)
        all_out = np.concatenate(outcomes_list)
        valid = (all_raw > 0.01) & (all_raw < 0.99)
        n_valid = int(valid.sum())
        if n_valid > 100:
            beta_cal = BetaCalibrator()
            beta_cal.fit(all_raw[valid], all_out[valid].astype(int))
            beta_path = CALIB_DIR / f"{model_name.lower()}_beta_cal.json"
            beta_cal.save(beta_path)

            # Measure bias reduction
            before = float(np.mean(all_raw[valid] - all_out[valid]))
            after = float(np.mean(beta_cal.calibrate(all_raw[valid]) - all_out[valid]))
            print(f"  BetaCal: bias {before:+.3f} → {after:+.3f} (n={n_valid})")
        else:
            print(f"  BetaCal: skipped ({n_valid} valid predictions < 100)")

        # ── Save ──────────────────────────────────────────────────────
        if cal_data:
            cal_path = CALIB_DIR / f"{model_name.lower()}_empirical.json"
            with open(cal_path, "w") as f:
                json.dump(cal_data, f, indent=2)
            print(f"  Saved → {cal_path.name}")
            total_bins = sum(len(v["bins"]) for v in cal_data.values())
            print(f"  {len(cal_data)} lines × {total_bins // max(len(cal_data), 1):.0f} bins avg")

    print(f"\nDone at {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    build_all()
