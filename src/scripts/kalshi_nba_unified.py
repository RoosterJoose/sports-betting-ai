#!/usr/bin/env python3
"""NBA Kalshi utility library — model loading, feature engineering, prediction.

Provides shared utilities imported by nba_bet.py (the main NBA scanner/bettor):
  - load_features()   → load & engineer NBA player features
  - _load_regressor()  → load XGBoost model + calibration
  - _match_player()    → match Kalshi title to feature row
  - _p_ge_line()      → P(stat >= line) with distribution + calibration
  - _is_current_market() → date filter for stale markets

Not intended to be run directly. Use nba_bet.py instead:
    python3 src/scripts/nba_bet.py --scan
    python3 src/scripts/nba_bet.py --bet
"""
import json, re, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config.settings import CONFIG_DIR, PROJECT_ROOT
from src.models.calibrator import BetaCalibrator
from src.models.distributions import p_ge_stat
from scipy.stats import norm as _norm

# NBA models are XGBoost .json format
MODEL_DIR = PROJECT_ROOT / "models" / "nba"
WANG_LAMBDA = 0.25  # moderate calibration adjustment for NBA


def _load_regressor(model_name: str):
    """Load XGBoost regressor and metadata for NBA model.

    NBA models are XGBoost .json format (trained by trainer.py).
    Returns (model, residual_std, feature_names, beta_calibrator).
    """
    if model_name is None:
        return None, None, None, BetaCalibrator()

    mn = model_name.lower()
    model_path = MODEL_DIR / f"{mn}.json"
    meta_path = MODEL_DIR / f"{mn}.metrics.json"

    if not model_path.exists():
        print(f"  No model found at {model_path}")
        return None, None, None, BetaCalibrator()

    import xgboost as xgb
    model = xgb.XGBRegressor()
    model.load_model(str(model_path))

    # Get feature names from model JSON (more reliable than feature_names_in_ after load_model)
    try:
        with open(model_path) as f:
            mdata = json.load(f)
        feature_names = mdata.get('learner', {}).get('feature_names', [])
    except Exception:
        feature_names = []

    # Load residual_std from metrics
    std = 1.0
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        std = meta.get("residual_std", meta.get("mae", 1.0))

    # Try loading BetaCal (may not exist for NBA)
    cal_path = MODEL_DIR / f"{mn}_beta_cal.json"
    beta_cal = BetaCalibrator.load(cal_path)

    return model, float(std), feature_names, beta_cal


def _parse_ticker_date(ticker: str):
    """Extract game date from Kalshi ticker like KXNBABLK-26MAY06MINSAS-...
    Returns datetime.date or None.
    """
    m = re.search(r'(\d{2})([A-Z]{3})(\d{2})', ticker)
    if m:
        yr, mo, dy = m.group(1), m.group(2), m.group(3)
        month_map = {
            'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
            'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12,
        }
        month = month_map.get(mo.upper())
        if month:
            from datetime import date
            try:
                return date(2000 + int(yr), month, int(dy))
            except ValueError:
                return None
    return None


def _is_current_market(ticker: str, today=None, window_before=1, window_after=3):
    """Check if a Kalshi market ticker is for a current/recent game.

    Filters out stale markets (e.g., May dates during June Finals).
    Allows markets from `window_before` days ago through `window_after` days ahead.
    """
    from datetime import date, timedelta
    game_date = _parse_ticker_date(ticker)
    if game_date is None:
        return True  # can't parse — let it through
    if today is None:
        today = date.today()
    cutoff_early = today - timedelta(days=window_before)
    cutoff_late = today + timedelta(days=window_after)
    return cutoff_early <= game_date <= cutoff_late


def _match_player(title: str, latest: pd.DataFrame) -> pd.Series:
    """Match player name from Kalshi title to NBA feature data."""
    if not title or latest is None or latest.empty:
        return None

    clean = title.replace("?", "").replace(":", "").strip()
    parts = clean.split()
    if len(parts) < 2:
        return None
    first, last = parts[0], parts[-1]

    # Try exact match on player_name
    if "player_name" in latest.columns:
        exact = latest[latest["player_name"].str.lower() == clean.lower()]
        if len(exact) >= 1:
            return exact.iloc[-1]

    # Try last name match
    if "player_name" in latest.columns:
        lm = latest[latest["player_name"].str.lower().str.contains(last.lower(), na=False)]
        if len(lm) >= 1:
            # Prefer first initial match
            fi = lm[lm["player_name"].str.lower().str[0] == first[0].lower()]
            if len(fi) >= 1:
                return fi.iloc[-1]
            return lm.iloc[-1]

    return None


def _p_ge_line(row, model, residual_std, line_val, feature_names, stat_name="", beta_cal=None):
    """Compute P(stat >= line_val) using distribution mapping + calibration.

    Uses Negative Binomial for volume stats (PTS, REB, AST, PRA, PR, PA, RA),
    Poisson for rare events (BLK, STL, 3PT, FTM).
    Applies Beta Calibration if available, otherwise Wang transform.
    """
    # Build feature vector matching model expectations
    feat_dict = {}
    for c in feature_names:
        if c in row.index:
            val = row[c]
            if pd.isna(val):
                val = 0.0
            feat_dict[c] = float(val)
        else:
            feat_dict[c] = 0.0

    X_pred = pd.DataFrame([feat_dict]).fillna(0)
    mu = model.predict(X_pred)[0]
    sigma = max(residual_std, 0.3)

    # Step 1: Distribution-appropriate probability mapping
    p_raw = p_ge_stat(stat_name, mu, sigma, line_val)

    # Step 2: Beta Calibration if available
    if beta_cal is not None and beta_cal._fitted:
        p_corrected = beta_cal(p_raw)
    else:
        # Fallback: Wang Transform
        z = _norm.ppf(p_raw)
        p_corrected = _norm.cdf(z - WANG_LAMBDA)

    p_corrected = min(0.999, float(p_corrected))
    return max(0.001, p_corrected), float(mu)


def load_features():
    """Load NBA data and build features using NBADataSource + NBAFeatureEngineer.

    Uses the same pipeline as training, loads from cached parquet.
    """
    import toml
    from src.data.nba import NBADataSource
    from src.features.nba import NBAFeatureEngineer
    from src.config.settings import SportConfig

    cfg_path = CONFIG_DIR / "nba.toml"
    if cfg_path.exists():
        cfg = toml.load(cfg_path)
    else:
        cfg = {"features": {"rolling_windows": [3, 5, 10, 20], "recency_decay": 0.001}}

    scfg = SportConfig(
        name="nba", display_name="NBA",
        rolling_windows=cfg["features"]["rolling_windows"],
        recency_decay=cfg["features"].get("recency_decay", 0.001),
    )
    fe = NBAFeatureEngineer(scfg)

    # Load from cache
    cache_path = PROJECT_ROOT / "data" / "nba_cache" / "game_logs_v14.parquet"
    if not cache_path.exists():
        print("  No cached NBA data. Run data pipeline first.")
        return None

    all_games = pd.read_parquet(cache_path)
    print(f"  Loaded {len(all_games)} raw rows", flush=True)

    # Ensure player_name exists in raw data
    if "player_name" not in all_games.columns:
        all_games["player_name"] = all_games.get("player_display_name", "")

    featured = fe.build_features(all_games)
    print(f"  Feature engineering: {len(featured)} rows, {len(featured.columns)} cols", flush=True)

    # Merge player_name back into featured data
    if "player_name" in all_games.columns and "player_id" in all_games.columns and "game_date" in all_games.columns:
        merge_df = all_games[["player_id", "game_date", "player_name"]].drop_duplicates(
            subset=["player_id", "game_date"]
        )
        merge_df["game_date"] = pd.to_datetime(merge_df["game_date"])
        featured["game_date"] = pd.to_datetime(featured["game_date"])
        featured = featured.merge(merge_df, on=["player_id", "game_date"], how="left")

    return featured


if __name__ == "__main__":
    # Redirect to nba_bet.py (this module's main() was broken by -m loading bug)
    print("Use nba_bet.py instead: python3 src/scripts/nba_bet.py --scan")
    import subprocess, sys as _sys
    _sys.exit(subprocess.call(["python3", str(Path(__file__).resolve().parent / "nba_bet.py")] + _sys.argv[1:]))
