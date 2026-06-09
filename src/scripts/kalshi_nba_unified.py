#!/usr/bin/env python3
"""Unified Kalshi NBA bettor — covers all available NBA player prop markets.

Scans Kalshi for NBA markets, loads the corresponding XGBoost regressor,
computes edge with distribution-appropriate probability + Wang calibration,
and identifies qualifying bets.

Market types:
  KXNBAPTS  → points
  KXNBAREB  → rebounds
  KXNBAAST  → assists
  KXNBABLK  → blocks
  KXNBASTL  → steals
  KXNBA3PT  → three-pointers made
  KXNBAFTM  → free throws made
  KXNBAPRA  → points + rebounds + assists
  KXNBAPA   → points + assists
  KXNBAPR   → points + rebounds
  KXNBARA   → rebounds + assists
  KXNBA2D   → double double (info_only — no trained model)
  KXNBA3D   → triple double (info_only — no trained model)

Usage:
    python -m src.scripts.kalshi_nba_unified --scan
    python -m src.scripts.kalshi_nba_unified --bet
"""
import sys, re, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.kalshi import KalshiClient
from src.config.settings import CONFIG_DIR, PROJECT_ROOT
from src.models.calibrator import BetaCalibrator
from src.models.distributions import p_ge_stat
from scipy.stats import norm as _norm
import toml

# NBA models are XGBoost .json format (not LGBM like NFL)
MODEL_DIR = PROJECT_ROOT / "models" / "nba"
WANG_LAMBDA = 0.25  # moderate calibration adjustment for NBA

# Market type configuration
MARKET_TYPES = [
    {
        "name": "PTS",
        "model_name": "PTS",
        "series_ticker": "KXNBAPTS",
        "pattern": r"^(.+?):\s*(\d+)\+?\s*points\??$",
        "desc": "points",
        "info_only": False,
    },
    {
        "name": "REB",
        "model_name": "REB",
        "series_ticker": "KXNBAREB",
        "pattern": r"^(.+?):\s*(\d+)\+?\s*rebounds?\??$",
        "desc": "rebounds",
        "info_only": False,
    },
    {
        "name": "AST",
        "model_name": "AST",
        "series_ticker": "KXNBAAST",
        "pattern": r"^(.+?):\s*(\d+)\+?\s*assists?\??$",
        "desc": "assists",
        "info_only": False,
    },
    {
        "name": "BLK",
        "model_name": "BLK",
        "series_ticker": "KXNBABLK",
        "pattern": r"^(.+?):\s*(\d+)\+?\s*blocks?\??$",
        "desc": "blocks",
        "info_only": False,
    },
    {
        "name": "STL",
        "model_name": "STL",
        "series_ticker": "KXNBASTL",
        "pattern": r"^(.+?):\s*(\d+)\+?\s*steals?\??$",
        "desc": "steals",
        "info_only": False,
    },
    {
        "name": "3PT",
        "model_name": "FG3M",
        "series_ticker": "KXNBA3PT",
        "pattern": r"^(.+?):\s*(\d+)\+?\s*threes?\??$",
        "desc": "three-pointers",
        "info_only": False,
    },
    {
        "name": "FTM",
        "model_name": "FTM",
        "series_ticker": "KXNBAFTM",
        "pattern": r"^(.+?):\s*(\d+)\+?\s*free\s*throws?\s*made\??$",
        "desc": "free throws made",
        "info_only": False,
    },
    {
        "name": "PRA",
        "model_name": "PRA",
        "series_ticker": "KXNBAPRA",
        "pattern": r"^(.+?):\s*(\d+)\+?\s*P\+R\+A\??$",
        "desc": "P+R+A",
        "info_only": False,
    },
    {
        "name": "PA",
        "model_name": "PA",
        "series_ticker": "KXNBAPA",
        "pattern": r"^(.+?):\s*(\d+)\+?\s*P\+A\??$",
        "desc": "P+A",
        "info_only": False,
    },
    {
        "name": "PR",
        "model_name": "PR",
        "series_ticker": "KXNBAPR",
        "pattern": r"^(.+?):\s*(\d+)\+?\s*P\+R\??$",
        "desc": "P+R",
        "info_only": False,
    },
    {
        "name": "RA",
        "model_name": "RA",
        "series_ticker": "KXNBARA",
        "pattern": r"^(.+?):\s*(\d+)\+?\s*R\+A\??$",
        "desc": "R+A",
        "info_only": False,
    },
    # Binary outcome markets (no line value) — no trained model for these
    {
        "name": "2D",
        "model_name": None,
        "series_ticker": "KXNBA2D",
        "pattern": r"^(.+?):\s*Double\s*Double\??$",
        "desc": "double double",
        "info_only": True,
    },
    {
        "name": "3D",
        "model_name": None,
        "series_ticker": "KXNBA3D",
        "pattern": r"^(.+?):\s*Triple\s*Double\??$",
        "desc": "triple double",
        "info_only": True,
    },
]


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
    import re as _re
    m = _re.search(r'(\d{2})([A-Z]{3})(\d{2})', ticker)
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


def _parse_kalshi_title(title: str, desc: str):
    """Extract player name and line value from Kalshi title.

    Titles follow format: "Player Name: NN+ stat_desc" or "Player Name: stat_desc"
    Returns (player_name, line_val) or (None, None) on failure.
    """
    if not title or ":" not in title:
        return None, None
    parts = title.split(":", 1)
    player_name = parts[0].strip()
    suffix = parts[1].strip()
    # Extract line value: look for digit pattern before the stat desc
    import re as _re
    m = _re.search(r'(\d+)\s*\+?', suffix)
    line_val = int(m.group(1)) if m else None
    return player_name, line_val


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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--bet", action="store_true")
    args = parser.parse_args()

    client = KalshiClient()
    balance = client.get_balance()
    print(f"Balance: ${balance:.2f}\n")

    # Load features
    latest = load_features()
    if latest is None or latest.empty:
        print("No feature data. Run data pipeline first.")
        return

    if "game_date" in latest.columns:
        latest = latest.sort_values("game_date").groupby("player_id").last().reset_index()
    print(f"Loaded features for {len(latest)} players\n")

    all_opps = []
    model_cache = {}

    for mt in MARKET_TYPES:
        name = mt["name"]
        model_name = mt["model_name"]
        series = mt["series_ticker"]
        desc = mt["desc"]
        info_only = mt.get("info_only", False)

        print(f"Scanning {name} ({series})...", flush=True)

        try:
            mkts = client.list_markets(series_ticker=series, limit=500)
            if mkts is None or mkts.empty:
                print(f"  No markets (off-season?)")
                continue
        except Exception as e:
            print(f"  Cannot reach Kalshi: {e}")
            continue
        print(f"  {len(mkts)} markets", flush=True)

        # Load model
        if model_name not in model_cache:
            m, s, feats, cal = _load_regressor(model_name)
            if m is None and not info_only:
                print(f"  No regressor for {model_name} — skipping")
                continue
            model_cache[model_name] = (m, s, feats, cal)
        reg_model, reg_std, feature_names, beta_cal = model_cache.get(model_name, (None, None, None, None))

        count = 0
        for _, mrow in mkts.iterrows():
            ticker = mrow.get("ticker", "")
            title = str(mrow.get("title", ""))

            # Date filter
            if not _is_current_market(ticker):
                continue

            # Parse title: "Player Name: NN+ stat"
            if ":" not in title:
                continue
            title_parts = title.split(":", 1)
            pname = title_parts[0].strip()
            suffix = title_parts[1].strip()

            # Extract line value
            import re as _re2
            line_m = _re2.search(r'(\d+)', suffix)
            line_val = int(line_m.group(1)) if line_m else 0

            # Bid/ask
            yb_v = mrow.get("yes_bid_dollars", 0)
            ya_v = mrow.get("yes_ask_dollars", 1)
            yb = 0.0 if (isinstance(yb_v, float) and (yb_v != yb_v)) else float(yb_v or 0)
            ya = 1.0 if (isinstance(ya_v, float) and (ya_v != ya_v)) else float(ya_v or 1)
            if yb <= 0 and ya >= 1.0:
                continue
            yes_mid = max(0.01, min(0.99, (yb + ya) / 2.0))

            # Info-only
            if info_only:
                all_opps.append({
                    "type": name, "ticker": ticker, "side": "yes",
                    "price_cents": max(1, int(yes_mid * 100)),
                    "model_prob": 0.5, "market_prob": round(yes_mid, 4),
                    "edge": 0.0, "contracts": 1,
                    "player": pname, "team": "", "line_val": 0,
                    "stat_desc": desc, "label": f"{pname} {desc}",
                })
                count += 1
                continue

            if line_val <= 0:
                continue

            row_match = _match_player(pname, latest)
            if row_match is None:
                continue

            avg_cols = [c for c in row_match.index
                        if c.endswith("_avg_3") and isinstance(row_match[c], (int, float))]
            if avg_cols and all(pd.isna(row_match[c]) for c in avg_cols):
                continue

            try:
                p_yes, mu = _p_ge_line(
                    row_match, reg_model, reg_std, line_val, feature_names,
                    stat_name=name, beta_cal=beta_cal,
                )
            except Exception:
                continue

            yes_edge = p_yes - yes_mid
            all_opps.append({
                "type": name, "ticker": ticker, "side": "yes",
                "price_cents": max(1, int(yes_mid * 100)),
                "model_prob": round(p_yes, 4), "market_prob": round(yes_mid, 4),
                "edge": round(yes_edge, 4), "contracts": 1,
                "player": pname, "team": "", "line_val": line_val,
                "stat_desc": desc, "label": f"{pname} {line_val}+ {desc}",
            })
            count += 1

        label = f"  {name:4s} ({series:10s}): {count} markets matched"
        if info_only:
            label += " (info_only)"
        print(label)

    print(f"\nTotal matched opportunities: {len(all_opps)}")

    # Sort by edge descending
    all_opps.sort(key=lambda x: abs(x.get("edge", 0)), reverse=True)

    if all_opps:
        print(f"\nTop 10:")
        print(f"  {'Type':5s} {'Player':25s} {'Bet':20s} {'Edge':>8s} {'Price':>6s}")
        print(f"  " + "-" * 66)
        for o in all_opps[:10]:
            bt = o.get("type", "?")
            player = o.get("player", "")[:24]
            bet_str = f"{o.get('line_val', 0)}+ {o.get('stat_desc', '')}" if o.get("line_val", 0) else o.get("stat_desc", "")
            edge_str = f"{o.get('edge', 0):+.0%}" if o.get("edge", 0) != 0 else "N/A"
            price = o.get("price_cents", 0)
            print(f"  {bt:5s} {player:25s} {bet_str:20s} {edge_str:>8s} {price:3d}c")

    if args.bet:
        print(f"\n--- PLACING YES ORDERS ---")
        placed = 0
        for o in all_opps:
            if placed >= 6:
                break
            if o.get("info_only", False):
                continue
            edge = o["edge"]
            if edge < 0.05 or o["market_prob"] < 0.10 or o["market_prob"] > 0.80:
                continue

            bid = min(98, max(1, int(o["market_prob"] * 100) + 1))
            b = client.get_balance()
            cost_per = bid / 100.0
            target_risk = b * 0.05
            count = int(target_risk / cost_per)
            if count < 1:
                continue
            try:
                client.create_order(ticker=o["ticker"], side="yes", yes_price=bid, count=str(count))
                print(f"  BUY YES {o['type']:5s} {o['player'][:25]:25s} {o.get('line_val', 0)}+ @ {bid}c x{count} "
                      f"(model={o['model_prob']:.0%} mkt={o['market_prob']:.0%})", flush=True)
                placed += 1
            except Exception as e:
                print(f"  FAILED {o['player']}: {e}", flush=True)
        print(f"  Placed {placed} | Balance: ${client.get_balance():.2f}")

    print(f"\nDone at {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
