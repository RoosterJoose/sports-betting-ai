#!/usr/bin/env python3
"""Unified Kalshi WNBA bettor — covers WNBA player prop markets.

Now using player-level data (PlayerGameLogs API, 11,615 rows across 3 seasons).
Models retrained June 8, 2026 with R² 0.21-0.53.

Market types:
  KXWNBAPTS  → points
  KXWNBAREB  → rebounds
  KXWNBAAST  → assists
  KXWNBA3PT  → three-pointers made
  KXWNBABLK  → blocks (no active markets)
  KXWNBASTL  → steals (no active markets)
  KXWNBAPRA  → P+R+A (no active markets)
  KXWNBATOTAL → team total points

Usage:
    python -m src.scripts.kalshi_wnba_unified --scan
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

MODEL_DIR = PROJECT_ROOT / "models" / "wnba"
WANG_LAMBDA = 0.25

# Market type configuration (player-level models now available)
MARKET_TYPES = [
    {"name": "PTS", "model_name": "PTS", "series_ticker": "KXWNBAPTS",
     "pattern": r"^(.+?):\s*(\d+)\+?\s*points\??$", "desc": "points", "info_only": False},
    {"name": "REB", "model_name": "REB", "series_ticker": "KXWNBAREB",
     "pattern": r"^(.+?):\s*(\d+)\+?\s*rebounds?\??$", "desc": "rebounds", "info_only": False},
    {"name": "AST", "model_name": "AST", "series_ticker": "KXWNBAAST",
     "pattern": r"^(.+?):\s*(\d+)\+?\s*assists?\??$", "desc": "assists", "info_only": False},
    {"name": "3PT", "model_name": "FG3M", "series_ticker": "KXWNBA3PT",
     "pattern": r"^(.+?):\s*(\d+)\+?\s*threes?\??$", "desc": "three-pointers", "info_only": False},
    {"name": "BLK", "model_name": "BLK", "series_ticker": "KXWNBABLK",
     "pattern": r"^(.+?):\s*(\d+)\+?\s*blocks?\??$", "desc": "blocks", "info_only": False},
    {"name": "STL", "model_name": "STL", "series_ticker": "KXWNBASTL",
     "pattern": r"^(.+?):\s*(\d+)\+?\s*steals?\??$", "desc": "steals", "info_only": True},
    {"name": "PRA", "model_name": "PRA", "series_ticker": "KXWNBAPRA",
     "pattern": r"^(.+?):\s*(\d+)\+?\s*P\+R\+A\??$", "desc": "P+R+A", "info_only": False},
    {"name": "TOTAL", "model_name": None, "series_ticker": "KXWNBATOTAL",
     "pattern": r"^(.+?):\s*(\d+)\+?\s*points\??$", "desc": "team total", "info_only": True},
]


def _load_regressor(model_name: str):
    if model_name is None:
        return None, None, None, BetaCalibrator()
    mn = model_name.lower()
    model_path = MODEL_DIR / f"{mn}.json"
    meta_path = MODEL_DIR / f"{mn}.metrics.json"
    if not model_path.exists():
        return None, None, None, BetaCalibrator()
    import xgboost as xgb
    model = xgb.XGBRegressor()
    model.load_model(str(model_path))
    try:
        with open(model_path) as f:
            mdata = json.load(f)
        feature_names = mdata.get('learner', {}).get('feature_names', [])
    except Exception:
        feature_names = []
    std = 1.0
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        std = meta.get("residual_std", meta.get("mae", 1.0))
    cal_path = MODEL_DIR / f"{mn}_beta_cal.json"
    beta_cal = BetaCalibrator.load(cal_path)
    return model, float(std), feature_names, beta_cal


def _match_player(title: str, latest: pd.DataFrame) -> pd.Series:
    """Match player name from Kalshi title to WNBA feature data.

    Now works with player-level data (PlayerGameLogs API provides player_name).
    """
    if not title or latest is None or latest.empty:
        return None
    clean = title.replace("?", "").replace(":", "").strip()
    parts = clean.split()
    if len(parts) < 2:
        return None
    first, last = parts[0], parts[-1]
    # Try exact match on team_name or team_abbreviation
    if "team_name" in latest.columns:
        exact = latest[latest["team_name"].str.lower() == clean.lower()]
        if len(exact) >= 1:
            return exact.iloc[-1]
    if "team_abbreviation" in latest.columns:
        exact = latest[latest["team_abbreviation"].str.lower() == clean.lower()]
        if len(exact) >= 1:
            return exact.iloc[-1]
    # Try player_name if it exists (for player-level data)
    if "player_name" in latest.columns:
        exact = latest[latest["player_name"].str.lower() == clean.lower()]
        if len(exact) == 1:
            return exact.iloc[0]
        if len(exact) > 1:
            return exact.iloc[0]  # return first if multiple (same player, different dates)
        lm = latest[latest["player_name"].str.lower().str.contains(last.lower(), na=False)]
        if len(lm) == 1:
            return lm.iloc[0]
        if len(lm) > 1:
            fi = lm[lm["player_name"].str.lower().str[0] == first[0].lower()]
            if len(fi) == 1:
                return fi.iloc[0]
            elif len(fi) > 1:
                return fi.iloc[0]
            return lm.iloc[0]
    return None


def _p_ge_line(row, model, residual_std, line_val, feature_names, stat_name="", beta_cal=None):
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
    p_raw = p_ge_stat(stat_name, mu, sigma, line_val)
    if beta_cal is not None and beta_cal._fitted:
        p_corrected = beta_cal(p_raw)
    else:
        z = _norm.ppf(p_raw)
        p_corrected = _norm.cdf(z - WANG_LAMBDA)
    p_corrected = min(0.999, float(p_corrected))
    return max(0.001, p_corrected), float(mu)


def load_features():
    """Load WNBA player-level data and build features.

    Uses PlayerGameLogs API data (player-level game logs with player_name).
    """
    from src.features.wnba import WNBAFeatureEngineer
    from src.config.settings import SportConfig

    cfg_path = CONFIG_DIR / "wnba.toml"
    if cfg_path.exists():
        cfg = toml.load(cfg_path)
    else:
        cfg = {"features": {"rolling_windows": [3, 5, 10], "recency_decay": 0.001}}

    scfg = SportConfig(
        name="wnba", display_name="WNBA",
        rolling_windows=cfg["features"]["rolling_windows"],
        recency_decay=cfg["features"].get("recency_decay", 0.001),
    )
    fe = WNBAFeatureEngineer(scfg)

    cache_path = PROJECT_ROOT / "data" / "wnba_cache" / "wnba_games.parquet"
    if not cache_path.exists():
        print("  No cached WNBA data. Run data pipeline first.")
        return None

    all_games = pd.read_parquet(cache_path)
    print(f"  Loaded {len(all_games)} raw rows", flush=True)

    featured = fe.build_features(all_games)
    print(f"  Feature engineering: {len(featured)} rows, {len(featured.columns)} cols", flush=True)

    # Merge team info and player_name back (feature engineer strips them)
    merge_cols_src = ["player_id", "game_date"]
    merge_cols_extra = ["team_name", "team_abbreviation"]
    if "player_name" in all_games.columns:
        merge_cols_extra.append("player_name")
    
    if all(c in all_games.columns for c in merge_cols_src + ["team_name"]):
        merge_df = all_games[merge_cols_src + merge_cols_extra].drop_duplicates(subset=merge_cols_src)
        merge_df["game_date"] = pd.to_datetime(merge_df["game_date"])
        featured["game_date"] = pd.to_datetime(featured["game_date"])
        featured = featured.merge(merge_df, on=merge_cols_src, how="left")

    return featured


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--bet", action="store_true")
    args = parser.parse_args()

    client = KalshiClient()
    print(f"Balance: ${client.get_balance():.2f}\n")

    latest = load_features()
    if latest is None or latest.empty:
        print("No feature data. Run data pipeline first.")
        return

    if "game_date" in latest.columns:
        latest = latest.sort_values("game_date").groupby("player_id").last().reset_index()
    print(f"Loaded features for {len(latest)} teams/players\n")

    all_opps = []
    model_cache = {}

    for mt in MARKET_TYPES:
        name = mt["name"]
        model_name = mt["model_name"]
        series = mt["series_ticker"]
        pattern = mt["pattern"]
        desc = mt["desc"]

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

        if model_name not in model_cache:
            m, s, feats, cal = _load_regressor(model_name)
            model_cache[model_name] = (m, s, feats, cal)
        reg_model, reg_std, feature_names, beta_cal = model_cache.get(model_name, (None, None, None, None))

        count = 0
        info_only = mt.get("info_only", True)
        for _, mrow in mkts.iterrows():
            try:
                ticker = mrow["ticker"]
                title = mrow.get("title", "")
                yb_v = mrow.get("yes_bid_dollars", 0)
                ya_v = mrow.get("yes_ask_dollars", 1)
                yb = 0.0 if (isinstance(yb_v, float) and (yb_v != yb_v)) else float(yb_v or 0)
                ya = 1.0 if (isinstance(ya_v, float) and (ya_v != ya_v)) else float(ya_v or 1)
                if yb <= 0 and ya >= 1.0:
                    continue
                yes_mid = max(0.01, min(0.99, (yb + ya) / 2.0))

                lm = re.match(pattern, title, re.IGNORECASE)
                if not lm:
                    continue
                pname = lm.group(1).strip()
                line_val = int(lm.group(2)) if len(lm.groups()) >= 2 else 0
                if line_val <= 0:
                    continue

                # Match player in feature data
                row = _match_player(pname, latest)
                if row is None and not info_only:
                    continue

                # Predict probability if model available
                if reg_model is not None and row is not None:
                    try:
                        p_yes, mu = _p_ge_line(row, reg_model, reg_std, line_val,
                                              feature_names, stat_name=model_name or "", beta_cal=beta_cal)
                    except Exception:
                        p_yes = 0.5
                        mu = 0
                else:
                    p_yes = 0.5
                    mu = 0

                yes_edge = p_yes - yes_mid
                no_edge = (1 - p_yes) - (1 - yes_mid)

                label = f"{pname} {line_val}+ {desc}"
                all_opps.append({
                    "type": name, "ticker": ticker,
                    "side": "yes",
                    "price_cents": max(1, int(yes_mid * 100)),
                    "model_prob": round(p_yes, 4),
                    "market_prob": round(yes_mid, 4),
                    "edge": round(max(yes_edge, no_edge), 4),
                    "yes_edge": round(yes_edge, 4),
                    "no_edge": round(no_edge, 4),
                    "contracts": 1,
                    "player": pname,
                    "team": "",
                    "line_val": line_val,
                    "stat_desc": desc,
                    "label": label,
                    "info_only": info_only,
                })
                count += 1
            except Exception:
                pass

        tag = "info_only" if info_only else "active"
        print(f"  {name:4s} ({series:11s}): {count} markets ({tag})")

    print(f"\nTotal markets found: {len(all_opps)}")
    if all_opps:
        active_count = sum(1 for o in all_opps if not o.get("info_only", True))
        info_count = len(all_opps) - active_count
        print(f"  Active models: {active_count}, info_only: {info_count}")
    if all_opps:
        print(f"\nTop 10:")
        print(f"  {'Type':5s} {'Player':25s} {'Bet':20s} {'Price':>6s}")
        print(f"  " + "-" * 58)
        for o in all_opps[:10]:
            bt = o.get("type", "?")
            player = o.get("player", "")[:24]
            bet_str = f"{o.get('line_val', 0)}+ {o.get('stat_desc', '')}"
            print(f"  {bt:5s} {player:25s} {bet_str:20s} {o.get('price_cents', 0):3d}c")

    if args.bet:
        print("\n  No model-based betting available for WNBA (team-level models only)")

    print(f"\nDone at {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
