#!/usr/bin/env python3
"""Unified Kalshi NFL bettor - ready for 2026 season.

Scans Kalshi for NFL player prop markets, loads the corresponding LGBM regressor,
computes edge with calibration, and places orders where edge >= 7%.

Market types (to be confirmed with Kalshi NFL series when available):
  KXNFLPASSYDS -> passing yards
  KXNFLRUSHYDS -> rushing yards  
  KXNFLRECYDS  -> receiving yards
  KXNFLTD      -> touchdowns
  KXNFLREC     -> receptions
  KXNFLPASSTD  -> passing TDs
  KXNFLINT     -> interceptions

Usage:
    python -m src.scripts.kalshi_nfl_unified --scan
    python -m src.scripts.kalshi_nfl_unified --bet
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
import toml, lightgbm as lgb
from scipy.stats import norm as _norm

MODEL_DIR = PROJECT_ROOT / "models" / "nfl"
WANG_LAMBDA = 0.183

# Market type configuration
# model_name maps to training STAT_TARGETS
# series_ticker is Kalshi's series ticker for each market
# When NFL markets launch on Kalshi, update these series_ticker values
MARKET_TYPES = [
    {
        "name": "PASS_YDS",
        "model_name": "PASS_YDS",
        "series_ticker": "KXNFLPASSYDS",
        "pattern": r"^(.+?):\s*(\d+)\+?\s*passing\s*yards?\?$",
        "desc": "passing yards",
        "info_only": False,
    },
    {
        "name": "PASS_TD",
        "model_name": "PASS_TD",
        "series_ticker": "KXNFLPASSTD",
        "pattern": r"^(.+?):\s*(\d+)\+?\s*passing\s*TDs?\?$",
        "desc": "passing TDs",
        "info_only": False,
    },
    {
        "name": "RUSH_YDS",
        "model_name": "RUSH_YDS",
        "series_ticker": "KXNFLRUSHYDS",
        "pattern": r"^(.+?):\s*(\d+)\+?\s*rushing\s*yards?\?$",
        "desc": "rushing yards",
        "info_only": False,
    },
    {
        "name": "REC_YDS",
        "model_name": "REC_YDS",
        "series_ticker": "KXNFLRECYDS",
        "pattern": r"^(.+?):\s*(\d+)\+?\s*receiving\s*yards?\?$",
        "desc": "receiving yards",
        "info_only": False,
    },
    {
        "name": "REC",
        "model_name": "REC",
        "series_ticker": "KXNFLREC",
        "pattern": r"^(.+?):\s*(\d+)\+?\s*receptions?\?$",
        "desc": "receptions",
        "info_only": False,
    },
    {
        "name": "TD",
        "model_name": "TD",
        "series_ticker": "KXNFLTD",
        "pattern": r"^(.+?):\s*(\d+)\+?\s*touchdowns?\?$",
        "desc": "touchdowns",
        "info_only": False,
    },
    {
        "name": "INT",
        "model_name": "INT",
        "series_ticker": "KXNFLINT",
        "pattern": r"^(.+?):\s*(\d+)\+?\s*interceptions?\?$",
        "desc": "interceptions",
        "info_only": True,
    },
]


def _load_regressor(model_name):
    """Load LGBM model and residual_std from saved files."""
    mn = model_name.lower()
    model_path = MODEL_DIR / f"lgb_{mn}.txt"
    meta_path = MODEL_DIR / f"lgb_{mn}.meta.json"
    std_path = MODEL_DIR / f"lgb_{mn}.std.json"
    
    if not model_path.exists():
        # Fall back to old XGBoost
        xgb_path = MODEL_DIR / f"{mn}.json"
        if xgb_path.exists():
            import xgboost as xgb
            model = xgb.XGBRegressor()
            model.load_model(str(xgb_path))
            std = 1.0
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                std = meta.get("residual_std", 1.0)
            # Get feature names from old model
            if hasattr(model, 'feature_names_in_'):
                return model, std, list(model.feature_names_in_)
            else:
                # Try loading old JSON model for feature names
                xgb_json_path = MODEL_DIR / f"{mn}.json"
                if xgb_json_path.exists():
                    import json as _json
                    with open(xgb_json_path) as f:
                        mdata = _json.load(f)
                    # XGBoost JSON format has feature_names
                    feat_names = mdata.get('learner', {}).get('feature_names', [])
                    if feat_names:
                        return model, std, feat_names
                return model, std, []
        return None, None, []
    
    # LGBM model
    model = lgb.Booster(model_file=str(model_path))
    true_features = model.feature_name()
    std = 1.0
    if std_path.exists():
        with open(std_path) as f:
            std = json.load(f).get("residual_std", 1.0)
    elif meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        std = meta.get("residual_std", 1.0)
    return model, float(std), true_features


def _match_player(title: str, latest: pd.DataFrame, position_filter: str = None) -> pd.Series:
    """Match player name from Kalshi title to our feature data.
    
    position_filter: "QB", "RB", "WR", "TE", or None for any position.
    """
    if not title or latest is None or latest.empty:
        return None
    
    # Apply position filter
    df = latest
    if position_filter and "position" in latest.columns:
        df = latest[latest["position"].str.upper() == position_filter.upper()]
        if df.empty:
            # Try fuzzy position match (some data has position_group)
            if "position_group" in latest.columns:
                df = latest[latest["position_group"].str.upper() == position_filter.upper()]
            if df.empty:
                return None
    
    clean = title.replace("?", "").replace(":", "").strip()
    parts = clean.split()
    if len(parts) < 2:
        return None
    first, last = parts[0], parts[-1]
    
    # Try exact match
    if "player_name" in df.columns:
        exact = df[df["player_name"].str.lower() == clean.lower()]
        if len(exact) >= 1:
            return exact.iloc[-1]
    
    # Try last name
    if "player_name" in df.columns:
        lm = df[df["player_name"].str.lower().str.contains(last.lower(), na=False)]
        if len(lm) >= 1:
            # Prefer first+last initial match
            fi = lm[lm["player_name"].str.lower().str[0] == first[0].lower()]
            if len(fi) >= 1:
                return fi.iloc[-1]
            return lm.iloc[-1]
    
    # Try player_display_name
    if "player_display_name" in df.columns:
        lm = df[df["player_display_name"].str.lower().str.contains(last.lower(), na=False)]
        if len(lm) >= 1:
            fi = lm[lm["player_display_name"].str.lower().str[0] == first[0].lower()]
            if len(fi) >= 1:
                return fi.iloc[-1]
            return lm.iloc[-1]
    
    return None


def _p_ge_line(row, model, residual_std, line_val, true_features):
    """Compute P(stat >= line_val) using model prediction + normal residual."""
    # Build feature vector matching what the model expects
    feat_dict = {}
    for c in true_features:
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
    p_raw = _norm.cdf(-(line_val - 0.5 - mu) / sigma)
    p_raw = max(0.001, min(0.999, float(p_raw)))
    
    # Wang Transform for calibration
    z = _norm.ppf(p_raw)
    p_corrected = _norm.cdf(z - WANG_LAMBDA)
    p_corrected = min(0.75, float(p_corrected))
    return max(0.001, p_corrected), float(mu)


def load_features():
    """Load and build NFL features using the same pipeline as training."""
    from src.config.settings import SportConfig
    from src.features.nfl import NFLFeatureEngineer
    
    cache_path = PROJECT_ROOT / "data" / "nfl_cache" / "weekly.parquet"
    if not cache_path.exists():
        print("No cached NFL data.")
        return None
    
    cfg_path = CONFIG_DIR / "nfl.toml"
    if cfg_path.exists():
        cfg = toml.load(cfg_path)
    else:
        cfg = {"features": {"rolling_windows": [3, 5, 7], "recency_decay": 0.001}}
    
    scfg = SportConfig(
        name="nfl", display_name="NFL",
        rolling_windows=cfg["features"]["rolling_windows"],
        recency_decay=cfg["features"].get("recency_decay", 0.001),
    )
    fe = NFLFeatureEngineer(scfg)
    all_games = pd.read_parquet(cache_path)
    
    if "player_name" not in all_games.columns and "player_display_name" in all_games.columns:
        all_games["player_name"] = all_games["player_display_name"]
    
    featured = fe.build_features(all_games)
    return featured


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--bet", action="store_true")
    args = parser.parse_args()

    client = KalshiClient()
    print(f"Balance: ${client.get_balance():.2f}\n")

    # Load features
    latest = load_features()
    if latest is None or latest.empty:
        print("No feature data. Run training pipeline first.")
        return
    
    # Get latest week for each player (most recent game data)
    if "game_date" in latest.columns:
        latest = latest.sort_values("game_date").groupby("player_id").last().reset_index()
    print(f"Loaded features for {len(latest)} players\n")

    all_opps = []
    model_cache = {}

    for mt in MARKET_TYPES:
        name = mt["name"]
        model_name = mt["model_name"]
        series = mt["series_ticker"]
        pattern = mt["pattern"]
        desc = mt["desc"]
        info_only = mt.get("info_only", False)

        if model_name not in model_cache:
            m, s, feats = _load_regressor(model_name)
            if m is None:
                print(f"  Skipping {name}: no regressor for {model_name}")
                continue
            model_cache[model_name] = (m, s, feats)
        reg_model, reg_std, true_features = model_cache[model_name]

        print(f"Scanning {name} ({series})...", flush=True)
        try:
            mkts = client.list_markets(series_ticker=series, limit=1000)
            if mkts is None or mkts.empty:
                print(f"  No markets found (off-season: markets appear when season starts)")
                continue
        except Exception as e:
            print(f"  Cannot scan {series}: {e}")
            continue
        
        print(f"  {len(mkts)} markets", flush=True)

        opps = []
        for _, m in mkts.iterrows():
            try:
                ticker = m["ticker"]
                title = m["title"]
                yb = float(m["yes_bid_dollars"]) if m.get("yes_bid_dollars") not in ("", "nan", "0.0000", None) else 0
                ya = float(m["yes_ask_dollars"]) if m.get("yes_ask_dollars") not in ("", "nan", None) else 1
                if yb <= 0 and ya >= 1.0:
                    continue
                if yb <= 0 and ya <= 0:
                    continue
                yes_mid = max(0.01, min(0.99, (yb + ya) / 2.0))

                line_match = re.match(pattern, title, re.IGNORECASE)
                if not line_match:
                    continue
                player_name = line_match.group(1).strip()
                line_val = int(line_match.group(2))
                if line_val <= 0:
                    continue

                row = _match_player(player_name, latest, position_filter=mt.get("position"))
                if row is None:
                    continue

                # Skip if all rolling features are NaN (no game history)
                avg_cols = [c for c in row.index if c.endswith("_avg_7") and isinstance(row[c], (int, float))]
                if avg_cols and all(pd.isna(row[c]) for c in avg_cols):
                    continue

                p_yes, mu = _p_ge_line(row, reg_model, reg_std, line_val, true_features)
                yes_edge = p_yes - yes_mid
                no_edge = (1 - p_yes) - (1 - yes_mid)

                opps.append({
                    "type": name,
                    "player": str(row.get("player_name", row.get("player_display_name", player_name))),
                    "stat": desc,
                    "line": line_val,
                    "mu": round(mu, 2),
                    "sigma": round(reg_std, 2),
                    "p_yes": round(p_yes, 3),
                    "mkt_yes": round(yes_mid, 3),
                    "yes_edge": round(yes_edge, 3),
                    "no_edge": round(no_edge, 3),
                    "ticker": ticker,
                    "info_only": info_only,
                })
            except Exception:
                pass

        opps.sort(key=lambda x: max(abs(x["yes_edge"]), abs(x["no_edge"])), reverse=True)
        all_opps.extend(opps)

        if opps:
            print(f"  {'Player':25s} {'Line':>5s} {'P(Y)':>5s} {'Mkt':>5s} {'Edge(Y)':>7s} {'Edge(N)':>7s}")
            print(f"  " + "-" * 60)
            for o in opps[:5]:
                print(f"  {o['player'][:25]:25s} {o['line']:5d}+ {o['p_yes']:.0%} {o['mkt_yes']:.0%} "
                      f"{o['yes_edge']:>+6.1%} {o['no_edge']:>+6.1%}")
        else:
            print(f"  No matched opportunities")

    all_opps.sort(key=lambda x: max(abs(x["yes_edge"]), abs(x["no_edge"])), reverse=True)

    print(f"\nTotal matched opportunities: {len(all_opps)}")
    if all_opps:
        print(f"\nTop 10 overall:")
        print(f"  {'Type':8s} {'Player':25s} {'Stat':20s} {'Line':>5s} {'P(Y)':>5s} {'Mkt':>5s} {'Edge(Y)':>7s}")
        print(f"  " + "-" * 80)
        for o in all_opps[:10]:
            print(f"  {o['type']:8s} {o['player'][:25]:25s} {o['stat'][:20]:20s} {o['line']:5d}+ "
                  f"{o['p_yes']:.0%} {o['mkt_yes']:.0%} {o['yes_edge']:>+6.1%}")

    if args.bet:
        print(f"\n--- PLACING YES ORDERS ---")
        placed = 0
        starting_balance = client.get_balance()
        for o in all_opps:
            if placed >= 6:
                break
            if o.get("info_only", False):
                continue

            yes_edge = o["yes_edge"]
            mkt_y = o["mkt_yes"]
            p_y = o["p_yes"]

            if yes_edge > 0.05 and 0.15 < mkt_y < 0.75:
                bid = min(98, int(mkt_y * 100) + 1)
                side = "yes"
                direction = "BUY YES"
            else:
                continue

            b = client.get_balance()
            cost_per = bid / 100.0
            target_risk = b * 0.05
            count = int(target_risk / cost_per)
            if count < 1:
                continue
            try:
                client.create_order(ticker=o["ticker"], side=side, yes_price=bid, count=str(count))
                print(f"  {direction:8s} {o['type']:8s} {o['player'][:25]:25s} {o['line']}+ @ {bid}c x{count} "
                      f"(model={p_y:.0%} mkt={mkt_y:.0%})", flush=True)
                placed += 1
            except Exception as e:
                print(f"  FAILED {o['player']}: {e}", flush=True)
        print(f"  Placed {placed} | Balance: ${client.get_balance():.2f}")

    print(f"\nDone at {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
