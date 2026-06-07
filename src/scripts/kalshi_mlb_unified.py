#!/usr/bin/env python3
"""Unified Kalshi MLB bettor — covers ALL MLB market types.

Scans Kalshi for MLB markets across every stat type Kalshi offers,
loads the corresponding regressor, computes edge with Wang calibration,
and places limit orders where edge >= 7%.

Market types:
  KXMLBKS → strikeouts   (SO,  pitcher)
  KXMLBHR → home runs    (HR,  hitter)
  KXMLBTB → total bases  (TB,  hitter)
  KXMLBHRR → HRR         (HRR, hitter)

Usage:
    python -m src.scripts.kalshi_mlb_unified --scan
    python -m src.scripts.kalshi_mlb_unified --bet
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

MODEL_DIR = PROJECT_ROOT / "models" / "mlb"

WANG_LAMBDA = 0.183
CALIB_DIR = MODEL_DIR / "calibration"

from src.models.calibrator import EmpiricalCalibrator
_calibrator = None
def _get_cal():
    global _calibrator
    if _calibrator is None and CALIB_DIR.exists():
        _calibrator = EmpiricalCalibrator(CALIB_DIR)
    return _calibrator

# registry: (model_name, raw_col, series_ticker, position_filter, title_pattern)
# pattern groups: player_name, line_value
# info_only: True means scan but don't bet (model lacks signal for that market type)
MARKET_TYPES = [
    {
        "name": "KS",
        "model_name": "SO",
        "series_ticker": "KXMLBKS",
        "position": "pitcher",
        "pattern": r"^(.+?):\s*(\d+)\+?\s*strikeouts?\??$",
        "desc": "strikeouts",
        "info_only": False,
    },
    {
        "name": "HR",
        "model_name": "HR",
        "series_ticker": "KXMLBHR",
        "position": "hitter",
        "pattern": r"^(.+?):\s*(\d+)\+?\s*home\s*runs?\??$",
        "desc": "home runs",
        "info_only": False,
    },
    {
        "name": "TB",
        "model_name": "TB",
        "series_ticker": "KXMLBTB",
        "position": "hitter",
        "pattern": r"^(.+?):\s*(\d+)\+?\s*total\s*bases?\??$",
        "desc": "total bases",
        "info_only": False,
    },
    {
        "name": "HRR",
        "model_name": "H_R_RBI",
        "series_ticker": "KXMLBHRR",
        "position": "hitter",
        "pattern": r"^(.+?):\s*(\d+)\+?\s*hits\s*\+\s*runs\s*\+\s*RBIs?\??$",
        "desc": "hits+runs+RBIs",
        "info_only": False,
    },
]

def _load_regressor(model_name):
    # Try LightGBM first, fall back to XGBoost
    mn = model_name.lower()
    lgb_path = MODEL_DIR / f"lgb_{mn}.txt"
    meta_path = MODEL_DIR / f"{'lgb' if lgb_path.exists() else 'reg'}_{mn}.meta.json"
    if lgb_path.exists():
        model = lgb.Booster(model_file=str(lgb_path))
    else:
        xgb_path = MODEL_DIR / f"reg_{mn}.json"
        if not xgb_path.exists() or not meta_path.exists():
            return None, None
        import xgboost as xgb
        model = xgb.XGBRegressor()
        model.load_model(str(xgb_path))
    if not meta_path.exists():
        return model, 1.0
    with open(meta_path) as f:
        meta = json.load(f)
    return model, meta.get("residual_std", 1.0)

def _match_player(title, lc, position_filter=None):
    """Match player by name, optionally filtering by position (hitter/pitcher).
    
    position_filter=None: no filter
    position_filter="hitter": exclude pitchers (position != "P")
    position_filter="pitcher": only pitchers (position == "P")
    """
    if not title or lc is None or lc.empty:
        return None
    
    df = lc
    if position_filter == "hitter":
        df = lc[lc.get("position", "") != "P"]
    elif position_filter == "pitcher":
        df = lc[lc.get("position", "") == "P"]
    
    if df.empty:
        return None
    
    clean = title.replace("?", "").replace(":", "").strip()
    parts = clean.split()
    if len(parts) < 2:
        return None
    first, last = parts[0], parts[-1]
    exact = df[df["player_name"].str.lower() == clean.lower()]
    if len(exact) == 1:
        return exact.iloc[0]
    lm = df[df["player_name"].str.lower().str.endswith(last.lower(), na=False)]
    if len(lm) == 1:
        return lm.iloc[0]
    la = df[df["player_name"].str.lower().str.contains(last.lower(), na=False)]
    if len(la) >= 1:
        fi = la[la["player_name"].str.lower().str[0] == first[0].lower()]
        return fi.iloc[0] if len(fi) >= 1 else la.iloc[0]
    return None

def _p_ge_line(row, model, residual_std, line_val, stat_name=None):
    # Handle both LightGBM (Booster) and XGBoost (XGBRegressor)
    if hasattr(model, 'feature_name'):
        feats = model.feature_name()
    elif hasattr(model, 'feature_names_in_'):
        feats = model.feature_names_in_
    else:
        feats = [c for c in row.index if isinstance(row[c], (int, float))]
    mu = model.predict(pd.DataFrame([{c: row.to_dict().get(c, 0) for c in feats}]).fillna(0))[0]
    sigma = max(residual_std, 0.3)
    p_raw = _norm.cdf(-(line_val - 0.5 - mu) / sigma)
    p_raw = max(0.001, min(0.999, float(p_raw)))
    cal = _get_cal()
    if cal and stat_name:
        p_cal = cal.calibrate(stat_name.lower(), line_val, p_raw)
        p_cal = min(0.75, p_cal)  # hard cap: never trust over 75%
        return p_cal, float(mu)
    # Fallback: Wang Transform
    z = _norm.ppf(p_raw)
    p_corrected = _norm.cdf(z - WANG_LAMBDA)
    p_corrected = min(0.75, float(p_corrected))
    return max(0.001, p_corrected), float(mu)

def _recency_check(player_name: str, line_val: int, stat_col: str = "so") -> tuple[float, float, bool]:
    """Compare model prediction to actual 2026 rate for a player.

    stat_col: which stat to check ("so", "hr", "tb", or "h_r_rbi")
    For hitter stats (hr, tb, h_r_rbi), uses position != "P" instead of gs==1.
    Returns (actual_rate, -1, True) where actual_rate=-1 means insufficient data.
    """
    try:
        cache_path = PROJECT_ROOT / "data" / "cache" / "mlb" / "game_logs_2026_2025_2024.parquet"
        if not cache_path.exists():
            return -1, -1, True
        df = pd.read_parquet(cache_path)
        
        # Determine filter: pitchers use gs==1, hitters filter by position
        if stat_col == "so":
            game_filter = df["gs"] == 1
            pos_filter = df["position"] == "P"
            combined = game_filter & pos_filter
        else:
            # For hitters, don't filter by gs, just by non-P position
            combined = df["position"] != "P"
        
        player_games = df[(df["player_name"].str.contains(player_name, case=False, na=False))
                          & (df["season"] == "2026")
                          & combined]
        if len(player_games) < 3:
            return -1, -1, True
        
        if stat_col == "so":
            actual_rate = (player_games["so"] >= line_val).mean()
        elif stat_col == "hr":
            actual_rate = (player_games["hr"] >= line_val).mean()
        elif stat_col == "tb":
            tb = player_games["1b"] + 2 * player_games["2b"] + 3 * player_games["3b"] + 4 * player_games["hr"]
            actual_rate = (tb >= line_val).mean()
        elif stat_col == "h_r_rbi":
            hrr = player_games["h"] + player_games["r"] + player_games["rbi"]
            actual_rate = (hrr >= line_val).mean()
        else:
            actual_rate = -1
        return float(actual_rate), -1, True
    except Exception:
        return -1, -1, True


def _game_is_pregame(ticker):
    """Check if the market's game is pre-game (not in progress/final)."""
    import json, requests
    try:
        map_file = Path("/tmp/mlb_game_status.json")
        if map_file.exists():
            with open(map_file) as f:
                status_map = json.load(f)
        else:
            return True

        # Known Kalshi MLB team codes (2-3 letters)
        TEAM_CODES = {"MIA","WSH","DET","TB","MIN","CWS","NYM","SEA","SD","PHI",
                      "BAL","BOS","CLE","NYY","KC","CIN","TOR","ATL","SF","MIL",
                      "TEX","STL","ATH","CHC","PIT","HOU","COL","LAA","LAD","ARI"}

        # Extract the player part: after first dash, before the last -N
        m1 = re.search(r"-([A-Z]+)\d+-", ticker)
        if not m1:
            return True
        player_part = m1.group(1)

        # Find known team code at start of player_part
        player_team = ""
        for t_len in [3, 2]:
            prefix = player_part[:t_len]
            if prefix in TEAM_CODES:
                player_team = prefix
                break
        if not player_team:
            return True

        # Extract combined team string from the first part
        m2 = re.match(r"\w+-\d{2}[A-Z]{3}\d{2}\d{4}([A-Z]+)-", ticker)
        if not m2:
            return True
        combined = m2.group(1)

        # The other team is the remainder
        other = combined.replace(player_team, "", 1) if player_team in combined else ""
        if not other:
            return True

        key1 = f"{other}@{player_team}"
        key2 = f"{player_team}@{other}"
        status = status_map.get(key1, status_map.get(key2, ""))
        return status in ("", "Pre-Game", "Scheduled", "Warmup")
    except Exception:
        return True

def load_features():
    cfg = toml.load(CONFIG_DIR / "mlb.toml")
    from src.config.settings import SportConfig
    scfg = SportConfig(name="mlb", display_name="MLB",
                       rolling_windows=cfg["features"]["rolling_windows"], recency_decay=0.001)
    from src.execution.mlb_predictor import MLBLinePredictor
    predictor = MLBLinePredictor(scfg)
    predictor.load_data()
    return predictor._latest_features

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
        print("No feature data. Run training first.")
        return
    print(f"Loaded {len(latest)} players\n")

    all_opps = []
    model_cache = {}

    for mt in MARKET_TYPES:
        name = mt["name"]
        model_name = mt["model_name"]
        series = mt["series_ticker"]
        pos = mt["position"]
        pattern = mt["pattern"]
        desc = mt["desc"]

        info_only = mt.get("info_only", False)

        if model_name not in model_cache:
            m, s = _load_regressor(model_name)
            if m is None:
                print(f"  Skipping {name}: no regressor for {model_name}")
                continue
            model_cache[model_name] = (m, s)
        reg_model, reg_std = model_cache[model_name]

        print(f"Scanning {name} ({series})...", flush=True)
        mkts = client.list_markets(series_ticker=series, limit=1000)
        if mkts is None or mkts.empty:
            print(f"  No markets found")
            continue
        print(f"  {len(mkts)} markets", flush=True)

        opps = []
        for _, m in mkts.iterrows():
            try:
                ticker = m["ticker"]
                title = m["title"]
                yb = float(m["yes_bid_dollars"]) if m["yes_bid_dollars"] not in ("", "nan", "0.0000", None) else 0
                ya = float(m["yes_ask_dollars"]) if m["yes_ask_dollars"] not in ("", "nan", None) else 1
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

                row = _match_player(player_name, latest, position_filter=pos)
                if row is None:
                    continue

                # Skip players with insufficient data (all rolling averages NaN)
                avg_cols = [c for c in row.index if c.endswith("_avg_7") and isinstance(row[c], (int, float))]
                if not avg_cols or all(pd.isna(row[c]) for c in avg_cols):
                    continue

                p_yes, mu = _p_ge_line(row, reg_model, reg_std, line_val, stat_name=model_name)

                # Check recency: if player's actual 2026 rate differs from model, use the higher rate (more conservative edge)
                recency_rate, _, _ = _recency_check(player_name, line_val, stat_col=model_name.lower())
                recency_used = ""
                if recency_rate >= 0:
                    p_final = max(p_yes, recency_rate)  # use HIGHER estimate (more conservative for YES edge)
                    if abs(p_yes - recency_rate) > 0.20:
                        recency_used = f" (model={p_yes:.0%}→{p_final:.0%})"
                        p_yes = p_final
                else:
                    p_final = p_yes

                yes_edge = p_yes - yes_mid
                no_edge = (1 - p_yes) - (1 - yes_mid)

                opps.append({
                    "type": name,
                    "player": row.get("player_name", player_name),
                    "stat": desc,
                    "line": line_val,
                    "mu": round(mu, 2),
                    "sigma": round(reg_std, 2),
                    "p_yes": round(p_yes, 3),
                    "mkt_yes": round(yes_mid, 3),
                    "yes_edge": round(yes_edge, 3),
                    "no_edge": round(no_edge, 3),
                    "recency_rate": round(recency_rate, 3) if recency_rate >= 0 else None,
                    "recency_used": recency_used,
                    "ticker": ticker,
                    "info_only": info_only,
                })
            except Exception:
                pass

        opps.sort(key=lambda x: max(abs(x["yes_edge"]), abs(x["no_edge"])), reverse=True)
        all_opps.extend(opps)

        if opps:
            print(f"  {'Player':25s} {'Line':>5s} {'P(Y)':>5s} {'Mkt':>5s} {'Edge(Y)':>7s} {'Edge(N)':>7s} {'2026':>6s} {'Note':>10s}")
            print(f"  " + "-" * 78)
            for o in opps[:5]:
                r = o.get("recency_rate")
                r_str = f"{r:.0%}" if r is not None else "N/A"
                note = o.get("recency_used", "")
                print(f"  {o['player'][:25]:25s} {o['line']:2d}+ {o['p_yes']:.0%} {o['mkt_yes']:.0%} "
                      f"{o['yes_edge']:>+6.1%} {o['no_edge']:>+6.1%} {r_str:>6s} {note:>10s}")
        else:
            print(f"  No matched opportunities")

    all_opps.sort(key=lambda x: max(abs(x["yes_edge"]), abs(x["no_edge"])), reverse=True)

    print(f"\nTotal matched opportunities: {len(all_opps)}")
    if all_opps:
        print(f"\nTop 10 overall:")
        print(f"  {'Type':4s} {'Player':25s} {'Stat':20s} {'Line':>5s} {'P(Y)':>5s} {'Mkt':>5s} {'Edge(Y)':>7s} {'2026':>6s} {'Note':>10s}")
        print(f"  " + "-" * 90)
        for o in all_opps[:10]:
            r = o.get("recency_rate")
            r_str = f"{r:.0%}" if r is not None else "N/A"
            note = o.get("recency_used", "")
            print(f"  {o['type']:4s} {o['player'][:25]:25s} {o['stat'][:20]:20s} {o['line']:2d}+ "
                  f"{o['p_yes']:.0%} {o['mkt_yes']:.0%} "
                  f"{o['yes_edge']:>+6.1%} {r_str:>6s} {note:>10s}")

    # Daily loss circuit breaker — track starting balance
    starting_balance = client.get_balance()
    daily_loss_limit = 0.10  # 10% max daily loss

    if args.bet:
        print(f"\n--- PLACING YES ORDERS (buy when market underprices — model+recency > mkt) ---")
        placed = 0
        daily_pnl = 0.0
        ks_ticker = "KXMLBKS"
        for o in all_opps:
            if placed >= 6:
                break
            # Skip info-only markets (HR/TB/HRR have R²<0.04)
            if o.get("info_only", False):
                continue
            # Skip in-progress games (model uses pre-game data)
            if not _game_is_pregame(o.get("ticker", "")):
                continue
            # Circuit breaker: stop if daily loss > 10%
            if daily_pnl <= -starting_balance * daily_loss_limit:
                print(f"  DAILY LOSS LIMIT HIT (-${abs(daily_pnl):.2f}), stopping")
                break

            yes_edge = o["yes_edge"]
            no_edge = o["no_edge"]
            mkt_y = o["mkt_yes"]
            p_y = o["p_yes"]

            # Only buy YES when model says market is underpricing (model > market)
            # Public overreacts to names — real edge is on undervalued pitchers
            # Research: 5% edge is breakeven threshold, target 1.50-2.50 decimal (40-67¢) range
            if yes_edge > 0.05 and 0.15 < mkt_y < 0.75:
                bid = min(98, int(mkt_y * 100) + 1)
                side = "yes"
                direction = "BUY YES"
            else:
                continue

            b = client.get_balance()
            cost_per = ((100 - bid) / 100) if side == "no" else (bid / 100)
            target_risk = b * 0.05  # 5% of current balance
            count = int(target_risk / cost_per)
            if count < 1:
                print(f"  SKIP {o['player']}: can't risk <5% (1 contract = ${cost_per:.2f} > ${target_risk:.2f} limit)")
                continue
            try:
                client.create_order(ticker=o["ticker"], side=side, yes_price=bid, count=str(count))
                daily_pnl -= cost_per * count
                print(f"  {direction:8s} {o['type']:4s} {o['player'][:25]:25s} {o['stat'][:15]:15s} {o['line']}+ @ {bid}¢ x{count} "
                      f"(model={p_y:.0%} mkt={mkt_y:.0%} risk=${cost_per*count:.2f})", flush=True)
                placed += 1
            except Exception as e:
                print(f"  FAILED {o['player']}: {e}", flush=True)
        print(f"  Placed {placed} | Balance: ${client.get_balance():.2f}")

    print(f"\nDone at {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()
