#!/usr/bin/env python3
"""Coordinated morning scan across all Kalshi sports.

Scans KS, F5, Safe Compounder markets, computes edges using trained models,
finds best 2/3/4-leg parlays, and displays ranked top plays with team info.

Usage:
    python -m src.scripts.morning_scan             # dry run
    python -m src.scripts.morning_scan --bet        # place orders
"""
import os, sys, json, warnings, time, re
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from src.data.kalshi import KalshiClient
from src.execution.risk import RiskManager
from src.utils.trade_tracker import TradeTracker
from src.execution.kalshi_parlay import KalshiParlayFinder, display_parlays

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# MLB team code -> full name mapping
MLB_TEAMS = {
    "ARI": "Arizona D-backs", "ATL": "Atlanta Braves", "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox", "CHC": "Chicago Cubs", "CIN": "Cincinnati Reds",
    "CLE": "Cleveland Guardians", "COL": "Colorado Rockies", "CWS": "Chicago White Sox",
    "DET": "Detroit Tigers", "HOU": "Houston Astros", "KC": "Kansas City Royals",
    "LAA": "LA Angels", "LAD": "LA Dodgers", "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers", "MIN": "Minnesota Twins", "NYM": "NY Mets",
    "NYY": "NY Yankees", "OAK": "Athletics", "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates", "SD": "San Diego Padres", "SEA": "Seattle Mariners",
    "SF": "San Francisco Giants", "STL": "St. Louis Cardinals", "TB": "Tampa Bay Rays",
    "TEX": "Texas Rangers", "TOR": "Toronto Blue Jays", "WSH": "Washington Nationals",
    "WSN": "Washington Nationals", "AZ": "Arizona D-backs",
}

KC_TEAM_CODES = {"MIA","WSH","DET","TB","MIN","CWS","NYM","SEA","SD","PHI",
                  "BAL","BOS","CLE","NYY","KC","CIN","TOR","ATL","SF","MIL",
                  "TEX","STL","OAK","CHC","PIT","HOU","COL","LAA","LAD","ARI"}


def get_team_from_ticker(ticker):
    """Extract team abbreviation and full name from Kalshi ticker."""
    parts = ticker.split("-")
    if len(parts) < 3:
        return "", ""
    # Player code: e.g. BALSBAZ34 -> first 3 chars = team
    player_part = parts[2]
    for t_len in [3, 2]:
        code = player_part[:t_len]
        if code in KC_TEAM_CODES:
            return code, MLB_TEAMS.get(code, code)
    return "", ""


def extract_team_key(ticker):
    """Extract game key from ticker for deduplication."""
    parts = ticker.split("-")
    if len(parts) >= 3:
        return parts[1]  # game key
    return ""


def verify_and_cancel(ticker, model_prob, order_price, side, order_id=None, min_edge=0.05):
    kc = KalshiClient()
    time.sleep(1.5)
    mkts = kc.list_markets(ticker_prefix=ticker, limit=10)
    if mkts is None or mkts.empty:
        return True
    row = mkts.iloc[0]
    yb = float(row.get("yes_bid", 0) or 0)
    ya = float(row.get("yes_ask", 0) or 0)
    fair_prob = yb + (ya - yb) / 2 if ya > 0 else yb
    current_edge = model_prob - fair_prob
    if current_edge < min_edge:
        print(f"  X Edge dropped to {current_edge:.1%} (needs {min_edge:.1%}) - cancelling")
        if order_id:
            kc.cancel_order(order_id)
            return False
        orders = kc._request("GET", "/portfolio/orders").get("orders", [])
        for o in orders:
            if o.get("ticker") == ticker and o.get("status") == "resting":
                kc.cancel_order(o.get("order_id", ""))
                return False
        return False
    print(f"  V Verified: edge={current_edge:.1%}")
    return True


def place_with_verify(kc, ticker, side, price_cents, contracts, model_prob):
    print(f"/  {ticker} {side.upper()} {price_cents}c x {contracts} "
          f"(model={model_prob:.0%}, edge pending verify)")
    try:
        resp = kc.create_order(ticker=ticker, side=side,
                               yes_price=price_cents, count=str(contracts))
    except Exception as e:
        print(f"  X Order failed: {e}")
        return False
    if not resp or not resp.get("order_id"):
        print(f"  X Order rejected: {resp}")
        return False
    oid = resp["order_id"]
    fill_count = float(resp.get("fill_count", "0"))
    remaining_count = float(resp.get("remaining_count", "0"))
    if fill_count > 0 and remaining_count == 0:
        print(f"  V Filled (${float(resp.get('average_fill_price', '0')):.2f} x {fill_count:.0f})")
        return True
    elif fill_count == 0 and remaining_count > 0:
        print(f"  V Resting (order_id={oid[:8]}...)")
        return verify_and_cancel(ticker, model_prob, price_cents, side, order_id=oid)
    else:
        print(f"  V Partial fill ({fill_count:.0f} filled, {remaining_count:.0f} remaining)")
        return True


def _load_dotenv():
    """Load .env file so env vars are available (e.g., BETTING_ENABLED)."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v)


def morning_scan(bankroll=None, auto_bet=False, min_edge=0.05):
    # ── Safety gate: BETTING_ENABLED must be explicitly 'true' to place live orders ──
    _load_dotenv()
    betting_enabled = os.environ.get("BETTING_ENABLED", "").strip().lower() == "true"
    if auto_bet and not betting_enabled:
        print()
        print("  " + "!" * 66)
        print("  !!! SAFETY: --bet passed but BETTING_ENABLED is not 'true' in .env")
        print("  !!! To enable live betting: add 'BETTING_ENABLED=true' to your .env file")
        print("  !!! Defaulting to DRY RUN for now")
        print("  " + "!" * 66)
        print()
        auto_bet = False

    print("=" * 70)
    print(f"  MORNING SCAN - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    kc = KalshiClient()
    balance = bankroll or kc.get_balance()
    risk = RiskManager(bankroll=balance)
    print(f"  Balance: ${balance:.2f}  |  Min edge: {min_edge:.0%}")

    all_bets = []

    # === 1. MLB PLAYER PROPS (KS = strikeouts, HR = home runs, TB = total bases, HRR = H+R+RBI) ===
    print()
    print("  " + "-" * 66)
    print("  1. MLB PLAYER PROPS (KS/strikeouts, HR, TB, HRR)")
    print("  " + "-" * 66)
    try:
        from src.scripts.kalshi_mlb_unified import load_features, MARKET_TYPES, _load_regressor, _match_player, _p_ge_line, _recency_check
        latest = load_features()
        if latest is not None and not latest.empty:
            print(f"  Loaded {len(latest)} players")
            today = datetime.now().strftime("%y%b%d").upper()
            for mt in MARKET_TYPES:
                if mt.get("info_only", True):
                    continue
                mname = mt["name"]
                series = mt["series_ticker"]
                pattern = mt["pattern"]
                pos = mt["position"]
                model_name = mt["model_name"]
                desc = mt["desc"]
                try:
                    mkts = kc.list_markets(series_ticker=series, limit=500)
                    if mkts is None or mkts.empty:
                        continue
                    mkts = mkts[mkts["ticker"].str.contains(today, regex=False, na=False)]
                    if mkts.empty:
                        continue
                except Exception:
                    continue
                m, s = _load_regressor(model_name)
                if m is None:
                    continue
                count = 0
                for _, row in mkts.iterrows():
                    try:
                        ticker = row["ticker"]
                        title = row.get("title", "")
                        yb_v = row.get("yes_bid_dollars", 0)
                        ya_v = row.get("yes_ask_dollars", 1)
                        yb = 0.0 if (isinstance(yb_v, float) and (yb_v != yb_v)) else float(yb_v or 0)
                        ya = 1.0 if (isinstance(ya_v, float) and (ya_v != ya_v)) else float(ya_v or 1)
                        if yb <= 0 and ya >= 1.0:
                            continue
                        yes_mid = max(0.01, min(0.99, (yb + ya) / 2.0))
                        lm = re.match(pattern, title, re.IGNORECASE)
                        if not lm:
                            continue
                        pname = lm.group(1).strip()
                        line_val = int(lm.group(2))
                        prow = _match_player(pname, latest, position_filter=pos)
                        if prow is None:
                            continue
                        # Use empirical calibration (pass stat_name for calibration lookup)
                        stat_col = model_name.lower()
                        p_yes, mu = _p_ge_line(prow, m, s, line_val, stat_name=model_name)
                        # Cross-reference with actual 2026 rate (conservative for YES edge)
                        recency_rate, _, _ = _recency_check(pname, line_val, stat_col=stat_col)
                        if recency_rate >= 0:
                            # Only override when recency differs meaningfully (>10%),
                            # matching kalshi_mlb_unified.py standalone script behavior.
                            # Use MIN to be conservative: if player is underperforming
                            # this season, use the lower rate.
                            if abs(p_yes - recency_rate) > 0.10:
                                p_yes = min(p_yes, recency_rate)
                        yes_edge = p_yes - yes_mid
                        if yes_edge >= min_edge and 0.10 <= yes_mid <= 0.80:
                            _, team = get_team_from_ticker(ticker)
                            label = f"{pname} ({team}) {line_val}+ {desc}"
                            all_bets.append({
                                "type": mname, "ticker": ticker,
                                "side": "yes",
                                "price_cents": max(1, int(yes_mid * 100)),
                                "model_prob": round(p_yes, 4),
                                "market_prob": round(yes_mid, 4),
                                "edge": round(yes_edge, 4),
                                "contracts": 1,
                                "player": pname,
                                "team": team,
                                "line_val": line_val,
                                "stat_desc": desc,
                                "label": label,
                            })
                            count += 1
                    except Exception:
                        pass
                print(f"  {mname:5s} ({series:10s}): {count} qualifying bets (info_only markets excluded)")
        else:
            print("  No feature data available")
    except Exception as e:
        print(f"  MLB props error: {e}")
    # Add message if no MLB markets were active (all info_only)
    if all(mt.get("info_only", True) for mt in MARKET_TYPES):
        print("  (All MLB models failed backtest — no bets from model-based player props)")

    # === 2. MLB F5 (First 5 Innings - Team Win) ===
    print()
    print("  " + "-" * 66)
    print("  2. MLB F5 MARKETS (Team Win - model quality: experimental)")
    print("  " + "-" * 66)
    try:
        from src.scripts.scan_f5 import F5Scanner
        sc = F5Scanner(balance=balance)
        if sc.model is not None:
            date = datetime.now().strftime("%Y-%m-%d")
            f5_bets = sc.scan(date)
            for b in f5_bets:
                ticker = b["ticker"]
                team_code, team_name = get_team_from_ticker(ticker)
                outcome = b.get("outcome", "") if b.get("outcome") != "TIE" else "Tie"
                label = f"F5-{b.get('game','?')} {outcome}"
                all_bets.append({
                    "type": "F5", "ticker": ticker,
                    "side": "yes",
                    "price_cents": int(b["market_mid"] * 100),
                    "model_prob": b["model_prob"] / 100.0,
                    "market_prob": b.get("fair_prob", b["market_prob"]) / 100.0,
                    "edge": b["edge"] / 100.0,
                    "contracts": 1,
                    "player": f"{b.get('game','?')}",
                    "team": outcome,
                    "line_val": 0,
                    "label": label,
                })
            f5_count = len([b for b in all_bets if b["type"] == "F5"])
            print(f"  -> {f5_count} F5 bets")
        else:
            print("  Model not available - skipping")
    except Exception as e:
        print(f"  F5 error: {e}")

    # === 3. World Cup 2026 ===
    print()
    print("  " + "-" * 66)
    print("  3. WORLD CUP 2026")
    print("  " + "-" * 66)
    try:
        from src.scripts.scan_wc import scan as wc_scan
        wc_bets = wc_scan()
        if wc_bets:
            for q in wc_bets:
                # scan_wc edge_pct is ratio-% (model_p/mkt_p - 1) × 100,
                # e.g. 692% for a 6.5c longshot.  Use simple diff as edge.
                model_p = q["model_p"]
                mkt_p = q["mkt_p"]
                edge = model_p - mkt_p
                all_bets.append({
                    "type": "WC", "ticker": q["ticker"],
                    "side": "yes",
                    "price_cents": max(1, int(q["ya"] * 100)),
                    "market_prob": mkt_p, "model_prob": model_p,
                    "edge": edge,
                    "contracts": q.get("contracts", 1),
                    "player": q.get("match", "?"),
                    "team": q.get("pick", "?"),
                    "line_val": 0,
                    "label": f"WC-{q.get('match','?')} {q.get('pick','?')}",
                })
            print(f"  -> {len(wc_bets)} WC bets")
        else:
            print("  No qualifying WC bets found")
    except Exception as e:
        print(f"  World Cup error: {e}")

    # === 4. NFL PLAYER PROPS (PASS_YDS, PASS_TD, RUSH_YDS, REC, REC_YDS, TD, INT) ===
    print()
    print("  " + "-" * 66)
    print("  4. NFL PLAYER PROPS (PASS_YDS, PASS_TD, RUSH_YDS, REC, REC_YDS, TD)")
    print("  " + "-" * 66)
    nfl_bet_count = 0
    try:
        from src.scripts.kalshi_nfl_unified import (
            load_features as load_nfl_features,
            MARKET_TYPES as NFL_MARKET_TYPES,
            _load_regressor as _load_nfl_regressor,
            _match_player as _match_nfl_player,
            _p_ge_line as _p_ge_nfl_line,
        )
        nfl_latest = load_nfl_features()
        if nfl_latest is not None and not nfl_latest.empty:
            # Take latest week per player
            if "game_date" in nfl_latest.columns:
                nfl_latest = nfl_latest.sort_values("game_date").groupby("player_id").last().reset_index()
            nfl_players = len(nfl_latest)
            print(f"  Loaded {nfl_players} players", flush=True)

            nfl_model_cache = {}
            for mt in NFL_MARKET_TYPES:
                if mt.get("info_only", True):
                    continue
                mname = mt["name"]
                series = mt["series_ticker"]
                pattern = mt["pattern"]
                pos_filter = mt.get("position")
                model_name = mt["model_name"]
                desc = mt["desc"]

                # Try to fetch markets — if none exist (off-season), skip gracefully
                try:
                    mkts = kc.list_markets(series_ticker=series, limit=500)
                    if mkts is None or mkts.empty:
                        print(f"  {mname:8s} ({series:12s}): no markets (off-season?)")
                        continue
                except Exception:
                    print(f"  {mname:8s} ({series:12s}): cannot reach Kalshi")
                    continue

                # Load model (cached per model_name)
                if model_name not in nfl_model_cache:
                    m, s, feats, cal = _load_nfl_regressor(model_name)
                    if m is None:
                        print(f"  {mname:8s}: no regressor for {model_name}")
                        continue
                    nfl_model_cache[model_name] = (m, s, feats, cal)
                reg_model, reg_std, true_features, beta_cal = nfl_model_cache[model_name]

                count = 0
                for _, row in mkts.iterrows():
                    try:
                        ticker = row["ticker"]
                        title = row.get("title", "")
                        yb_v = row.get("yes_bid_dollars", 0)
                        ya_v = row.get("yes_ask_dollars", 1)
                        yb = 0.0 if (isinstance(yb_v, float) and (yb_v != yb_v)) else float(yb_v or 0)
                        ya = 1.0 if (isinstance(ya_v, float) and (ya_v != ya_v)) else float(ya_v or 1)
                        if yb <= 0 and ya >= 1.0:
                            continue
                        yes_mid = max(0.01, min(0.99, (yb + ya) / 2.0))

                        lm = re.match(pattern, title, re.IGNORECASE)
                        if not lm:
                            continue
                        pname = lm.group(1).strip()
                        line_val = int(lm.group(2))
                        if line_val <= 0:
                            continue

                        row_match = _match_nfl_player(pname, nfl_latest, position_filter=pos_filter)
                        if row_match is None:
                            continue

                        # Skip players with insufficient data
                        avg_cols = [c for c in row_match.index
                                    if c.endswith("_avg_7") and isinstance(row_match[c], (int, float))]
                        if avg_cols and all(pd.isna(row_match[c]) for c in avg_cols):
                            continue

                        p_yes, mu = _p_ge_nfl_line(
                            row_match, reg_model, reg_std, line_val, true_features,
                            stat_name=mname, beta_cal=beta_cal,
                        )
                        yes_edge = p_yes - yes_mid

                        if yes_edge >= min_edge and 0.10 <= yes_mid <= 0.80:
                            label = f"{pname} {line_val}+ {desc}"
                            all_bets.append({
                                "type": mname, "ticker": ticker,
                                "side": "yes",
                                "price_cents": max(1, int(yes_mid * 100)),
                                "model_prob": round(p_yes, 4),
                                "market_prob": round(yes_mid, 4),
                                "edge": round(yes_edge, 4),
                                "contracts": 1,
                                "player": pname,
                                "team": "",
                                "line_val": line_val,
                                "stat_desc": desc,
                                "label": label,
                            })
                            count += 1
                    except Exception:
                        pass

                print(f"  {mname:8s} ({series:12s}): {count} qualifying bets")
                nfl_bet_count += count
        else:
            print("  No feature data available (run data pipeline first)")
    except Exception as e:
        print(f"  NFL props error: {e}")

    if nfl_bet_count == 0:
        print("  (NFL is off-season — no markets or no qualifying edges)")

    # === 5. NBA PLAYER PROPS (PTS, REB, AST, BLK, STL, 3PT, FTM, PRA, PA, PR, RA) ===
    print()
    print("  " + "-" * 66)
    print("  5. NBA PLAYER PROPS (PTS, REB, AST, BLK, STL, 3PT, FTM, PRA)")
    print("  " + "-" * 66)
    nba_bet_count = 0
    try:
        from src.scripts.nba_bet import get_nba_bets
        nba_bets = get_nba_bets(kc=kc, min_edge=min_edge)
        if nba_bets:
            for b in nba_bets:
                all_bets.append(b)
            nba_bet_count = len(nba_bets)
            print(f"  -> {nba_bet_count} qualifying NBA bets")
            for b in sorted(nba_bets, key=lambda x: -x["edge"])[:5]:
                print(f"     {b['player']:25s} {b.get('line_val',0)}+ {b.get('stat_desc',''):10s} "
                      f"model={b['model_prob']:.0%} mkt={b['market_prob']:.0%} edge={b['edge']:+.0%}")
        else:
            print("  No qualifying NBA bets found")
    except Exception as e:
        print(f"  NBA props error: {e}")

    if nba_bet_count == 0:
        print("  (NBA is off-season — no markets or no qualifying edges)")

    # === 6. WNBA PLAYER PROPS (PTS, REB, AST, 3PT) ===
    print()
    print("  " + "-" * 66)
    print("  6. WNBA PLAYER PROPS (PTS, REB, AST, 3PT — team-level models)")
    print("  " + "-" * 66)
    wnba_bet_count = 0
    try:
        from src.scripts.kalshi_wnba_unified import (
            load_features as load_wnba_features,
            MARKET_TYPES as WNBA_MARKET_TYPES,
        )
        wnba_latest = load_wnba_features()
        if wnba_latest is not None and not wnba_latest.empty:
            if "game_date" in wnba_latest.columns:
                wnba_latest = wnba_latest.sort_values("game_date").groupby("player_id").last().reset_index()
            print(f"  Loaded features for {len(wnba_latest)} teams", flush=True)

            for mt in WNBA_MARKET_TYPES:
                mname = mt["name"]
                series = mt["series_ticker"]
                pattern = mt["pattern"]
                desc = mt["desc"]

                try:
                    mkts = kc.list_markets(series_ticker=series, limit=50)
                    if mkts is None or mkts.empty:
                        print(f"  {mname:4s} ({series:11s}): no markets (off-season?)")
                        continue
                except Exception:
                    print(f"  {mname:4s} ({series:11s}): cannot reach Kalshi")
                    continue

                count = 0
                for _, row in mkts.iterrows():
                    try:
                        ticker = row["ticker"]
                        title = row.get("title", "")
                        yb_v = row.get("yes_bid_dollars", 0)
                        ya_v = row.get("yes_ask_dollars", 1)
                        yb = 0.0 if (isinstance(yb_v, float) and (yb_v != yb_v)) else float(yb_v or 0)
                        ya = 1.0 if (isinstance(ya_v, float) and (ya_v != ya_v)) else float(ya_v or 1)
                        if yb <= 0 and ya >= 1.0:
                            continue
                        yes_mid = max(0.01, min(0.99, (yb + ya) / 2.0))
                        lm = re.match(pattern, title, re.IGNORECASE)
                        if not lm:
                            continue
                        pname = lm.group(1).strip()
                        line_val = int(lm.group(2))
                        label = f"WNBA-{pname} {line_val}+ {desc}"
                        all_bets.append({
                            "type": f"WNBA-{mname}", "ticker": ticker,
                            "side": "yes",
                            "price_cents": max(1, int(yes_mid * 100)),
                            "model_prob": 0.5,
                            "market_prob": round(yes_mid, 4),
                            "edge": 0.0,
                            "contracts": 1,
                            "player": pname,
                            "team": "",
                            "line_val": line_val,
                            "stat_desc": f"WNBA {desc}",
                            "label": label,
                        })
                        count += 1
                    except Exception:
                        pass

                print(f"  {mname:4s} ({series:11s}): {count} markets (info_only — team-level models)")
                wnba_bet_count += count
        else:
            print("  No feature data available")
    except Exception as e:
        print(f"  WNBA props error: {e}")

    if wnba_bet_count == 0:
        print("  (WNBA is off-season — no markets or no qualifying edges)")

    # === 7. NHL PLAYER PROPS (GOALS, ASSISTS, POINTS, SHOTS, PIM) ===
    print()
    print("  " + "-" * 66)
    print("  7. NHL PLAYER PROPS (GOALS, ASSISTS, POINTS, SHOTS, PIM)")
    print("  " + "-" * 66)
    nhl_bet_count = 0
    try:
        from src.scripts.kalshi_nhl_unified import (
            load_features as load_nhl_features,
            MARKET_TYPES as NHL_MARKET_TYPES,
        )
        nhl_latest = load_nhl_features()
        if nhl_latest is not None and not nhl_latest.empty:
            if "game_date" in nhl_latest.columns:
                nhl_latest = nhl_latest.sort_values("game_date").groupby("player_id").last().reset_index()
            print(f"  Loaded {len(nhl_latest)} players", flush=True)

            for mt in NHL_MARKET_TYPES:
                mname = mt["name"]
                series = mt["series_ticker"]
                pattern = mt["pattern"]
                desc = mt["desc"]

                try:
                    mkts = kc.list_markets(series_ticker=series, limit=50)
                    if mkts is None or mkts.empty:
                        print(f"  {mname:5s} ({series:11s}): no markets (off-season?)")
                        continue
                except Exception:
                    print(f"  {mname:5s} ({series:11s}): cannot reach Kalshi")
                    continue

                count = 0
                for _, row in mkts.iterrows():
                    try:
                        ticker = row["ticker"]
                        title = row.get("title", "")
                        yb_v = row.get("yes_bid_dollars", 0)
                        ya_v = row.get("yes_ask_dollars", 1)
                        yb = 0.0 if (isinstance(yb_v, float) and (yb_v != yb_v)) else float(yb_v or 0)
                        ya = 1.0 if (isinstance(ya_v, float) and (ya_v != ya_v)) else float(ya_v or 1)
                        if yb <= 0 and ya >= 1.0:
                            continue
                        yes_mid = max(0.01, min(0.99, (yb + ya) / 2.0))
                        lm = re.match(pattern, title, re.IGNORECASE)
                        if not lm:
                            continue
                        pname = lm.group(1).strip()
                        line_val = int(lm.group(2))
                        label = f"NHL-{pname} {line_val}+ {desc}"
                        all_bets.append({
                            "type": f"NHL-{mname}", "ticker": ticker,
                            "side": "yes",
                            "price_cents": max(1, int(yes_mid * 100)),
                            "model_prob": 0.5,
                            "market_prob": round(yes_mid, 4),
                            "edge": 0.0,
                            "contracts": 1,
                            "player": pname,
                            "team": "",
                            "line_val": line_val,
                            "stat_desc": f"NHL {desc}",
                            "label": label,
                        })
                        count += 1
                    except Exception:
                        pass

                print(f"  {mname:5s} ({series:11s}): {count} markets (info_only — off-season)")
                nhl_bet_count += count
        else:
            print("  No feature data available")
    except Exception as e:
        print(f"  NHL props error: {e}")

    if nhl_bet_count == 0:
        print("  (NHL is off-season — no markets or no qualifying edges)")

    # === 8. UFC FIGHT MARKETS ===
    print()
    print("  " + "-" * 66)
    print("  8. UFC FIGHT MARKETS (Winner — experimental)")
    print("  " + "-" * 66)
    ufc_bet_count = 0
    try:
        from src.scripts.kalshi_ufc import get_ufc_bets
        ufc_bets = get_ufc_bets(kc=kc, min_edge=min_edge)
        if ufc_bets:
            for b in ufc_bets:
                all_bets.append(b)
            ufc_bet_count = len(ufc_bets)
            print(f"  -> {ufc_bet_count} qualifying UFC bets")
            for b in sorted(ufc_bets, key=lambda x: -x["edge"])[:5]:
                print(f"     {b['player']:25s} model={b['model_prob']:.0%} mkt={b['market_prob']:.0%} edge={b['edge']:+.0%}")
        else:
            print("  No qualifying UFC bets found")
    except Exception as e:
        print(f"  UFC error: {e}")

    # === 9. COLLEGE FOOTBALL (game winner markets — OFF-SEASON, info_only) ===
    print()
    print("  " + "-" * 66)
    print("  9. COLLEGE FOOTBALL (off-season — models trained, betting disabled)")
    print("  " + "-" * 66)
    try:
        from src.scripts.kalshi_cfb import get_cfb_bets
        cfb_bets = get_cfb_bets(kc=kc, min_edge=min_edge)
        if cfb_bets:
            print(f"  Found {len(cfb_bets)} opportunities (info_only — season hasn't started)")
            for b in sorted(cfb_bets, key=lambda x: -x["edge"])[:5]:
                print(f"     {b['player']:25s} vs {b['team']:25s} "
                      f"model={b['model_prob']:.0%} mkt={b['market_prob']:.0%} edge={b['edge']:+.0%}")
            # Don't add CFB bets to all_bets — season hasn't started
        else:
            print("  No CFB markets found")
    except Exception as e:
        print(f"  CFB error: {e}")

    # === 10. Safe Compounder (display only — never place live) ===
    print()
    print("  " + "-" * 66)
    print("  10. SAFE COMPOUNDER (NO-side on longshots — INFO ONLY, never traded)")
    print("  " + "-" * 66)
    try:
        from src.execution.kalshi_trader import KalshiTrader
        kt = KalshiTrader(risk=risk)
        opps = kt.safe_compounder_scan()
        if opps:
            for o in opps[:10]:
                if o.get("size", 0) > 0:
                    all_bets.append({
                        "type": "COMP", "ticker": o["ticker"],
                        "side": "no",
                        "price_cents": o["order_price_cents"],
                        "contracts": o.get("size", 1),
                        "model_prob": 1 - o.get("yes_price", 0.5),
                        "market_prob": 1 - o["order_price_cents"] / 100.0,
                        "edge": o["edge_cents"] / 100.0,
                        "player": o["ticker"][:30], "team": "",
                        "line_val": 0,
                        "label": f"COMP-{o['ticker'][:30]}",
                    })
            print(f"  -> {len(opps)} compounder opportunities")
        else:
            print("  No opportunities")
    except Exception as e:
        print(f"  Compounder error: {e}")

    # === 11. Multi-Leg Parlays (all prop types + F5 + CFB + NFL + NBA + WNBA + NHL) ===
    print()
    print("  " + "-" * 66)
    print("  11. MULTI-LEG PARLAYS (2/3/4-leg — all prop types)")
    print("  " + "-" * 66)
    PARLAY_ALLOWED_TYPES = {"KS", "HR", "TB", "HRR", "F5", "WC", "UFC",
                           "PASS_YDS", "PASS_TD", "RUSH_YDS", "REC", "REC_YDS", "TD",
                           "PTS", "REB", "AST", "BLK", "STL", "3PT", "FTM",
                           "PRA", "PA", "PR", "RA", "2D", "3D",
                           "WNBA-PTS", "WNBA-REB", "WNBA-AST", "WNBA-3PT", "WNBA-TOTAL",
                           "NHL-GOALS", "NHL-ASSISTS", "NHL-POINTS", "NHL-SHOTS", "NHL-PIM", "NHL-G+A"}
    parlay_opps = []
    for b in all_bets:
        bt = b["type"]
        # Include all prop types; exclude COMP (safe compounder is a different strategy)
        if bt not in PARLAY_ALLOWED_TYPES:
            continue
        if b.get("price_cents", 50) < 1 or b.get("price_cents", 50) > 99:
            continue
        if b.get("edge", 0) < min_edge:
            continue
        parlay_opps.append({
            "ticker": b["ticker"],
            "title": b.get("label", ""),
            "type": bt,
            "stat_type": b.get("stat_desc", bt),
            "market_prob": b.get("market_prob", 0.5),
            "model_prob": b.get("model_prob", 0.5),
            "edge": b.get("edge", 0),
            "price_cents": b.get("price_cents", 50),
        })
    if len(parlay_opps) >= 2:
        finder = KalshiParlayFinder(min_edge=min_edge)
        parlay_results = finder.find_best(parlay_opps, top_n=5)
        display_parlays(parlay_results, "Best 2/3/4-leg Combos (all props + F5 + CFB)")
    else:
        print(f"  Need 2+ edges (have {len(parlay_opps)})")

    # === 12. TOP PLAYS Leaderboard ===
    print()
    print("=" * 70)
    print("  * TOP PLAYS FOR TODAY - Ranked by Edge")
    print("=" * 70)
    top_plays = sorted(all_bets, key=lambda x: -x.get("edge", 0))
    if top_plays:
        print(f"  {'Rank':4s} {'Type':5s} {'Player':30s} {'Bet':18s} {'Edge':8s} {'Price':6s}")
        print(f"  {'-'*4} {'-'*5} {'-'*30} {'-'*18} {'-'*8} {'-'*6}")
        for i, p in enumerate(top_plays[:15]):
            player = p.get("player", "")[:29]
            team = p.get("team", "")
            line = p.get("line_val", 0)
            stat_desc = p.get("stat_desc", "")
            edge_str = f"{p.get('edge',0):+.0%}"
            price = p.get("price_cents", 0)
            bt = p["type"]
            if bt in ("KS", "HR", "TB", "HRR"):
                bet_str = f"{line}+ {stat_desc}"
            elif bt == "F5":
                bet_str = f"{team[:10]} win"
            else:
                bet_str = team[:16]
            if team:
                player_display = f"{player} ({team})"
            else:
                player_display = player
            print(f"  #{i+1:<2d} {bt:5s} {player_display[:29]:30s} {bet_str:18s} {edge_str:>8s} {price:3d}c")
        print(f"\n  Total qualifying: {len(top_plays)}")

        # Recommended bets
        seen = set()
        rec = []
        for p in top_plays:
            gk = extract_team_key(p.get("ticker", ""))
            if gk not in seen or not gk:
                rec.append(p)
                seen.add(gk)
                if len(rec) >= 5:
                    break
        if rec:
            print(f"\n  * RECOMMENDED (1/game, quarter-Kelly sized):")
            for i, p in enumerate(rec):
                price = p.get("price_cents", 50) / 100.0
                edge = max(0.001, p.get("edge", 0))
                kelly = min(edge / max(0.001, 1 - p.get("market_prob", 0.5)), 0.03) * 0.25
                bet = min(kelly * balance, balance * 0.03)
                ctr = max(1, int(bet / max(0.001, price)))
                cost = ctr * price
                print(f"  #{i+1}: {p.get('label',''):50s} {ctr}ctr @ {p.get('price_cents',0)}c = ${cost:.2f}")
    else:
        print("  No qualifying plays found")

    # === 13. Place Orders ===
    if auto_bet and all_bets:
        print()
        print("=" * 70)
        print("  13. PLACING SINGLE ORDERS (--bet mode)")
        print("=" * 70)
        top_plays = sorted(all_bets, key=lambda x: -x.get("edge", 0))
        placed = 0
        max_bets = 5
        total_exposure = 0.0
        max_exposure_pct = 0.25
        seen_orders = set()
        orders = kc._request("GET", "/portfolio/orders").get("orders", [])
        existing_tickers = set(o.get("ticker", "") for o in orders if o.get("status") in ("resting", "open"))
        for p in top_plays:
            if placed >= max_bets:
                break
            # NEVER place compounder/non-sports bets (Pope, Mars, etc.)
            if p["type"] == "COMP":
                continue
            ticker = p["ticker"]
            if ticker in existing_tickers:
                continue
            if ticker in seen_orders:
                continue
            seen_orders.add(ticker)
            price_cents = p["price_cents"]
            edge = p["edge"]
            if edge < min_edge:
                continue
            kelly_full = max(0.0, edge / max(0.001, 1 - p.get("market_prob", 0.5)))
            kelly_quarter = kelly_full * 0.25
            kelly_capped = min(kelly_quarter, 0.03)
            max_contracts = int(kelly_capped * balance / (price_cents / 100.0)) if price_cents > 0 else 0
            contracts = max(1, max_contracts) if max_contracts >= 1 else 0
            cost = (price_cents / 100.0) * contracts if contracts > 0 else 0
            if contracts < 1:
                continue
            if total_exposure + cost > balance * max_exposure_pct:
                continue
            if place_with_verify(kc, ticker, p["side"], price_cents, contracts, p["model_prob"]):
                placed += 1
                total_exposure += cost
                time.sleep(2)
        print(f"  Placed {placed} single orders | Exposure: ${total_exposure:.2f} | Balance: ${kc.get_balance():.2f}")

        # === 13b. Place Parlay Orders ===
        parlay_placed = 0
        total_parlay_exposure = 0.0
        print()
        print("=" * 70)
        print("  13b. PLACING PARLAY ORDERS (--bet mode, if parlays found)")
        print("=" * 70)
        try:
            parlay_bets = []
            for b in all_bets:
                bt = b["type"]
                if bt not in PARLAY_ALLOWED_TYPES:
                    continue
                if b.get("price_cents", 50) < 1 or b.get("price_cents", 50) > 99:
                    continue
                if b.get("edge", 0) < min_edge:
                    continue
                parlay_bets.append({
                    "ticker": b["ticker"],
                    "title": b.get("label", ""),
                    "type": bt,
                    "stat_type": b.get("stat_desc", bt),
                    "market_prob": b.get("market_prob", 0.5),
                    "model_prob": b.get("model_prob", 0.5),
                    "edge": b.get("edge", 0),
                    "price_cents": b.get("price_cents", 50),
                })
            if len(parlay_bets) >= 2:
                finder = KalshiParlayFinder(min_edge=min_edge, kc=kc)
                parlay_results = finder.find_best(parlay_bets, top_n=3)
                for n_legs in sorted(parlay_results.keys()):
                    for parlay in parlay_results[n_legs][:2]:
                        if total_exposure + total_parlay_exposure >= balance * max_exposure_pct:
                            break
                        # Stake = parlay Kelly of remaining bankroll
                        remaining = balance - total_exposure - total_parlay_exposure
                        stake_pct = parlay.kelly_fraction
                        total_stake = remaining * stake_pct
                        stake_per_leg = total_stake / max(len(parlay.legs), 1)
                        print(f"\n  Parlay {n_legs}-leg (EV={parlay.expected_value:+.1%}, ρ̅={parlay.implied_correlation:.0%}, stake=${total_stake:.2f}):")
                        for leg in parlay.legs:
                            price = leg.price_cents / 100.0
                            if price <= 0 or price >= 1:
                                continue
                            contracts = max(1, int(stake_per_leg / max(price, 0.001)))
                            cost = contracts * price
                            if place_with_verify(kc, leg.ticker, "yes", leg.price_cents, contracts, leg.model_prob):
                                parlay_placed += 1
                                total_parlay_exposure += cost
                                time.sleep(2)
                            seen_orders.add(leg.ticker)
                print(f"  Placed {parlay_placed} parlay legs | Exposure: ${total_parlay_exposure:.2f}")
            else:
                print("  Need 2+ edges for parlays, skipping")
        except Exception as e:
            print(f"  Parlay betting error: {e}")

        total_orders = placed + parlay_placed
        total_exp = total_exposure + total_parlay_exposure
        print(f"\n  Total: {total_orders} orders | Total exposure: ${total_exp:.2f} | Balance: ${kc.get_balance():.2f}")
    elif auto_bet and not all_bets:
        print()
        print("  No qualifying bets found — nothing to place")
    elif not auto_bet:
        print()
        print("  DRY RUN - no orders placed")
        print("  To place live orders: (1) add BETTING_ENABLED=true to .env, (2) run with --bet")

    # === 14. Log to Trade Tracker (paper or live) ===
    if all_bets:
        paper_mode = "--paper" in sys.argv
        tt = TradeTracker()

        # Sport lookup by bet type — ordered dict, first match wins
        SPORT_BY_TYPE = {
            # MLB
            "KS": "mlb", "HR": "mlb", "TB": "mlb", "HRR": "mlb", "F5": "mlb",
            # NFL
            "PASS_YDS": "nfl", "PASS_TD": "nfl", "RUSH_YDS": "nfl",
            "REC": "nfl", "REC_YDS": "nfl", "TD": "nfl", "INT": "nfl",
            "PASS_ATT": "nfl",            "PASS_ATT": "nfl", "RUSH+REC_YDS": "nfl",
            # NBA
            "PTS": "nba", "REB": "nba", "AST": "nba", "BLK": "nba",
            "STL": "nba", "3PT": "nba", "FTM": "nba",
            "PRA": "nba", "PA": "nba", "PR": "nba", "RA": "nba",
            "2D": "nba", "3D": "nba",
            # World Cup
            "WC": "world_cup",
            # UFC
            "UFC": "ufc",
            # CFB
            "CFB": "cfb",
            # Safe Compounder
            "COMP": "compounder",
        }
        # Prefix-based fallbacks (checked after exact match)
        PREFIX_SPORT = [
            ("WNBA-", "wnba"),
            ("NHL-", "nhl"),
            ("MLB-", "mlb"),
        ]

        for b in all_bets:
            bt = b.get("type", "")
            # Exact match first
            _sport = SPORT_BY_TYPE.get(bt)
            # Prefix fallback
            if _sport is None:
                for pfx, sport_label in PREFIX_SPORT:
                    if bt.startswith(pfx):
                        _sport = sport_label
                        break
            if _sport is None:
                _sport = "unknown"
            tt.log_trade(
                sport=_sport,
                model_name=b.get("type", "unknown"),
                ticker=b.get("ticker", ""),
                title=b.get("label", ""),
                side=b.get("side", "yes"),
                price_cents=b.get("price_cents", 50),
                size=b.get("contracts", 1),
                model_prob=b.get("model_prob", 0.5),
                market_prob=b.get("market_prob", 0.5),
                edge=b.get("edge", 0),
                live=not paper_mode,
                notes="paper_trade" if paper_mode else ("live_bet" if auto_bet else "dry_run"),
            )
        mode_label = "PAPER" if paper_mode else ("LIVE" if auto_bet else "DRY")
        print(f"  Logged {len(all_bets)} trades to tracker ({mode_label})")

    print()
    print("=" * 70)
    print(f"  DONE - Balance: ${kc.get_balance():.2f}")
    print("=" * 70)


if __name__ == "__main__":
    # Load .env so BETTING_ENABLED is available
    _load_dotenv()

    auto = "--bet" in sys.argv
    bankroll = None
    for i, a in enumerate(sys.argv):
        if a == "--bankroll" and i + 1 < len(sys.argv):
            bankroll = float(sys.argv[i + 1])

    # ── Safety gate at entry point ──
    betting_enabled = os.environ.get("BETTING_ENABLED", "").strip().lower() == "true"
    if auto and not betting_enabled:
        print()
        print("  " + "!" * 66)
        print("  !!! SAFETY BLOCK: --bet was passed but BETTING_ENABLED is not 'true'")
        print("  !!! Add BETTING_ENABLED=true to .env to enable live betting")
        print("  !!! Running as DRY RUN instead")
        print("  " + "!" * 66)
        print()
        auto = False

    morning_scan(bankroll=bankroll, auto_bet=auto)
