#!/usr/bin/env python3
"""Coordinated morning scan across all Kalshi sports.

Scans KS, F5, Safe Compounder markets, computes edges using trained models,
finds best 2/3/4-leg parlays, and displays ranked top plays with team info.

Usage:
    python -m src.scripts.morning_scan             # dry run
    python -m src.scripts.morning_scan --bet        # place orders
"""
import sys, json, warnings, time, re
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


def morning_scan(bankroll=None, auto_bet=False, min_edge=0.05):
    print("=" * 70)
    print(f"  MORNING SCAN - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    kc = KalshiClient()
    balance = bankroll or kc.get_balance()
    risk = RiskManager(bankroll=balance)
    print(f"  Balance: ${balance:.2f}  |  Min edge: {min_edge:.0%}")

    all_bets = []

    # === 1. MLB KS (Strikeouts) ===
    print()
    print("  " + "-" * 66)
    print("  1. MLB STRIKEOUT PROPS (KS) - RELIABLE MODEL")
    print("  " + "-" * 66)
    try:
        from src.scripts.kalshi_mlb_unified import load_features, MARKET_TYPES, _load_regressor, _match_player, _p_ge_line
        latest = load_features()
        if latest is not None and not latest.empty:
            print(f"  Loaded {len(latest)} players")
            ks_mkts = kc.list_markets(series_ticker="KXMLBKS", limit=200)
            if ks_mkts is not None and not ks_mkts.empty:
                today = datetime.now().strftime("%y%b%d").upper()
                ks_mkts = ks_mkts[ks_mkts["ticker"].str.contains(today, regex=False, na=False)]
                print(f"  {len(ks_mkts)} markets for today")
                for mt in MARKET_TYPES:
                    if mt["series_ticker"] != "KXMLBKS" or mt.get("info_only", False):
                        continue
                    m, s = _load_regressor(mt["model_name"])
                    if m is None:
                        continue
                    for _, row in ks_mkts.iterrows():
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
                            lm = re.match(mt["pattern"], title, re.IGNORECASE)
                            if not lm:
                                continue
                            pname = lm.group(1).strip()
                            line_val = int(lm.group(2))
                            prow = _match_player(pname, latest, position_filter=mt["position"])
                            if prow is None:
                                continue
                            p_yes, mu = _p_ge_line(prow, m, s, line_val)
                            yes_edge = p_yes - yes_mid
                            if yes_edge >= min_edge and 0.10 <= yes_mid <= 0.80:
                                _, team = get_team_from_ticker(ticker)
                                all_bets.append({
                                    "type": "KS", "ticker": ticker,
                                    "side": "yes",
                                    "price_cents": max(1, int(yes_mid * 100)),
                                    "model_prob": round(p_yes, 4),
                                    "market_prob": round(yes_mid, 4),
                                    "edge": round(yes_edge, 4),
                                    "contracts": 1,
                                    "player": pname,
                                    "team": team,
                                    "line_val": line_val,
                                    "label": f"{pname} ({team}) {line_val}+ Ks",
                                })
                        except Exception:
                            pass
                ks_bets = [b for b in all_bets if b["type"] == "KS"]
                print(f"  -> {len(ks_bets)} qualifying KS bets")
    except Exception as e:
        print(f"  KS error: {e}")

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

    # === 3. World Cup (starts June 11) ===
    print()
    print("  " + "-" * 66)
    print("  3. WORLD CUP 2026")
    print("  " + "-" * 66)
    wc_start = datetime(2026, 6, 11)
    days_until_wc = (wc_start - datetime.now()).days
    if days_until_wc <= 0:
        print("  World Cup is live! Running scan...")
        try:
            from src.scripts.scan_wc import scan as wc_scan
            wc_bets = wc_scan()
            if wc_bets:
                for q in wc_bets:
                    all_bets.append({
                        "type": "WC", "ticker": q["ticker"],
                        "side": "yes",
                        "price_cents": max(1, int(q["ya"] * 100)),
                        "market_prob": q["mkt_p"], "model_prob": q["model_p"],
                        "edge": q["edge_pct"] / 100.0,
                        "contracts": q.get("contracts", 1),
                        "player": q.get("match", "?"),
                        "team": q.get("pick", "?"),
                        "line_val": 0,
                        "label": f"WC-{q.get('match','?')} {q.get('pick','?')}",
                    })
                print(f"  -> {len(wc_bets)} WC bets")
        except Exception as e:
            print(f"  World Cup error: {e}")
    else:
        print(f"  World Cup starts in {days_until_wc} days ({wc_start.date()})")

    # === 4. Safe Compounder ===
    print()
    print("  " + "-" * 66)
    print("  4. SAFE COMPOUNDER (NO-side on longshots)")
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

    # === 5. Multi-Leg Parlays ===
    print()
    print("  " + "-" * 66)
    print("  5. MULTI-LEG PARLAYS (2/3/4-leg - KS only)")
    print("  " + "-" * 66)
    parlay_opps = []
    for b in all_bets:
        if b["type"] not in ("KS",):
            continue  # Only reliable models for parlays
        if b.get("price_cents", 50) < 1 or b.get("price_cents", 50) > 99:
            continue
        if b.get("edge", 0) < min_edge:
            continue
        parlay_opps.append({
            "ticker": b["ticker"],
            "title": b.get("label", ""),
            "market_prob": b.get("market_prob", 0.5),
            "model_prob": b.get("model_prob", 0.5),
            "edge": b.get("edge", 0),
            "price_cents": b.get("price_cents", 50),
        })
    if len(parlay_opps) >= 2:
        finder = KalshiParlayFinder(min_edge=min_edge)
        parlay_results = finder.find_best(parlay_opps, top_n=5)
        display_parlays(parlay_results, "Best 2/3/4-leg Combos (KS only)")
    else:
        print(f"  Need 2+ edges (have {len(parlay_opps)})")

    # === 6. TOP PLAYS Leaderboard ===
    print()
    print("=" * 70)
    print("  * TOP PLAYS FOR TODAY - Ranked by Edge")
    print("=" * 70)
    top_plays = sorted(all_bets, key=lambda x: -x.get("edge", 0))
    if top_plays:
        print(f"  {'Rank':4s} {'Type':5s} {'Player/Team':30s} {'Bet':18s} {'Edge':8s} {'Price':6s}")
        print(f"  {'-'*4} {'-'*5} {'-'*30} {'-'*18} {'-'*8} {'-'*6}")
        for i, p in enumerate(top_plays[:15]):
            player = p.get("player", "")[:29]
            team = p.get("team", "")
            line = p.get("line_val", 0)
            edge_str = f"{p.get('edge',0):+.0%}"
            price = p.get("price_cents", 0)
            bet_str = f"{line}+ Ks" if p["type"] == "KS" else team[:16]
            if team:
                player_display = f"{player} ({team})"
            else:
                player_display = player
            print(f"  #{i+1:<2d} {p['type']:5s} {player_display[:29]:30s} {bet_str:18s} {edge_str:>8s} {price:3d}c")
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

    # === 7. Place Orders ===
    if auto_bet and all_bets:
        print()
        print("=" * 70)
        print("  6. PLACING ORDERS (--bet mode)")
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
        print(f"  Placed {placed} orders | Exposure: ${total_exposure:.2f} | Balance: ${kc.get_balance():.2f}")
    elif not auto_bet:
        print()
        print("  DRY RUN - no orders placed (use --bet to enable)")

    print()
    print("=" * 70)
    print(f"  DONE - Balance: ${kc.get_balance():.2f}")
    print("=" * 70)


if __name__ == "__main__":
    auto = "--bet" in sys.argv
    bankroll = None
    for i, a in enumerate(sys.argv):
        if a == "--bankroll" and i + 1 < len(sys.argv):
            bankroll = float(sys.argv[i + 1])
    morning_scan(bankroll=bankroll, auto_bet=auto)
