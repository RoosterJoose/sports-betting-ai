#!/usr/bin/env python3
"""NBA Kalshi bettor — mirrors the working heredoc test pattern exactly.

Usage:
    python3 src/scripts/nba_bet.py --scan        # scan + display
    python3 src/scripts/nba_bet.py --bet          # scan + place orders
    python3 src/scripts/nba_bet.py --scan --json  # JSON output for morning_scan

Also exposes get_nba_bets() for programmatic use by morning_scan.py.
"""
import sys, re, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import datetime, date, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Import EXACTLY as the working heredoc test does
from src.data.kalshi import KalshiClient
from src.data.nba_injuries import get_out_players, is_player_out
from src.scripts.kalshi_nba_unified import _is_current_market, _match_player, load_features, _load_regressor, _p_ge_line

MARKETS = [
    ("PTS", "KXNBAPTS", "PTS", False), ("REB", "KXNBAREB", "REB", False),
    ("AST", "KXNBAAST", "AST", False), ("BLK", "KXNBABLK", "BLK", False),
    ("STL", "KXNBASTL", "STL", False), ("3PT", "KXNBA3PT", "FG3M", False),
    ("FTM", "KXNBAFTM", "FTM", False),
]


def _extract(title):
    """Extract player name and line value from Kalshi title."""
    if ":" not in title: return None, None
    parts = title.split(":", 1)
    pname = parts[0].strip()
    suffix = parts[1].strip()
    m = re.search(r'(\d+)', suffix)
    line_val = int(m.group(1)) if m else None
    return pname, line_val


def get_nba_bets(kc=None, min_edge=0.05) -> list:
    """Return structured list of qualifying NBA bets for morning_scan integration.

    Each bet dict matches the morning_scan schema:
      type, ticker, side, price_cents, model_prob, market_prob, edge,
      contracts, player, team, line_val, stat_desc, label

    Automatically filters out players listed as OUT on the ESPN injury report.
    """
    kc = kc or KalshiClient()
    latest = load_features()
    if latest is None or latest.empty:
        return []
    if "game_date" in latest.columns:
        latest = latest.sort_values("game_date").groupby("player_id").last().reset_index()

    # Fetch injury report — players who are definitively OUT
    out_players = get_out_players()
    if out_players:
        print(f"  Injury report: {len(out_players)} players OUT", flush=True)
    else:
        print(f"  Injury report: no data (ESPN unavailable — proceeding without)", flush=True)

    model_cache = {}
    results = []
    skipped_injured = 0

    for name, series, model_name, info_only in MARKETS:
        try:
            mkts = kc.list_markets(series_ticker=series, limit=500)
            if mkts is None or mkts.empty:
                continue
        except Exception:
            continue

        if model_name not in model_cache:
            m, s, feats, cal = _load_regressor(model_name)
            if m is None:
                continue
            model_cache[model_name] = (m, s, feats, cal)
        reg_model, reg_std, feature_names, beta_cal = model_cache[model_name]

        for _, row in mkts.iterrows():
            try:
                ticker = str(row.get("ticker", ""))
                title = str(row.get("title", ""))
                if not _is_current_market(ticker):
                    continue
                pname, line_val = _extract(title)
                if pname is None or line_val is None:
                    continue
                if line_val <= 0:
                    continue

                yb = float(row.get("yes_bid_dollars", 0) or 0)
                ya = float(row.get("yes_ask_dollars", 1) or 1)
                if yb <= 0 and ya >= 1.0:
                    continue
                yes_mid = max(0.01, min(0.99, (yb + ya) / 2.0))

                mrow = _match_player(pname, latest)
                if mrow is None:
                    continue

                # Skip injured/OUT players — never bet on someone who won't play
                if out_players and is_player_out(pname, out_set=out_players):
                    skipped_injured += 1
                    continue

                try:
                    p_yes, mu = _p_ge_line(mrow, reg_model, reg_std, line_val, feature_names,
                                           stat_name=name, beta_cal=beta_cal)
                except Exception:
                    continue

                edge = p_yes - yes_mid
                if edge < min_edge:
                    continue
                if yes_mid < 0.10 or yes_mid > 0.80:
                    continue

                results.append({
                    "type": name, "ticker": ticker, "side": "yes",
                    "price_cents": max(1, int(yes_mid * 100)),
                    "model_prob": round(p_yes, 4), "market_prob": round(yes_mid, 4),
                    "edge": round(edge, 4), "contracts": 1,
                    "player": pname, "team": "", "line_val": line_val,
                    "stat_desc": name, "label": f"{pname} {line_val}+ {name}",
                })
            except Exception:
                pass

    if skipped_injured:
        print(f"  Skipped {skipped_injured} injured/OUT player markets", flush=True)

    return results


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--scan", action="store_true")
    p.add_argument("--bet", action="store_true")
    p.add_argument("--json", action="store_true", help="Output JSON for morning_scan")
    args = p.parse_args()

    client = KalshiClient()
    print(f"Balance: ${client.get_balance():.2f}\n")

    latest = load_features()
    if latest is None or latest.empty:
        print("No features."); return
    if "game_date" in latest.columns:
        latest = latest.sort_values("game_date").groupby("player_id").last().reset_index()
    print(f"Players: {len(latest)}")

    all_opps = []
    out_players = get_out_players()
    skipped_injured = 0
    if out_players:
        print(f"Injury report: {len(out_players)} players OUT")

    for name, series, model_name, info_only in MARKETS:
        print(f"\nScanning {name} ({series})...", flush=True)
        try:
            mkts = client.list_markets(series_ticker=series, limit=500)
            if mkts is None or mkts.empty:
                print(f"  No markets"); continue
        except Exception as e:
            print(f"  Error: {e}"); continue
        print(f"  {len(mkts)} total", flush=True)

        model, std, feats, cal = _load_regressor(model_name)
        if model is None:
            print(f"  No model — skipping"); continue
        print(f"  Model loaded: {len(feats)} features, std={std:.2f}", flush=True)

        count = 0
        for _, row in mkts.iterrows():
            ticker = str(row.get("ticker", ""))
            title = str(row.get("title", ""))
            if not _is_current_market(ticker): continue
            pname, line_val = _extract(title)
            if pname is None or line_val is None: continue
            if line_val <= 0: continue

            yb = float(row.get("yes_bid_dollars", 0) or 0)
            ya = float(row.get("yes_ask_dollars", 1) or 1)
            if yb <= 0 and ya >= 1.0: continue
            yes_mid = max(0.01, min(0.99, (yb + ya) / 2.0))

            mrow = _match_player(pname, latest)
            if mrow is None: continue

            # Skip injured/OUT players
            if out_players and is_player_out(pname, out_set=out_players):
                skipped_injured += 1
                continue

            try:
                p_yes, mu = _p_ge_line(mrow, model, std, line_val, feats,
                                       stat_name=name, beta_cal=cal)
            except Exception:
                continue

            edge = p_yes - yes_mid
            all_opps.append({
                "type": name, "ticker": ticker, "side": "yes",
                "price_cents": max(1, int(yes_mid*100)),
                "model_prob": round(p_yes,4), "market_prob": round(yes_mid,4),
                "edge": round(edge,4), "contracts": 1,
                "player": pname, "team": "", "line_val": line_val,
                "stat_desc": name, "label": f"{pname} {line_val}+ {name}",
            })
            count += 1

        print(f"  Matched: {count}", flush=True)

    if skipped_injured:
        print(f"\nSkipped {skipped_injured} injured/OUT player markets")

    print(f"\nTotal: {len(all_opps)}")

    if args.json:
        print(json.dumps(all_opps, default=str))
        return

    all_opps.sort(key=lambda x: abs(x.get("edge",0)), reverse=True)

    if all_opps:
        print(f"\nTop 10:")
        for o in all_opps[:10]:
            print(f"  {o['type']:5s} {o['player'][:25]:25s} {o.get('line_val',0)}+ "
                  f"edge={o['edge']:+.0%} @ {o['price_cents']}c model={o['model_prob']:.0%}")

    if args.bet and all_opps:
        print(f"\n--- PLACING ORDERS ---")
        placed = 0
        for o in all_opps:
            if placed >= 8: break
            if o["edge"] < 0.04 or o["market_prob"] < 0.01: continue
            bid = min(98, max(1, int(o["market_prob"]*100)+1))
            bal = client.get_balance()
            n = max(1, int(bal * 0.05 / (bid/100.0)))
            try:
                client.create_order(ticker=o["ticker"], side="yes", yes_price=bid, count=str(n))
                print(f"  BUY {o['type']:5s} {o['player'][:25]:25s} {o.get('line_val',0)}+ "
                      f"@ {bid}c x{n} (model={o['model_prob']:.0%} mkt={o['market_prob']:.0%})", flush=True)
                placed += 1
            except Exception as e:
                print(f"  FAILED {o['player']}: {e}", flush=True)
        print(f"  Placed {placed} | Balance: ${client.get_balance():.2f}")

    print(f"\nDone at {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
