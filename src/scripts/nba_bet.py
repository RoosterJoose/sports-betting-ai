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

# NOTE on FG3A (3-point attempts):
# The FG3A model exists at models/nba/FG3A.json and backtests 10/10 lines
# beating the naive baseline (formal backtest, June 9 2026 — see PROJECT.md).
# However, it is NOT wired into the scanner here because Kalshi's NBA series
# ticker KXNBA3PT is for 3-POINT MAKES (FG3M), not attempts. The "3PT" entry
# below loads model FG3M, not FG3A. If Kalshi ever launches a 3PT attempts
# market (e.g., KXNBAG3A), the FG3A model is ready to be wired by adding a
# new tuple to MARKETS — see _load_regressor in kalshi_nba_unified.py.
#
# NOTE on composite stats (PA, PR, RA, PRA):
# All 4 composite models exist at models/nba/{PA,PR,RA,PRA}.json and backtest
# 100% beat naive (June 9, 2026, see PROJECT.md):
#   PA   sigma=6.82  Mean |Bias| 2.3%  Beats naive 38/38 (100%)
#   PR   sigma=7.43  Mean |Bias| 2.4%  Beats naive 42/42 (100%)
#   RA   sigma=3.50  Mean |Bias| 2.3%  Beats naive 19/19 (100%)
#   PRA  sigma=8.12  Mean |Bias| 2.1%  Beats naive 49/49 (100%)
# They are wired below as no-ops because Kalshi does NOT currently list series
# tickers for combo stats (KXNBAPA/KXNBAPR/KXNBARA/KXNBAPRA all return 0
# markets as of June 9, 2026). When Kalshi lists any of these series, the
# scanner picks them up automatically — just re-run --scan.
MARKETS = [
    ("PTS", "KXNBAPTS", "PTS", False), ("REB", "KXNBAREB", "REB", False),
    ("AST", "KXNBAAST", "AST", False),
    # BLK + STL gated to info_only: backtest only 57% / 71% beat naive
    # (June 9, 2026 — see PROJECT.md). Low-count Poisson stats with high
    # variance; keep models loaded for diagnostics but don't trade them.
    ("BLK", "KXNBABLK", "BLK", True), ("STL", "KXNBASTL", "STL", True),
    ("3PT", "KXNBA3PT", "FG3M", False), ("FTM", "KXNBAFTM", "FTM", False),
    # Combo stats — trained + calibrated, awaiting Kalshi series tickers
    ("PA",  "KXNBAPA",  "PA",  False),
    ("PR",  "KXNBAPR",  "PR",  False),
    ("RA",  "KXNBARA",  "RA",  False),
    ("PRA", "KXNBAPRA", "PRA", False),
]

# Cap on per-trade edge. Empirical data (1,581 re-resolved NBA trades, Jun 8-10
# 2026) shows win rate is INVERTED with edge: peak WR at 15-20% edge (24.5%),
# then collapsing — 30-40% edge = 9.7% WR, 40%+ edge = 1.1% WR (87 trades, 86
# lost). High edge = high model overconfidence, not high signal. Filtering here
# is the single highest-ROI fix from the calibration investigation.
MAX_EDGE = 0.20


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
    skipped_info_only = 0

    for name, series, model_name, info_only in MARKETS:
        # Honor the info_only gate: scan but never bet these markets
        if info_only:
            skipped_info_only += 1
            continue
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
                if edge > MAX_EDGE:
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
    if skipped_info_only:
        print(f"  Skipped {skipped_info_only} info-only markets (no trade signal)", flush=True)

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
        if info_only:
            print(f"\nSkipping {name} ({series}) — info_only (no trade signal)", flush=True)
            continue
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

    # Apply edge cap (filter out overconfident model picks)
    pre_cap = len(all_opps)
    all_opps = [o for o in all_opps if o.get("edge", 0) <= MAX_EDGE]
    filtered = pre_cap - len(all_opps)
    if filtered:
        print(f"\n  Edge cap @ {MAX_EDGE:.0%}: filtered {filtered} overconfident picks")

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
            if o["edge"] > MAX_EDGE:  # defense in depth — should already be filtered upstream
                continue
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
