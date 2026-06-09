#!/usr/bin/env python3
"""Resolve pending paper trades from the trade tracker.

Scans all pending trades (both paper/live), checks Kalshi for the current
market state, and resolves them as won/lost based on settlement price.

Usage:
    python -m src.scripts.resolve_paper_trades              # resolve all pending
    python -m src.scripts.resolve_paper_trades --sport mlb   # filter by sport
    python -m src.scripts.resolve_paper_trades --model KS    # filter by model
    python -m src.scripts.resolve_paper_trades --report-only # just report, no resolve
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import numpy as np

from src.data.kalshi import KalshiClient
from src.utils.trade_tracker import TradeTracker


def _safe_float(val, default=0.0):
    """Safely convert a value to float, handling empty strings and None."""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        if pd.isna(val):
            return default
        return float(val)
    s = str(val).strip()
    if s == "" or s.lower() == "none" or s.lower() == "nan":
        return default
    try:
        return float(s)
    except (ValueError, TypeError):
        return default


def resolve_pending_trades(sport: str = None, model_name: str = None,
                           report_only: bool = False) -> dict:
    """Resolve pending trades by checking Kalshi settlement prices.

    Batches by ticker prefix (first 2 segments, e.g. "KXMLBKS-25JUN08") to
    minimize API calls.  Makes one list_markets call per unique prefix,
    then matches individual tickers against the returned markets.
    """
    import time
    kc = KalshiClient()
    tt = TradeTracker()

    # Build WHERE clause
    where = ["status='pending'"]
    params = []
    if sport:
        where.append("sport=?")
        params.append(sport)
    if model_name:
        where.append("model_name=?")
        params.append(model_name)

    q = f"""
        SELECT id, sport, model_name, ticker, title, side, price_cents, size,
               model_prob, market_prob, edge, live
        FROM trades
        WHERE {' AND '.join(where)}
        ORDER BY id
    """
    pending = pd.read_sql_query(q, tt._conn, params=params)
    if pending.empty:
        return {"resolved": 0, "message": "No pending trades found"}

    print(f"Found {len(pending)} pending trades to resolve")

    # ──  Batch by ticker prefix (first 2 dash-segments) ──────────────
    def _ticker_prefix(t):
        parts = str(t).split("-")
        if len(parts) >= 2:
            return "-".join(parts[:2])
        return str(t)

    pending["_prefix"] = pending["ticker"].apply(_ticker_prefix)
    prefixes = pending["_prefix"].unique()
    print(f"  {len(prefixes)} unique ticker prefixes → {len(prefixes)} API calls")

    # ──  Fetch markets for each prefix, build ticker→market lookup ───
    print(f"  Fetching markets...", end=" ", flush=True)
    mkts_by_prefix = {}
    market_lookup = {}  # ticker → market row
    not_found_prefixes = 0
    for i, pfx in enumerate(prefixes):
        try:
            # Use series_ticker param — first segment is the series
            series = pfx.split("-")[0] if "-" in pfx else pfx
            mkts = kc.list_markets(series_ticker=series, limit=1000)
            if mkts is not None and not mkts.empty:
                mkts_by_prefix[pfx] = mkts
                for _, m in mkts.iterrows():
                    t = str(m.get("ticker", ""))
                    if t:
                        market_lookup[t] = m
            else:
                not_found_prefixes += 1
        except Exception:
            not_found_prefixes += 1
        if (i + 1) % 10 == 0:
            print(f"{i+1}/{len(prefixes)}...", end=" ", flush=True)
            time.sleep(0.5)  # rate limit
    print(f"done ({len(market_lookup)} markets indexed, {not_found_prefixes} prefixes not found)", flush=True)
    print(f"{'='*80}")

    # ──  Resolve each trade ──────────────────────────────────────────
    resolved = 0
    skipped = 0
    win_count = 0
    loss_count = 0
    total_pnl = 0.0
    total_volume = 0.0

    for _, row in pending.iterrows():
        ticker = row["ticker"]
        tid = row["id"]
        side = row["side"]
        price_cents = row["price_cents"]
        size = row["size"]
        sport_s = row["sport"]
        model = row["model_name"]
        live = row["live"]

        # Look up market from batched cache
        mkt = market_lookup.get(ticker)
        if mkt is None:
            skipped += 1
            continue

        # Check if market has settled
        result = mkt.get("result", None)
        status = str(mkt.get("status", "")).strip()
        settle_price = mkt.get("settlement_price", None)

        yes_bid = _safe_float(mkt.get("yes_bid_dollars", 0), default=0.0)
        yes_ask = _safe_float(mkt.get("yes_ask_dollars", 0), default=0.0)

        # Determine if settled
        is_settled = False
        won = None

        if result is not None and str(result).strip() not in ("", "none", "None"):
            is_settled = True
            result_f = _safe_float(result, default=0.0)
            won = (result_f >= 0.5) if side == "yes" else (result_f < 0.5)
            settle_price_f = result_f
        elif status.lower() in ("settled", "closed"):
            is_settled = True
            if settle_price is not None and str(settle_price).strip() not in ("", "none", "None"):
                settle_price_f = _safe_float(settle_price, default=0.5)
                won = (settle_price_f >= 0.5) if side == "yes" else (settle_price_f < 0.5)
            else:
                won = None
        elif yes_bid == 0 and yes_ask == 0:
            is_settled = True
            won = False if side == "yes" else True
            settle_price_f = 0.0
        elif yes_bid >= 0.99 and yes_ask >= 0.99:
            is_settled = True
            won = True if side == "yes" else False
            settle_price_f = 1.0
        else:
            skipped += 1
            continue

        if not is_settled or won is None:
            skipped += 1
            continue

        # Compute P&L
        if side == "yes":
            if won:
                pnl = size * (settle_price_f - price_cents / 100.0)
            else:
                pnl = -size * (price_cents / 100.0)
        else:
            if won:
                pnl = size * (price_cents / 100.0 - settle_price_f)
            else:
                pnl = -size * (1.0 - price_cents / 100.0)

        status_str = "won" if won else "lost"
        if not report_only:
            tt._conn.execute(
                "UPDATE trades SET status=?, resolved_price=?, pnl=? WHERE id=?",
                (status_str, settle_price_f, round(pnl, 2), tid)
            )
            tt._conn.commit()

        win_count += won
        loss_count += not won
        total_pnl += pnl
        total_volume += (price_cents / 100.0) * size
        resolved += 1

        edge_str = f"{row['edge']:.0%}"
        vol_str = f"${(price_cents/100)*size:.2f}"
        pnl_str = f"${pnl:+.2f}"
        icon = "✅" if won else "❌"
        info = f"live={live}" if live else "paper"
        print(f"  {icon} {ticker[:55]:55s} {side:4s} {row['price_cents']:3d}c "
              f"x{size:2d}={vol_str:>6s} edge={edge_str:>4s} → "
              f"{'WON' if won else 'LOST'} {pnl_str:>7s} [{info}]")

    # Summary
    print(f"\n{'='*80}")
    print(f"RESOLUTION SUMMARY")
    print(f"{'='*80}")
    print(f"  Resolved: {resolved}")
    print(f"  Skipped (still active/unknown): {skipped}")
    print(f"  Wins: {win_count} | Losses: {loss_count}")
    if win_count + loss_count > 0:
        win_rate = win_count / (win_count + loss_count)
        print(f"  Win Rate: {win_rate:.1%}")
    print(f"  Total P&L: ${total_pnl:.2f}")
    if total_volume > 0:
        roi = total_pnl / total_volume
        print(f"  Total Volume: ${total_volume:.2f}")
        print(f"  ROI: {roi:.1%}")

    return {
        "resolved": resolved,
        "skipped": skipped,
        "win_count": win_count,
        "loss_count": loss_count,
        "win_rate": round(win_count / max(win_count + loss_count, 1), 3),
        "total_pnl": round(total_pnl, 2),
        "total_volume": round(total_volume, 2),
        "roi": round(total_pnl / max(total_volume, 1), 4),
    }


if __name__ == "__main__":
    sport = None
    model = None
    report_only = "--report-only" in sys.argv

    for i, a in enumerate(sys.argv):
        if a == "--sport" and i + 1 < len(sys.argv):
            sport = sys.argv[i + 1]
        if a == "--model" and i + 1 < len(sys.argv):
            model = sys.argv[i + 1]

    result = resolve_pending_trades(sport=sport, model_name=model, report_only=report_only)
    print(f"\nDone at {datetime.now().strftime('%H:%M:%S')}")
