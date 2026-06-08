#!/usr/bin/env python3
"""Daily Top Plays - scans Kalshi for the best bets today.

WARNING: ALL MLB player prop models (KS, HR, TB, HRR) failed backtest.
All are worse than the naive baseline (predicting the prior).
info_only=True means scan but don't bet — edges are noise.
"""
import sys, re, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from datetime import datetime
from src.data.kalshi import KalshiClient
from src.execution.kalshi_parlay import KalshiParlayFinder, display_parlays
from src.scripts.kalshi_mlb_unified import load_features, MARKET_TYPES, _load_regressor, _match_player, _p_ge_line


def main():
    kc = KalshiClient()
    balance = kc.get_balance()
    min_edge = 0.05
    all_bets = []

    print("=" * 70)
    print(f"  TOP PLAYS - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Balance: ${balance:.2f}  |  Min edge: {min_edge:.0%}")
    print("=" * 70)

    # All MLB prop models failed backtest — short circuit
    print()
    print("  " + "-" * 66)
    print("  1. MLB PLAYER PROPS — all models info_only (backtest: worse than naive)")
    print("  " + "-" * 66)
    print("  All MLB prop models failed backtest (worse than naive baseline).")
    print("  info_only=True — no bets will be placed on these models.")

    # === 2. Multi-Leg Parlays (all info_only) ===
    print()
    print("  " + "-" * 66)
    print("  2. MULTI-LEG PARLAYS — all models info_only")
    print("  " + "-" * 66)

    parlay_opps = []
    for b in all_bets:
        if b.get("price_cents", 50) < 1 or b.get("price_cents", 50) > 99:
            continue
        if b.get("edge", 0) < min_edge:
            continue
        parlay_opps.append({
            "ticker": b["ticker"],
            "title": b.get("label", b.get("title", b.get("ticker", ""))),
            "market_prob": b.get("market_prob", 0.5),
            "model_prob": b.get("model_prob", 0.5),
            "edge": b.get("edge", 0),
            "price_cents": b.get("price_cents", 50),
        })

    if len(parlay_opps) >= 2:
        finder = KalshiParlayFinder(min_edge=min_edge)
        parlay_results = finder.find_best(parlay_opps, top_n=5)
        display_parlays(parlay_results, "Best 2/3/4-leg Combos (KS only)")
        print(f"  Source: {len(parlay_opps)} qualifying KS opportunities")
    else:
        print(f"  Need 2+ KS edges (have {len(parlay_opps)})")

    # === 3. Top Plays Leaderboard ===
    print()
    print("=" * 70)
    print("  * TOP PLAYS - Ranked by Edge (KS only)")
    print("=" * 70)

    top_plays = sorted(all_bets, key=lambda x: -x.get("edge", 0))

    if top_plays:
        print(f"  {'Rank':4s} {'Label':50s} {'Edge':8s} {'Price':6s} {'Model':6s} {'Mkt':6s}")
        print(f"  {'-'*4} {'-'*50} {'-'*8} {'-'*6} {'-'*6} {'-'*6}")
        for i, p in enumerate(top_plays[:20]):
            label = p.get("label", "")[:49]
            print(f"  #{i+1:<2d} {label:50s} {p.get('edge',0):+.0%} {p.get('price_cents',0):3d}c {p.get('model_prob',0):.0%} {p.get('market_prob',0):.0%}")

        print(f"\n  Total qualifying: {len(top_plays)}")

        # Recommended (max 1 per game)
        seen = set()
        rec = []
        for p in top_plays:
            tk = p.get("ticker", "")
            gk = tk.split("-")[1] if "-" in tk and len(tk.split("-")) > 1 else ""
            if gk not in seen or not gk:
                rec.append(p)
                seen.add(gk)
                if len(rec) >= 5:
                    break

        if rec:
            print(f"\n  * RECOMMENDED (1/game, quarter-Kelly):")
            for i, p in enumerate(rec):
                price = p.get("price_cents", 50) / 100.0
                edge = max(0.001, p.get("edge", 0))
                kelly = min(edge / max(0.001, 1 - p.get("market_prob", 0.5)), 0.03) * 0.25
                bet = min(kelly * balance, balance * 0.03)
                ctr = max(1, int(bet / max(0.001, price)))
                cost = ctr * price
                print(f"  #{i+1}: {p['type']:>3s} {p.get('label',''):45s} {ctr}ctr @ {p.get('price_cents',0)}c = ${cost:.2f}")
    else:
        print("  No qualifying plays found")

    print()
    print("=" * 70)
    print(f"  DONE - Balance: ${kc.get_balance():.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
