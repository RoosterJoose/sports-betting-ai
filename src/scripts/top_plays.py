#!/usr/bin/env python3
"""Daily Top Plays - scans Kalshi for the best bets today.

Only uses market types with non-info_only (R² > 0.04) models:
  - KS (strikeouts): SO model, pitcher props

TB, HR, HRR are info_only (R² < 0.04) - their edges are noise.
They are excluded from parlays.
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

    # === 1. MLB Strikeouts (only non-info_only market) ===
    print()
    print("  " + "-" * 66)
    print("  1. MLB STRIKEOUT PROPS - KS (model: R² > 0.04)")
    print("  " + "-" * 66)

    latest = load_features()
    if latest is None or latest.empty:
        print("  No feature data. Run training first.")
        return
    print(f"  Loaded {len(latest)} players")

    ks_mkts = kc.list_markets(series_ticker="KXMLBKS", limit=200)
    if ks_mkts is None or ks_mkts.empty:
        print("  No KS markets found")
        return

    today = datetime.now().strftime("%y%b%d").upper()
    ks_mkts = ks_mkts[ks_mkts["ticker"].str.contains(today, regex=False, na=False)]
    print(f"  {len(ks_mkts)} KS markets for today")

    for mt in MARKET_TYPES:
        if mt["series_ticker"] != "KXMLBKS":
            continue
        m, s = _load_regressor(mt["model_name"])
        if m is None:
            continue

        for _, row in ks_mkts.iterrows():
            try:
                ticker = row["ticker"]
                title = row.get("title", "")
                yb = float(row.get("yes_bid_dollars", 0) or 0)
                ya = float(row.get("yes_ask_dollars", 1) or 1)
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
                    all_bets.append({
                        "type": "KS",
                        "ticker": ticker,
                        "title": title,
                        "player": pname,
                        "side": "yes",
                        "price_cents": max(1, int(yes_mid * 100)),
                        "model_prob": round(p_yes, 4),
                        "market_prob": round(yes_mid, 4),
                        "edge": round(yes_edge, 4),
                        "label": f"KS-{pname[:20]} {line_val}+ Ks",
                    })
            except Exception:
                pass

    ks_count = len([b for b in all_bets if b["type"] == "KS"])
    print(f"  -> {ks_count} qualifying KS bets")
    for b in sorted(all_bets, key=lambda x: -x["edge"])[:5]:
        print(f"  {b['label'][:50]:50s} edge={b['edge']:.0%} mkt={b['market_prob']:.0%} @ {b['price_cents']}c")

    # === 2. Multi-Leg Parlays (KS only) ===
    print()
    print("  " + "-" * 66)
    print("  2. MULTI-LEG PARLAYS (KS only - info_only markets excluded)")
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
