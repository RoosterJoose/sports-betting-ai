#!/usr/bin/env python3
"""Paper-trade tracker for the 14 WC 2026 first-round picks (13 ML + 1 draw).

Logs the picks from reports/BestBets.md into the trade tracker with full
metadata: source (model + research consensus), implied odds, edge estimate,
and research reasoning. Provides a resolution method to settle each pick
after the game ends, and a report method to compare settled outcomes to
the model's predicted win rate after the first round ends.

The 14 picks are the STRONG PLAYS table from BestBets.md (13 ML + 1 draw).
Edge estimates are research-based (the model probs aren't all explicit in
the doc) — the report section computes the actual win rate and compares
to the model-implied win rate.

Usage:
    # Log the 14 picks (idempotent — safe to re-run, UNIQUE on ticker+timestamp)
    python -m src.scripts.paper_trade_wc_2026_picks --log

    # Resolve a specific match after the game ends
    python -m src.scripts.paper_trade_wc_2026_picks --resolve "Mexico vs South Africa" won
    python -m src.scripts.paper_trade_wc_2026_picks --resolve "Ivory Coast vs Ecuador" won
    # (for the draw pick, "won" means the match ended in a draw)

    # Bulk-resolve a whole date
    python -m src.scripts.paper_trade_wc_2026_picks --resolve-date 2026-06-11

    # Show the report: settled vs model-implied win rate
    python -m src.scripts.paper_trade_wc_2026_picks --report

    # Show all picks (pending + settled)
    python -m src.scripts.paper_trade_wc_2026_picks --list
"""
import sys, json, argparse, warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.trade_tracker import TradeTracker

# ── The 14 picks (from reports/BestBets.md "STRONG PLAYS" table) ────────
# Each pick: date, match, pick, model_prob, market_prob (implied odds), source, reasoning
# model_prob is the research-based estimate; market_prob is the implied prob from
# the research-cited odds. For the Scotland pick, the model_prob is explicit (58.9%).
# For Germany (-800), the model_prob equals market_prob (no edge, but research Strong).
PICKS = [
    # ── Jun 11 ──
    {
        "date": "2026-06-11",
        "match": "Mexico vs South Africa",
        "venue": "Mexico City",
        "pick": "Mexico ML",
        "model_prob": 0.70,
        "market_prob": 0.60,   # -150 implied
        "implied_odds": "-150",
        "source": "model + research consensus (both researchers: Mexico Strong)",
        "edge_pct": 16.7,       # (0.70 - 0.60) / 0.60 * 100
        "edge_pp": 10.0,        # 0.70 - 0.60
        "reasoning": "7,349 ft altitude tax; home opener; both researchers Strong",
        "notes": "Used in 4-leg play; altitude tax is the dominant factor",
    },
    {
        "date": "2026-06-11",
        "match": "South Korea vs Czechia",
        "venue": "Guadalajara",
        "pick": "Czechia ML",
        "model_prob": 0.65,
        "market_prob": 0.545,  # -120 implied
        "implied_odds": "-120",
        "source": "research consensus (Czechia Moderate: European tactical discipline)",
        "edge_pct": 19.3,
        "edge_pp": 10.5,
        "reasoning": "European tactical discipline vs Asian counter-attacking style",
        "notes": "Used in 4-leg play",
    },
    # ── Jun 12 ──
    {
        "date": "2026-06-12",
        "match": "USA vs Paraguay",
        "venue": "Inglewood",
        "pick": "USA ML",
        "model_prob": 0.75,
        "market_prob": 0.667,  # -200 implied
        "implied_odds": "-200",
        "source": "research consensus (USA Strong: home pressing)",
        "edge_pct": 12.4,
        "edge_pp": 8.3,
        "reasoning": "Home opener for the US; pressing style suits altitude-neutral LA venue",
        "notes": "Home opener = high motivation",
    },
    # ── Jun 13 ──
    {
        "date": "2026-06-13",
        "match": "Brazil vs Morocco",
        "venue": "Miami",
        "pick": "Brazil ML",
        "model_prob": 0.80,
        "market_prob": 0.75,   # -300 implied
        "implied_odds": "-300",
        "source": "research consensus (Brazil Strong: talent gap)",
        "edge_pct": 6.7,
        "edge_pp": 5.0,
        "reasoning": "Talent gap; Brazil's squad depth overwhelms Morocco",
        "notes": "Talent-gap play, not model-driven",
    },
    {
        "date": "2026-06-13",
        "match": "Haiti vs Scotland",
        "venue": "Boston",
        "pick": "Scotland ML",
        "model_prob": 0.589,  # EXPLICIT in doc
        "market_prob": 0.145,  # +500 implied (longshot)
        "implied_odds": "+500",
        "source": "MODEL + RESEARCH AGREE",
        "edge_pct": 306.2,     # massive edge
        "edge_pp": 44.4,
        "reasoning": "Only pick where model and research both agree; model 58.9% vs market 14.5%",
        "notes": "FLAG: huge +306% edge — verify no line-movement issue before relying on it",
    },
    # ── Jun 14 ──
    {
        "date": "2026-06-14",
        "match": "Germany vs Curaçao",
        "venue": "Houston",
        "pick": "Germany ML",
        "model_prob": 0.89,
        "market_prob": 0.889,  # -800 implied
        "implied_odds": "-800",
        "source": "research consensus (Germany Strong)",
        "edge_pct": 0.1,        # essentially no edge, but research Strong
        "edge_pp": 0.1,
        "reasoning": "Heavy favorite; no model edge but research-confirmed favorite",
        "notes": "No edge vs market — research-only pick",
    },
    {
        "date": "2026-06-14",
        "match": "Ivory Coast vs Ecuador",
        "venue": "Philadelphia",
        "pick": "Draw",
        "model_prob": 0.35,    # draw is typically 25-35% in even matches
        "market_prob": 0.286,  # +250 implied
        "implied_odds": "+250",
        "source": "research consensus (both researchers lean Draw)",
        "edge_pct": 22.4,
        "edge_pp": 6.4,
        "reasoning": "Model phantom +256% Ecuador — research overrules; both researchers lean Draw",
        "notes": "Model had phantom edge on Ecuador; research overrules in favor of draw",
    },
    # ── Jun 15 ──
    {
        "date": "2026-06-15",
        "match": "Spain vs Cape Verde",
        "venue": "Atlanta",
        "pick": "Spain ML",
        "model_prob": 0.80,
        "market_prob": 0.75,
        "implied_odds": "-300",
        "source": "research consensus (Spain Strong: possession dominance)",
        "edge_pct": 6.7,
        "edge_pp": 5.0,
        "reasoning": "Possession dominance; Spain's tiki-taka overwhelms minnow",
        "notes": "Talent-gap play",
    },
    {
        "date": "2026-06-15",
        "match": "Belgium vs Egypt",
        "venue": "Kansas City",
        "pick": "Belgium ML",
        "model_prob": 0.70,
        "market_prob": 0.667,
        "implied_odds": "-200",
        "source": "research consensus (Belgium Strong: vet core)",
        "edge_pct": 5.0,
        "edge_pp": 3.3,
        "reasoning": "Vet core; Belgium's golden generation still has 1-2 tournaments left",
        "notes": "Squad-depth play",
    },
    {
        "date": "2026-06-15",
        "match": "Saudi Arabia vs Uruguay",
        "venue": "Miami",
        "pick": "Uruguay ML",
        "model_prob": 0.70,
        "market_prob": 0.667,
        "implied_odds": "-200",
        "source": "research consensus (Uruguay Strong: physical, Elo 73)",
        "edge_pct": 5.0,
        "edge_pp": 3.3,
        "reasoning": "Physical; model had Saudi +332% phantom — Elo 73 says no",
        "notes": "Model phantom on Saudi; research overrules",
    },
    # ── Jun 16 ──
    {
        "date": "2026-06-16",
        "match": "France vs Senegal",
        "venue": "New York",
        "pick": "France ML",
        "model_prob": 0.80,
        "market_prob": 0.75,
        "implied_odds": "-300",
        "source": "research consensus (both researchers: France R1 Strong, R2 Moderate)",
        "edge_pct": 6.7,
        "edge_pp": 5.0,
        "reasoning": "Depth; France's squad is 2-deep at every position",
        "notes": "Squad-depth play",
    },
    {
        "date": "2026-06-16",
        "match": "Argentina vs Algeria",
        "venue": "Miami",
        "pick": "Argentina ML",
        "model_prob": 0.80,
        "market_prob": 0.75,
        "implied_odds": "-300",
        "source": "research consensus (Argentina Strong: title defense)",
        "edge_pct": 6.7,
        "edge_pp": 5.0,
        "reasoning": "Title defense; Messi-era core still motivated",
        "notes": "Title-defense motivation",
    },
    # ── Jun 17 ──
    {
        "date": "2026-06-17",
        "match": "Portugal vs DR Congo",
        "venue": "Atlanta",
        "pick": "Portugal ML",
        "model_prob": 0.80,
        "market_prob": 0.75,
        "implied_odds": "-300",
        "source": "research consensus (Portugal Strong: firepower)",
        "edge_pct": 6.7,
        "edge_pp": 5.0,
        "reasoning": "Firepower; Ronaldo's last WC + Bruno/Silva generation",
        "notes": "Firepower play",
    },
    {
        "date": "2026-06-17",
        "match": "Uzbekistan vs Colombia",
        "venue": "Houston",
        "pick": "Colombia ML",
        "model_prob": 0.65,
        "market_prob": 0.60,   # -150 implied
        "implied_odds": "-150",
        "source": "research consensus (both researchers: Colombia Strong — most altitude-acclimated team)",
        "edge_pct": 8.3,
        "edge_pp": 5.0,
        "reasoning": "Altitude-acclimated; Colombia's players are from high-altitude cities",
        "notes": "Altitude-acclimation play",
    },
]


def make_ticker(pick: dict) -> str:
    """Generate a unique ticker for the trade tracker's UNIQUE constraint."""
    # Format: WC2026-YYYYMMDD-MATCHSLUG
    date_compact = pick["date"].replace("-", "")
    # Slugify match: lowercase, replace spaces and punctuation
    slug = pick["match"].lower()
    slug = slug.replace(" vs ", "-").replace(".", "").replace("'", "")
    return f"WC2026-{date_compact}-{slug}"


def make_title(pick: dict) -> str:
    """Generate a human-readable title for the trade tracker."""
    return f"{pick['date']} {pick['match']} \u2014 {pick['pick']}"


def make_notes(pick: dict) -> str:
    """Pack all metadata into the notes field for downstream consumption."""
    return (f"source: {pick['source']}; "
            f"implied_odds: {pick['implied_odds']}; "
            f"edge_pct: {pick['edge_pct']:.1f}%; "
            f"reasoning: {pick['reasoning']}; "
            f"extra: {pick['notes']}")


def log_all(tt: TradeTracker):
    """Log all 14 picks to the trade tracker. Truly idempotent (pre-checks ticker)."""
    print(f"\n{'='*70}")
    print(f"  LOGGING {len(PICKS)} WC 2026 FIRST-ROUND PICKS")
    print(f"{'='*70}")
    logged = 0
    skipped = 0
    for pick in PICKS:
        ticker = make_ticker(pick)
        title = make_title(pick)
        notes = make_notes(pick)
        # Pre-check: skip if ticker already exists (true idempotency, not
        # relying on UNIQUE(ticker, timestamp) which doesn't fire when
        # timestamps differ between runs)
        existing = tt._conn.execute(
            "SELECT 1 FROM trades WHERE ticker=? LIMIT 1", (ticker,)
        ).fetchone()
        if existing:
            skipped += 1
            print(f"  \u2192 {ticker}: already logged (skipped)")
            continue
        # side: "yes" for ML picks (we're betting the favorite to win)
        # For the draw pick, "yes" means we bet on a draw outcome
        side = "yes"
        # price_cents: implied market prob as cents
        price_cents = int(round(pick["market_prob"] * 100))
        # size: 1 contract per pick (paper trade)
        size = 1
        edge = pick["model_prob"] - pick["market_prob"]
        tt.log_trade(
            sport="worldcup",
            model_name="WC_2026_R1",
            ticker=ticker,
            title=title,
            side=side,
            price_cents=price_cents,
            size=size,
            model_prob=pick["model_prob"],
            market_prob=pick["market_prob"],
            edge=edge,
            live=False,
            notes=notes,
        )
        logged += 1
        print(f"  \u2713 {ticker}: {pick['pick']} ({pick['implied_odds']}) "
              f"model={pick['model_prob']:.0%} mkt={pick['market_prob']:.0%} "
              f"edge=+{edge:.1%}")
    print(f"\n  Logged: {logged}  |  Skipped (already exists): {skipped}")


def resolve_pick(tt: TradeTracker, match: str, status: str):
    """Resolve a single pick by match name. status: 'won' or 'lost'.

    For the draw pick (Ivory Coast vs Ecuador), 'won' means the match ended
    in a draw. For ML picks, 'won' means the favorite won.
    """
    ticker = None
    pick_info = None
    for p in PICKS:
        if p["match"].lower() == match.lower():
            ticker = make_ticker(p)
            pick_info = p
            break
    if not ticker:
        print(f"  \u2717 No pick found for match: {match}")
        return
    # For ML picks: resolved_price = 1.0 if won (favorite won outright), 0.0 if lost
    # For draw pick: caller passes "won" only if the match actually ended in a draw
    resolved_price = 1.0 if status == "won" else 0.0
    pnl = tt.resolve_trade(ticker, resolved_price, status)
    print(f"  \u2713 {ticker}: {status} (resolved_price={resolved_price}, pnl=${pnl:+.2f})")


def resolve_date(tt: TradeTracker, date_str: str):
    """Bulk-resolve all picks for a given date. Prints prompts for each."""
    print(f"\n  Resolving all picks for {date_str}:")
    for pick in PICKS:
        if pick["date"] == date_str:
            print(f"\n  {pick['match']} ({pick['pick']}):")
            print(f"    reasoning: {pick['reasoning']}")
            # Don't auto-resolve — user must confirm
            print(f"    \u2192 Use --resolve \"{pick['match']}\" won|lost to settle")


def list_picks(tt: TradeTracker):
    """List all WC picks with current status."""
    import sqlite3
    conn = sqlite3.connect(str(tt._conn.execute("PRAGMA database_list").fetchone()[0][2] if False else "data/trade_tracker.db"))
    # Simpler: use the trade tracker's get_analytics
    analytics = tt.get_analytics(sport="worldcup", min_sample=0)
    if not analytics:
        print("\n  No WC picks logged yet. Run with --log first.")
        return
    print(f"\n{'='*70}")
    print(f"  WC 2026 PICKS — STATUS")
    print(f"{'='*70}")
    # Get all trades for worldcup
    rows = tt._conn.execute("""
        SELECT ticker, title, side, price_cents, model_prob, market_prob, edge, status, pnl
        FROM trades WHERE sport='worldcup' ORDER BY id
    """).fetchall()
    if not rows:
        print("  No picks logged.")
        return
    print(f"\n  {'Ticker':45s} {'Pick':12s} {'Model':>6s} {'Mkt':>5s} {'Edge':>6s} {'Status':>8s} {'PnL':>7s}")
    print(f"  {'-'*45} {'-'*12} {'-'*6} {'-'*5} {'-'*6} {'-'*8} {'-'*7}")
    for r in rows:
        ticker, title, side, price_cents, model_prob, market_prob, edge, status, pnl = r
        # Extract pick from title
        pick_name = title.split(" \u2014 ")[-1] if " \u2014 " in title else title
        edge_str = f"{edge*100:+.1f}%" if edge is not None else "N/A"
        pnl_str = f"${pnl:+.2f}" if pnl is not None else "$0.00"
        print(f"  {ticker:45s} {pick_name:12s} {model_prob*100:5.0f}% {market_prob*100:4.0f}% "
              f"{edge_str:>6s} {status:>8s} {pnl_str:>7s}")


def report(tt: TradeTracker):
    """Compare settled outcomes to the model's actual win rate.

    Computes:
    - Actual win rate (settled trades only)
    - Model-implied win rate (weighted by model_prob)
    - Brier score on settled trades
    - Calibration by decile
    - Edge-weighted PnL (what the model said the edge was)
    """
    print(f"\n{'='*70}")
    print(f"  WC 2026 PICKS — POST-ROUND REPORT")
    print(f"{'='*70}")

    # Get all settled trades
    rows = tt._conn.execute("""
        SELECT ticker, title, model_prob, market_prob, edge, status, pnl
        FROM trades
        WHERE sport='worldcup' AND status IN ('won', 'lost')
        ORDER BY id
    """).fetchall()
    pending = tt._conn.execute("""
        SELECT COUNT(*) FROM trades WHERE sport='worldcup' AND status='pending'
    """).fetchone()[0]
    total_logged = tt._conn.execute("""
        SELECT COUNT(*) FROM trades WHERE sport='worldcup'
    """).fetchone()[0]

    print(f"\n  Total logged: {total_logged}  |  Settled: {len(rows)}  |  Pending: {pending}")

    if not rows:
        print("\n  No settled trades yet. Use --resolve to settle matches as they finish.")
        return

    # Actual stats
    n_settled = len(rows)
    n_wins = sum(1 for r in rows if r[5] == "won")
    actual_wr = n_wins / n_settled
    total_pnl = sum(r[6] for r in rows if r[6] is not None)
    # True volume: query SUM(price_cents * size) / 100.0 directly from DB
    vol_row = tt._conn.execute("""
        SELECT COALESCE(SUM(price_cents * size), 0) / 100.0
        FROM trades WHERE sport='worldcup' AND status IN ('won', 'lost')
    """).fetchone()
    total_volume = float(vol_row[0]) if vol_row and vol_row[0] else 0.0
    actual_roi = (total_pnl / total_volume * 100) if total_volume > 0 else 0

    # Model-implied stats (what the model said would happen)
    model_implied_wr = sum(r[2] for r in rows) / n_settled  # avg model_prob
    model_implied_pnl = sum((r[2] - r[3]) for r in rows)  # sum of (model_prob - market_prob)
    # Brier score
    brier = sum((r[2] - (1 if r[5] == "won" else 0)) ** 2 for r in rows) / n_settled
    # Naive Brier (always predict base rate)
    base_rate = n_wins / n_settled
    naive_brier = base_rate * (1 - base_rate)

    print(f"\n  {'Metric':30s} {'Actual':>10s} {'Model-implied':>15s} {'Gap':>10s}")
    print(f"  {'-'*30} {'-'*10} {'-'*15} {'-'*10}")
    print(f"  {'Win rate':30s} {actual_wr*100:>9.1f}% {model_implied_wr*100:>14.1f}% "
          f"{(actual_wr-model_implied_wr)*100:>+9.1f}pp")
    print(f"  {'Brier score':30s} {brier:>10.4f} {naive_brier:>15.4f} "
          f"{(brier-naive_brier):>+10.4f}")
    print(f"  {'Total PnL':30s} ${total_pnl:>9.2f} ${model_implied_pnl*100:>14.2f} "
          f"${total_pnl - model_implied_pnl*100:>+9.2f}")

    # Per-pick breakdown
    print(f"\n  Per-pick breakdown:")
    print(f"  {'Match':35s} {'Pick':>8s} {'Model':>6s} {'Actual':>7s} {'Result':>8s} {'PnL':>7s}")
    print(f"  {'-'*35} {'-'*8} {'-'*6} {'-'*7} {'-'*8} {'-'*7}")
    for r in rows:
        ticker, title, model_prob, market_prob, edge, status, pnl = r
        match = title.split(" \u2014 ")[0].split(" ", 1)[1] if " \u2014 " in title else title
        pick = title.split(" \u2014 ")[-1] if " \u2014 " in title else "?"
        actual = 1 if status == "won" else 0
        pnl_str = f"${pnl:+.2f}" if pnl is not None else "$0.00"
        print(f"  {match:35s} {pick:>8s} {model_prob*100:5.0f}% {actual:>6.0%} "
              f"{status:>8s} {pnl_str:>7s}")

    # Calibration by bucket
    print(f"\n  Calibration by model-prob bucket:")
    print(f"  {'Bucket':>12s} {'N':>3s} {'Model avg':>10s} {'Actual WR':>10s} {'Gap':>8s}")
    print(f"  {'-'*12} {'-'*3} {'-'*10} {'-'*10} {'-'*8}")
    buckets = [(0.0, 0.5, "<50%"), (0.5, 0.7, "50-70%"), (0.7, 0.9, "70-90%"), (0.9, 1.01, "90%+")]
    for lo, hi, label in buckets:
        bucket_rows = [r for r in rows if lo <= r[2] < hi]
        if not bucket_rows:
            continue
        n = len(bucket_rows)
        model_avg = sum(r[2] for r in bucket_rows) / n
        actual_wr_bucket = sum(1 for r in bucket_rows if r[5] == "won") / n
        gap = actual_wr_bucket - model_avg
        print(f"  {label:>12s} {n:>3d} {model_avg*100:>9.1f}% {actual_wr_bucket*100:>9.1f}% "
              f"{gap*100:>+7.1f}pp")

    # Verdict
    print(f"\n  {'='*66}")
    print(f"  VERDICT")
    print(f"  {'='*66}")
    if actual_wr >= model_implied_wr * 0.9:
        verdict = "\u2705 Model is calibrated (actual WR within 10% of model-implied)"
    elif actual_wr >= model_implied_wr * 0.7:
        verdict = "\u26a0\ufe0f Model is slightly overconfident (actual WR 70-90% of model-implied)"
    else:
        verdict = "\u274c Model is severely overconfident (actual WR < 70% of model-implied)"
    print(f"\n  {verdict}")
    print(f"  Actual win rate: {actual_wr*100:.1f}%")
    print(f"  Model-implied win rate: {model_implied_wr*100:.1f}%")
    print(f"  Calibration gap: {(actual_wr-model_implied_wr)*100:+.1f}pp")
    if brier < naive_brier:
        print(f"  Brier beats naive: YES (Brier={brier:.4f} < naive={naive_brier:.4f})")
    else:
        print(f"  Brier beats naive: NO (Brier={brier:.4f} >= naive={naive_brier:.4f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", action="store_true",
                        help="Log all 14 WC picks to the trade tracker (idempotent)")
    parser.add_argument("--resolve", nargs=2, metavar=("MATCH", "STATUS"),
                        help="Resolve a pick: --resolve 'Mexico vs South Africa' won")
    parser.add_argument("--show-date", metavar="DATE",
                        help="Show all picks for a date (use --resolve to settle each)")
    parser.add_argument("--report", action="store_true",
                        help="Show post-round report: actual vs model-implied win rate")
    parser.add_argument("--list", action="store_true",
                        help="List all WC picks with current status")
    args = parser.parse_args()

    if not any([args.log, args.resolve, args.show_date, args.report, args.list]):
        parser.print_help()
        return

    tt = TradeTracker()
    try:
        if args.log:
            log_all(tt)
        if args.resolve:
            match, status = args.resolve
            if status not in ("won", "lost"):
                print(f"  \u2717 Status must be 'won' or 'lost', got: {status}")
                return
            resolve_pick(tt, match, status)
        if args.show_date:
            resolve_date(tt, args.show_date)
        if args.report:
            report(tt)
        if args.list:
            list_picks(tt)
    finally:
        tt.close()


if __name__ == "__main__":
    main()
