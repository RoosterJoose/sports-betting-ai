"""Paper-trade the June 10, 2026 portfolio as logged candidates.

Portfolio (from reports/BestBets.md, June 10 section):
  1. Robinson REB 6+  YES @ 36c  — $0.36  (NBA / Kalshi)
  2. Robinson REB 5+  YES @ 49c  — $0.49  (NBA / Kalshi)
  3. McBride  3PT 2+  YES @ 26c  — $0.26  (NBA / Kalshi)
  4. Davis Martin   SO 3.5 OVER  — $2.00  (MLB / PrizePicks)
  5. Chris Sale     H allowed 3.5 OVER — $2.00  (MLB / PrizePicks)
  6. Ian Happ       H+R+RBI 1.5 OVER — $2.00  (MLB / PrizePicks)

Total exposure: $9.11
Status: PAPER (live=0). Will be resolved after the games.

Usage:
    python -m scripts.paper_tonight_jun10
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.trade_tracker import TradeTracker


PORTFOLIO = [
    # ── NBA (Kalshi) ─────────────────────────────────────────────────────────
    {
        "sport": "nba",
        "model_name": "REB",
        "ticker": "KXNBAREBMIT-ROBINSON6+",
        "title": "Mitchell Robinson REB 6+",
        "side": "yes",
        "price_cents": 36,
        "contracts": 1,           # 1 contract × 36c = $0.36
        "model_prob": 0.79,
        "market_prob": 0.36,
        "edge": 0.43,
        "notes": "Game 4 NBA Finals, model says starting C clears 6+ REB at 79% (backtest REB 11/11 100%)",
    },
    {
        "sport": "nba",
        "model_name": "REB",
        "ticker": "KXNBAREBMIT-ROBINSON5+",
        "title": "Mitchell Robinson REB 5+",
        "side": "yes",
        "price_cents": 49,
        "contracts": 1,           # 1 contract × 49c = $0.49
        "model_prob": 0.87,
        "market_prob": 0.49,
        "edge": 0.38,
        "notes": "Game 4 NBA Finals, even lower bar (5+), model 87% confidence",
    },
    {
        "sport": "nba",
        "model_name": "FG3M",
        "ticker": "KXNBA3PTMCBRIDE-2+",
        "title": "Miles McBride 3PT 2+",
        "side": "yes",
        "price_cents": 26,
        "contracts": 1,           # 1 contract × 26c = $0.26
        "model_prob": 0.56,
        "market_prob": 0.27,
        "edge": 0.29,
        "notes": "Game 4 NBA Finals, sniper cleared 2+ 3PT in 4 of last 5 games",
    },
    # ── MLB (PrizePicks) ─────────────────────────────────────────────────────
    # PrizePicks treats entries as dollar-amount picks, not binary contracts.
    # Logged with size=1 and price_cents = entry cost in cents for normalization
    # in the tracker. Payouts on PrizePicks 5/6 flex are 1.5-2x stake.
    {
        "sport": "mlb",
        "model_name": "ALL_SO",
        "ticker": "PP-DAVIS-MARTIN-SO3.5-OVER",
        "title": "Davis Martin SO 3.5 OVER",
        "side": "over",
        "price_cents": 0,         # PrizePicks over/under, no contract price
        "contracts": 200,         # $2.00 entry (× 1 cent = 200 cent-units, normalize via size)
        "model_prob": 0.998,
        "market_prob": 0.542,     # PrizePicks 5/6 flex breakeven
        "edge": 0.456,
        "notes": "PrizePicks paper pick. R²=0.486 SO model, 5/5 backtest pass. Full size (5★).",
    },
    {
        "sport": "mlb",
        "model_name": "ALL_H",
        "ticker": "PP-CHRIS-SALE-H3.5-OVER",
        "title": "Chris Sale H allowed 3.5 OVER",
        "side": "over",
        "price_cents": 0,
        "contracts": 200,
        "model_prob": 0.991,
        "market_prob": 0.542,
        "edge": 0.449,
        "notes": "PrizePicks paper pick. R²=0.487 H model, 5/5 backtest pass. Full size (5★).",
    },
    {
        "sport": "mlb",
        "model_name": "ALL_H_R_RBI",
        "ticker": "PP-IAN-HAPP-HRR1.5-OVER",
        "title": "Ian Happ H+R+RBI 1.5 OVER",
        "side": "over",
        "price_cents": 0,
        "contracts": 200,
        "model_prob": 0.805,
        "market_prob": 0.542,
        "edge": 0.264,
        "notes": "PrizePicks paper pick. R²=0.041 H_R_RBI model, 5/5 backtest pass. 50% size (4★).",
    },
]


def main():
    tracker = TradeTracker()
    print(f"Logging {len(PORTFOLIO)} paper trades for 2026-06-10 portfolio...")
    tracker.log_batch([{**t, "live": False} for t in PORTFOLIO])
    print(f"\nPortfolio summary:")
    print(f"{'#':<3} {'Sport':<5} {'Stat':<11} {'Title':<35} {'Side':<5} {'Edge':>6} {'$':>6}")
    print("-" * 80)
    total = 0
    for i, t in enumerate(PORTFOLIO, 1):
        if t["sport"] == "nba":
            cost = t["contracts"] * t["price_cents"] / 100.0
        else:  # mlb PrizePicks
            cost = t["contracts"] / 100.0  # 200 cent-units = $2.00
        total += cost
        print(f"{i:<3} {t['sport']:<5} {t['model_name']:<11} {t['title'][:35]:<35} "
              f"{t['side']:<5} {t['edge']:>+5.1%} ${cost:>5.2f}")
    print("-" * 80)
    print(f"{'TOTAL EXPOSURE':<60} ${total:>5.2f}")
    print()
    print("All 6 logged with status=pending, live=0 (PAPER).")
    print("After games resolve, run scripts/resolve_paper_trades.py to mark wins/losses.")
    print()
    print(tracker.summary())
    tracker.close()


if __name__ == "__main__":
    main()
