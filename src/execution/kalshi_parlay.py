"""Kalshi Multi-Leg Play Finder.

Finds optimal 2/3/4-leg combinations of individual Kalshi binary markets.
Since Kalshi doesn't offer native parlays, this simulates parlays by
combining independent YES contracts purchased simultaneously.
"""
import re
import itertools
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class KalshiLeg:
    """A single Kalshi binary market opportunity."""
    ticker: str
    title: str
    player_name: str
    game_id: str
    stat_type: str
    market_prob: float  # market-implied probability (price)
    model_prob: float   # our model's probability
    edge: float         # model_prob - market_prob
    price_cents: int    # YES price in cents


@dataclass
class KalshiParlay:
    """A multi-leg combination of Kalshi markets."""
    legs: list[KalshiLeg]
    num_legs: int
    combined_model_prob: float
    combined_market_prob: float
    combined_edge_pct: float
    payout_multiple: float
    expected_value: float
    kelly_fraction: float
    correlation_penalty: float


def extract_player(ticker: str, title: str) -> str:
    """Extract player name from Kalshi ticker or title."""
    # Try from title first (e.g., "José Soriano: 6+ strikeouts?")
    m = re.match(r'^([^:]+):', title)
    if m:
        return m.group(1).strip()
    # Try from ticker: KXMLBKS-26JUN071610LAALAD-LAAJSORIANO59-6
    parts = ticker.split('-')
    for part in parts:
        # Look for team+player pattern
        m2 = re.match(r'[A-Z]{2,4}([A-Z]+)\d+', part)
        if m2:
            return m2.group(1).strip()
    return ticker.split('-')[-2] if len(ticker.split('-')) >= 2 else title[:20]


def extract_game_id(ticker: str) -> str:
    """Extract game identifier from ticker.
    KXMLBKS-26JUN071610LAALAD-LAAJSORIANO59-6 → game key = 26JUN071610LAALAD
    """
    parts = ticker.split('-')
    if len(parts) >= 3:
        return parts[1]
    return ticker


def extract_stat_type(ticker: str, title: str) -> str:
    """Extract stat type from title."""
    m = re.search(r':\s*(\d+[+]?)\s*(.+?)\?', title)
    if m:
        return f"{m.group(2).strip()}"
    return "unknown"


def parse_opportunities(opportunities: list[dict]) -> list[KalshiLeg]:
    """Parse raw opportunities from scanners into KalshiLeg objects."""
    legs = []
    for opp in opportunities:
        ticker = opp.get("ticker", "")
        title = opp.get("title", opp.get("label", ""))
        market_prob = opp.get("market_prob", opp.get("mkt_yes", 0))
        model_prob = opp.get("model_prob", opp.get("p_yes", 0))
        edge = opp.get("edge", model_prob - market_prob)
        price_cents = opp.get("price_cents", int(market_prob * 100))

        if not ticker or market_prob <= 0 or model_prob <= 0:
            continue

        leg = KalshiLeg(
            ticker=ticker,
            title=str(title),
            player_name=extract_player(ticker, str(title)),
            game_id=extract_game_id(ticker),
            stat_type=extract_stat_type(ticker, str(title)),
            market_prob=market_prob,
            model_prob=model_prob,
            edge=edge,
            price_cents=price_cents,
        )
        legs.append(leg)
    return legs


class KalshiParlayFinder:
    """Finds optimal multi-leg combinations of Kalshi binary markets."""

    # Same player in multiple legs → correlation penalty
    SAME_PLAYER_PENALTY = 0.3
    # Same game (different players) → moderate correlation
    SAME_GAME_PENALTY = 0.15
    # Same stat type across different games → slight correlation
    SAME_STAT_PENALTY = 0.05

    def __init__(self, min_edge: float = 0.05, max_legs: int = 4, kc=None):
        self.min_edge = min_edge
        self.max_legs = max_legs
        self.min_legs = 2
        self.kc = kc

    def find_best(self, opportunities: list[dict], top_n: int = 10) -> dict[int, list[KalshiParlay]]:
        """Find best parlays for each leg count (2, 3, 4)."""
        legs = parse_opportunities(opportunities)

        # Filter by min edge
        legs = [l for l in legs if l.edge >= self.min_edge]

        # Deduplicate: for same player + same stat + same line, keep highest edge
        seen = {}
        for l in legs:
            key = (l.player_name.lower(), l.stat_type.lower(), l.ticker.split('-')[-1])
            if key not in seen or l.edge > seen[key].edge:
                seen[key] = l
        legs = list(seen.values())

        if len(legs) < 2:
            return {}

        results = {}
        for n_legs in range(self.min_legs, min(self.max_legs, len(legs)) + 1):
            parlays = self._find_for_leg_count(legs, n_legs)
            if parlays:
                results[n_legs] = parlays[:top_n]

        return results

    def place_best(self, opportunities: list[dict], top_parlays: int = 3, bankroll: float = 10.0) -> list[dict]:
        """Find best parlays and buy each leg as individual YES contracts."""
        if self.kc is None:
            from src.data.kalshi import KalshiClient
            self.kc = KalshiClient()
        
        results = self.find_best(opportunities, top_n=top_parlays)
        placed = []
        
        for n_legs in sorted(results.keys()):
            for parlay in results[n_legs][:top_parlays]:
                # Skip if any legs were already bought
                for leg in parlay.legs:
                    from src.data.kalshi import KalshiClient
                    price_cents = leg.price_cents
                    if price_cents < 1 or price_cents > 99:
                        continue
                    contracts = max(1, int(bankroll * 0.1 / (price_cents / 100.0)))
                    if contracts < 1:
                        continue
                    try:
                        resp = self.kc.create_order(
                            ticker=leg.ticker, side="yes",
                            yes_price=price_cents, count=str(contracts)
                        )
                        placed.append({
                            "ticker": leg.ticker,
                            "player": leg.player_name,
                            "parlay_legs": n_legs,
                            "price_cents": price_cents,
                            "contracts": contracts,
                            "cost": contracts * price_cents / 100.0,
                            "response": resp.get("order_id", "?") if resp else "failed",
                        })
                    except Exception as e:
                        placed.append({"ticker": leg.ticker, "error": str(e)})
        return placed

    def _find_for_leg_count(self, legs: list[KalshiLeg], n: int) -> list[KalshiParlay]:
        """Find best N-leg parlays."""
        candidates = []

        for combo in itertools.combinations(legs, n):
            correlation = self._estimate_correlation(combo)
            if correlation >= 1.0:  # Too correlated, skip
                continue

            combined_market = np.prod([l.market_prob for l in combo])
            combined_model = np.prod([l.model_prob for l in combo])
            payout = 1.0 / combined_market if combined_market > 0 else 0

            # Apply correlation penalty to model probability
            adjusted_model = combined_model * (1.0 - correlation)

            # Combined edge
            if combined_market > 0:
                combined_edge = (adjusted_model - combined_market) / combined_market
            else:
                combined_edge = 0

            # Expected value per $1 risked
            ev = adjusted_model * payout - 1.0

            # Kelly fraction for parlay stake sizing
            # For a parlay: f* = (p * b - q) / b where b = payout - 1
            if payout > 1:
                q = 1.0 - adjusted_model
                kelly = (adjusted_model * (payout - 1) - q) / (payout - 1)
            else:
                kelly = 0

            if combined_edge > 0 and ev > 0:
                candidates.append(KalshiParlay(
                    legs=list(combo),
                    num_legs=n,
                    combined_model_prob=adjusted_model,
                    combined_market_prob=combined_market,
                    combined_edge_pct=round(combined_edge * 100, 2),
                    payout_multiple=round(payout, 2),
                    expected_value=round(ev, 4),
                    kelly_fraction=round(max(0, kelly), 4),
                    correlation_penalty=round(correlation, 3),
                ))

        candidates.sort(key=lambda p: p.expected_value, reverse=True)
        return candidates

    def _estimate_correlation(self, combo: tuple[KalshiLeg, ...]) -> float:
        """Estimate correlation between legs. Returns 0 (independent) to 1 (perfectly correlated)."""
        n = len(combo)
        if n < 2:
            return 0.0

        penalties = []
        for i in range(n):
            for j in range(i + 1, n):
                a, b = combo[i], combo[j]

                # Same player → near-perfect correlation, hard reject
                if a.player_name.lower() == b.player_name.lower():
                    return 1.0  # Hard reject - same player always correlated
                # Same game (same game_id prefix) → moderate
                elif a.game_id == b.game_id:
                    penalties.append(self.SAME_GAME_PENALTY)
                # Same stat type across different games → slight
                elif a.stat_type.lower() == b.stat_type.lower():
                    penalties.append(self.SAME_STAT_PENALTY)
                else:
                    penalties.append(0.0)

        return np.mean(penalties) if penalties else 0.0


def format_parlay(parlay: KalshiParlay, index: int = 0) -> str:
    """Format a parlay for display."""
    lines = []
    lines.append(f"\n  #{index+1}: {parlay.num_legs}-leg | EV={parlay.expected_value:+.2f} | Edge={parlay.combined_edge_pct:+.1f}% | Payout={parlay.payout_multiple:.1f}x | Kelly={parlay.kelly_fraction:.1%}")
    lines.append(f"  {'─'*70}")
    for i, leg in enumerate(parlay.legs):
        lines.append(f"  Leg {i+1}: {leg.player_name:25s} {leg.title[:40]:40s}")
        lines.append(f"          mkt={leg.market_prob:.0%} model={leg.model_prob:.0%} edge={leg.edge:+.0%} @ {leg.price_cents}¢")
    lines.append(f"  Correlation penalty: {parlay.correlation_penalty:.1%}")
    return '\n'.join(lines)


def display_parlays(results: dict[int, list[KalshiParlay]], title: str = "Kalshi Multi-Leg Plays"):
    """Display parlay results grouped by leg count."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

    if not results:
        print("  No qualifying parlays found (need at least 2 edges)")
        return

    for n_legs in sorted(results.keys()):
        parlays = results[n_legs]
        print(f"\n{'─'*70}")
        print(f"  BEST {n_legs}-LEG PLAYS ({len(parlays)} found)")
        print(f"{'─'*70}")
        for i, parlay in enumerate(parlays[:5]):
            print(format_parlay(parlay, i))

    print()
