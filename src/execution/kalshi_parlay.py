"""Kalshi Multi-Leg Play Finder.

Finds optimal 2/3/4-leg combinations of individual Kalshi binary markets.
Since Kalshi doesn't offer native parlays, this simulates parlays by
combining independent YES contracts purchased simultaneously.

Uses data-driven stat correlations from parlay_correlation.py for accurate
joint probability estimation and Kelly sizing.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.execution.parlay_correlation import (
    SAME_PLAYER_SAME_STAT,
    build_correlation_pairs,
    compute_payout,
    joint_probability,
    market_joint_probability,
    parlay_kelly,
)


@dataclass
class KalshiLeg:
    """A single Kalshi binary market opportunity."""
    ticker: str
    title: str
    player_name: str
    game_id: str
    market_type: str           # e.g. "KS", "HR", "TB", "HRR"
    position: str              # "pitcher" or "hitter"
    market_prob: float         # market-implied probability (price)
    model_prob: float          # our model's probability
    edge: float                # model_prob - market_prob
    price_cents: int           # YES price in cents


@dataclass
class KalshiParlay:
    """A multi-leg combination of Kalshi markets."""
    legs: list[KalshiLeg]
    num_legs: int
    combined_model_prob: float       # correlation-adjusted joint probability
    combined_market_prob: float      # market-implied joint prob (also correlation-adjusted)
    payout_multiple: float           # $ payout per $1 stake if all hit
    expected_value: float            # EV per $1 stake
    kelly_fraction: float            # fraction of bankroll to risk (quarter-Kelly)
    correlation_pairs: list[tuple[int, int, float]]  # (i, j, rho) used
    implied_correlation: float       # average |rho| across all pairs
    joint_independent_prob: float    # product of individual probs (for comparison)


def extract_player(ticker: str, title: str) -> str:
    """Extract player name from Kalshi ticker or title."""
    m = re.match(r'^([^:]+):', title)
    if m:
        return m.group(1).strip()
    parts = ticker.split('-')
    for part in parts:
        m2 = re.match(r'[A-Z]{2,4}([A-Z]+)\d+', part)
        if m2:
            return m2.group(1).strip()
    return parts[-2] if len(parts) >= 2 else title[:20]


def extract_game_id(ticker: str) -> str:
    """Extract game identifier from ticker.

    KXMLBKS-26JUN071610LAALAD-LAAJSORIANO59-6  →  26JUN071610LAALAD
    """
    parts = ticker.split('-')
    if len(parts) >= 3:
        return parts[1]
    return ticker


def extract_position(market_type: str) -> str:
    """Determine position from market type name."""
    from src.execution.parlay_correlation import PITCHER_STATS, HITTER_STATS
    if market_type in PITCHER_STATS:
        return "pitcher"
    if market_type in HITTER_STATS:
        return "hitter"
    return "unknown"


def parse_opportunities(opportunities: list[dict]) -> list[KalshiLeg]:
    """Parse raw opportunities from scanners into KalshiLeg objects."""
    legs = []
    for opp in opportunities:
        ticker = opp.get("ticker", "")
        title = opp.get("title", opp.get("label", ""))
        market_type = opp.get("type", opp.get("stat_type", ""))
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
            market_type=market_type,
            position=extract_position(market_type),
            market_prob=market_prob,
            model_prob=model_prob,
            edge=edge,
            price_cents=price_cents,
        )
        legs.append(leg)
    return legs


class KalshiParlayFinder:
    """Finds optimal multi-leg combinations of Kalshi binary markets.

    Uses data-driven stat correlations, proper joint probability via
    covariance expansion, and quarter-Kelly sizing.
    """

    def __init__(self, min_edge: float = 0.05, max_legs: int = 4, kc=None):
        self.min_edge = min_edge
        self.max_legs = max_legs
        self.min_legs = 2
        self.kc = kc

    # Max legs to consider for combination generation (performance guard)
    MAX_COMBO_LEGS = 80

    def find_best(self, opportunities: list[dict], top_n: int = 10
                  ) -> dict[int, list[KalshiParlay]]:
        """Find best parlays for each leg count (2, 3, 4)."""
        legs = parse_opportunities(opportunities)

        # Filter by min edge
        legs = [l for l in legs if l.edge >= self.min_edge]

        # Deduplicate: for same player + same stat + same line, keep highest edge
        seen = {}
        for l in legs:
            line_val = l.ticker.split('-')[-1]
            key = (l.player_name.lower(), l.market_type.lower(), line_val)
            if key not in seen or l.edge > seen[key].edge:
                seen[key] = l
        legs = list(seen.values())

        # ── Performance guard: pre-filter to top N legs by edge ──────────
        # C(80, 2) = 3,160, C(80, 3) = 82,160, C(80, 4) = 1.58M — tractable
        # Without filtering, 400+ legs produce C(400, 4) ≈ 1.6B — impossible
        if len(legs) > self.MAX_COMBO_LEGS:
            legs.sort(key=lambda l: l.edge, reverse=True)
            legs = legs[:self.MAX_COMBO_LEGS]

        if len(legs) < 2:
            return {}

        results = {}
        for n_legs in range(self.min_legs, min(self.max_legs, len(legs)) + 1):
            parlays = self._find_for_leg_count(legs, n_legs)
            if parlays:
                results[n_legs] = parlays[:top_n]

        return results

    def place_best(self, opportunities: list[dict], top_parlays: int = 3,
                   bankroll: float = 10.0) -> list[dict]:
        """Find best parlays and buy each leg as individual YES contracts.

        Sizes the total parlay stake using the correlation-adjusted Kelly,
        then divides equally across legs.
        """
        if self.kc is None:
            from src.data.kalshi import KalshiClient
            self.kc = KalshiClient()

        results = self.find_best(opportunities, top_n=top_parlays)
        placed = []

        for n_legs in sorted(results.keys()):
            for parlay in results[n_legs][:top_parlays]:
                # Total parlay stake from Kelly
                stake_pct = parlay.kelly_fraction
                total_stake = bankroll * stake_pct
                # Divide across legs (equal allocation)
                stake_per_leg = total_stake / max(len(parlay.legs), 1)

                for leg in parlay.legs:
                    price = leg.price_cents / 100.0
                    if price <= 0 or price >= 1:
                        continue
                    contracts = max(1, int(stake_per_leg / price))
                    if contracts < 1:
                        continue
                    try:
                        resp = self.kc.create_order(
                            ticker=leg.ticker, side="yes",
                            yes_price=leg.price_cents, count=str(contracts)
                        )
                        placed.append({
                            "ticker": leg.ticker,
                            "player": leg.player_name,
                            "parlay_legs": n_legs,
                            "price_cents": leg.price_cents,
                            "contracts": contracts,
                            "cost": contracts * price,
                            "response": resp.get("order_id", "?") if resp else "failed",
                        })
                    except Exception as e:
                        placed.append({"ticker": leg.ticker, "error": str(e)})
        return placed

    # ── Internal ────────────────────────────────────────────────────────

    def _find_for_leg_count(self, legs: list[KalshiLeg], n: int) -> list[KalshiParlay]:
        """Find best N-leg parlays using correlation-adjusted probability."""
        candidates = []

        for combo in itertools.combinations(legs, n):
            combo_list = list(combo)

            # ── Detect same-player same-stat (perfect correlation) ──────
            skip = False
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = combo_list[i], combo_list[j]
                    if (a.player_name.lower() == b.player_name.lower()
                            and a.market_type == b.market_type):
                        skip = True
                        break
                if skip:
                    break
            if skip:
                continue

            # ── Build correlation pairs ─────────────────────────────────
            pairs = build_correlation_pairs(combo_list)
            model_probs = [l.model_prob for l in combo_list]
            market_probs = [l.market_prob for l in combo_list]

            # ── Joint probabilities ─────────────────────────────────────
            joint_model = joint_probability(model_probs, pairs)
            joint_market = market_joint_probability(market_probs, pairs)
            joint_independent = float(np.prod(model_probs))

            # ── Payout & EV ─────────────────────────────────────────────
            payout = 1.0 / float(np.prod(market_probs)) if all(m > 0 for m in market_probs) else 0.0
            ev = joint_model * payout - 1.0

            # ── Kelly ───────────────────────────────────────────────────
            kelly = parlay_kelly(joint_model, payout, fraction=0.25)

            # ── Implied correlation (average absolute rho) ──────────────
            avg_rho = float(np.mean([abs(r) for _, _, r in pairs])) if pairs else 0.0

            if ev > 0 and kelly > 0:
                candidates.append(KalshiParlay(
                    legs=combo_list,
                    num_legs=n,
                    combined_model_prob=round(joint_model, 4),
                    combined_market_prob=round(joint_market, 4),
                    payout_multiple=round(payout, 2),
                    expected_value=round(ev, 4),
                    kelly_fraction=round(kelly, 4),
                    correlation_pairs=pairs,
                    implied_correlation=round(avg_rho, 3),
                    joint_independent_prob=round(joint_independent, 4),
                ))

        candidates.sort(key=lambda p: p.expected_value, reverse=True)
        return candidates


# ── Display helpers ─────────────────────────────────────────────────────

def format_parlay(parlay: KalshiParlay, index: int = 0) -> str:
    """Format a parlay for display."""
    lines = []
    ev_pct = parlay.expected_value * 100
    indep_prob = parlay.joint_independent_prob
    corr_boost = (parlay.combined_model_prob / max(indep_prob, 1e-10) - 1) * 100

    lines.append(
        f"\n  #{index+1}: {parlay.num_legs}-leg | "
        f"EV={ev_pct:+.1f}% | "
        f"Payout={parlay.payout_multiple:.1f}x | "
        f"Kelly={parlay.kelly_fraction:.1%} | "
        f"ρ̅={parlay.implied_correlation:.0%}"
    )
    lines.append(f"  P(joint)={parlay.combined_model_prob:.1%} "
                 f"(indep={indep_prob:.1%}, "
                 f"corr_boost={corr_boost:+.0%})")
    lines.append(f"  {'─' * 70}")
    for i, leg in enumerate(parlay.legs):
        lines.append(
            f"  Leg {i+1}: {leg.player_name:25s} {leg.market_type:4s} "
            f"mkt={leg.market_prob:.0%} model={leg.model_prob:.0%} "
            f"edge={leg.edge:+.0%} @ {leg.price_cents}¢"
        )
    return '\n'.join(lines)


def display_parlays(results: dict[int, list[KalshiParlay]],
                    title: str = "Kalshi Multi-Leg Plays"):
    """Display parlay results grouped by leg count."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")

    if not results:
        print("  No qualifying parlays found (need at least 2 edges)")
        return

    for n_legs in sorted(results.keys()):
        parlays = results[n_legs]
        print(f"\n{'─' * 70}")
        print(f"  BEST {n_legs}-LEG PLAYS ({len(parlays)} found)")
        print(f"{'─' * 70}")
        for i, parlay in enumerate(parlays[:5]):
            print(format_parlay(parlay, i))

    print()
