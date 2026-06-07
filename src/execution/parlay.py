import itertools
from typing import Optional

import numpy as np
import pandas as pd

from src.execution.edge_scanner import Edge, STANDARD_PAYOUTS, FLEX_PAYOUTS


class ParlayOptimizer:
    SAME_PLAYER = 0.1

    def __init__(
        self,
        max_legs: int = 6,
        min_legs: int = 2,
        max_correlation: float = 0.4,
        entry_type: str = "flex",
    ):
        self.max_legs = max_legs
        self.min_legs = min_legs
        self.max_correlation = max_correlation
        self.entry_type = entry_type

    def build_entries(self, edges: list[Edge], max_entries: int = 10) -> list[dict]:
        edges_sorted = sorted(edges, key=lambda e: e.edge_pct, reverse=True)
        entries = []
        used_combos = set()

        for legs in range(min(self.max_legs, len(edges_sorted)), self.min_legs - 1, -1):
            for combo in itertools.combinations(edges_sorted, legs):
                if len(entries) >= max_entries:
                    break
                if self._is_valid_slip(combo):
                    key = tuple(sorted(e.player + e.stat_type for e in combo))
                    if key not in used_combos:
                        used_combos.add(key)
                        slip = self._build_slip(combo)
                        entries.append(slip)

        return entries

    def _is_valid_slip(self, edges: tuple[Edge]) -> bool:
        correlation = self._estimate_correlation(edges)
        return correlation <= self.max_correlation

    def _estimate_correlation(self, edges: tuple[Edge]) -> float:
        players = [e.player for e in edges]
        total = len(edges)
        if total < 2:
            return 0.0
        same_player = sum(1 for i in range(total) for j in range(i + 1, total) if players[i] == players[j])
        weight = (same_player * self.SAME_PLAYER * 2) / (total * (total - 1))
        return weight

    def _build_slip(self, edges: tuple[Edge]) -> dict:
        legs = len(edges)
        probs = [e.model_prob for e in edges]

        if self.entry_type == "flex":
            ev, combined_prob, payout_dist = self._flex_expected_value(probs, legs)
        else:
            combined_prob = np.prod(probs)
            payout = STANDARD_PAYOUTS.get(legs, 0)
            ev = combined_prob * payout - (1 - combined_prob)
            payout_dist = {legs: payout}

        return {
            "legs": legs,
            "entry_type": self.entry_type,
            "entries": [
                {
                    "player": e.player,
                    "stat_type": e.stat_type,
                    "line": e.line,
                    "direction": e.direction,
                    "model_prob": e.model_prob,
                }
                for e in edges
            ],
            "combined_prob": round(combined_prob if self.entry_type != "flex" else ev + 1, 4),
            "payout_distribution": payout_dist,
            "expected_value": round(ev, 4),
            "expected_roi": round(ev * 100, 2),
        }

    def _flex_expected_value(self, probs: list[float], legs: int) -> tuple[float, float, dict]:
        payout_dist = {}
        ev = 0.0
        weighted_prob = 0.0

        for correct in range(legs + 1):
            payout = FLEX_PAYOUTS.get((legs, correct), 0.0)
            if payout == 0:
                continue
            prob = self._prob_exactly_n(probs, correct)
            payout_dist[correct] = (prob, payout)
            ev += prob * payout
            weighted_prob += prob

        return ev - 1.0, weighted_prob, payout_dist

    @staticmethod
    def _prob_exactly_n(probs: list[float], n: int) -> float:
        total = 0.0
        probs_arr = np.array(probs)
        legs = len(probs)
        for combo in itertools.combinations(range(legs), n):
            win_mask = np.zeros(legs, dtype=bool)
            win_mask[list(combo)] = True
            prob = np.prod(np.where(win_mask, probs_arr, 1 - probs_arr))
            total += prob
        return total
