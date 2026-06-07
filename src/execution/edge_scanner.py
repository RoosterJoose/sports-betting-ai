from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from src.config.settings import SportConfig


# PrizePicks payout structure
STANDARD_PAYOUTS: dict[int, float] = {
    2: 3.0, 3: 6.0, 4: 10.0, 5: 20.0, 6: 37.5,
}

FLEX_PAYOUTS: dict[tuple[int, int], float] = {
    (5, 5): 10.0, (5, 4): 2.0, (5, 3): 1.0,
    (6, 6): 25.0, (6, 5): 2.0, (6, 4): 1.0,
}


@dataclass
class Edge:
    player: str
    sport: str
    stat_type: str
    predicted_value: float
    line_score: float
    edge_pct: float
    direction: str
    confidence_tier: str
    entry_type: str = "standard"
    legs: int = 5


class EdgeScanner:
    CONFIDENCE_TIERS = [
        ("HIGH", 0.15),
        ("MEDIUM", 0.08),
        ("LOW", 0.03),
    ]

    def __init__(self, sport_config: SportConfig, model_dir=None, legs: int = 5, entry_type: str = "flex"):
        self.config = sport_config
        self.legs = legs
        self.entry_type = entry_type
        self.model_dir = model_dir
        self._models: dict[str, xgb.XGBRegressor] = {}

        if entry_type == "flex":
            self.breakeven = self._flex_breakeven(legs)
        else:
            payout = STANDARD_PAYOUTS.get(legs, 0)
            self.breakeven = (1.0 / payout) ** (1.0 / legs) if legs > 0 and payout > 0 else 0.55

    @staticmethod
    def _flex_breakeven(legs: int) -> float:
        return {5: 0.513, 6: 0.542}.get(legs, 0.55)

    def load_model(self, stat_type: str) -> Optional[xgb.XGBRegressor]:
        if stat_type in self._models:
            return self._models[stat_type]
        if self.model_dir is None:
            return None
        path = self.model_dir / self.config.name / f"{stat_type}.json"
        if not path.exists():
            return None
        model = xgb.XGBRegressor()
        model.load_model(str(path))
        self._models[stat_type] = model
        return model

    def predict_value(self, stat_type: str, features: pd.DataFrame) -> Optional[float]:
        """Predict the actual stat value using the regression model."""
        model = self.load_model(stat_type)
        if model is None:
            return None
        try:
            pred = model.predict(features)[0]
            return float(pred)
        except Exception as e:
            print(f"  [edge] prediction failed for {stat_type}: {e}")
            return None

    def evaluate_edge(self, stat_type: str, line_score: float, predicted_value: float) -> Optional[Edge]:
        """Compute edge from regression prediction vs PrizePicks line."""
        if predicted_value is None or line_score is None:
            return None

        line_score = float(line_score)
        predicted_value = float(predicted_value)

        # Edge = how far from the line, as a percentage of the line
        if line_score == 0:
            return None

        over_edge = (predicted_value - line_score) / line_score
        under_edge = (line_score - predicted_value) / line_score

        if over_edge > under_edge and over_edge > 0:
            direction = "over"
            edge_pct = over_edge
        elif under_edge > 0:
            direction = "under"
            edge_pct = under_edge
        else:
            return None

        tier = "LOW"
        for t, threshold in self.CONFIDENCE_TIERS:
            if edge_pct >= threshold:
                tier = t
                break

        return Edge(
            player="",
            sport=self.config.name,
            stat_type=stat_type,
            predicted_value=round(predicted_value, 2),
            line_score=line_score,
            edge_pct=round(edge_pct * 100, 2),
            direction=direction,
            confidence_tier=tier,
            entry_type=self.entry_type,
            legs=self.legs,
        )

    def scan(self, candidates: list[dict]) -> list[Edge]:
        """Scan a list of candidate bets.
        Each candidate: {"stat_type": str, "line_score": float, "features": pd.DataFrame}
        """
        edges = []
        for c in candidates:
            pred = self.predict_value(c["stat_type"], c["features"])
            if pred is None:
                continue
            edge = self.evaluate_edge(c["stat_type"], c["line_score"], pred)
            if edge:
                edge.player = c.get("player", "")
                edges.append(edge)
        return sorted(edges, key=lambda e: e.edge_pct, reverse=True)

    def filter_edges(self, edges: list[Edge], min_edge_pct: float = 3.0) -> list[Edge]:
        return [e for e in edges if e.edge_pct >= min_edge_pct]
