from dataclasses import dataclass
from typing import Optional


STANDARD_PAYOUTS: dict[int, float] = {
    2: 3.0, 3: 6.0, 4: 10.0, 5: 20.0, 6: 37.5,
}

FLEX_PAYOUTS: dict[tuple[int, int], float] = {
    (5, 5): 10.0, (5, 4): 2.0, (5, 3): 1.0,
    (6, 6): 25.0, (6, 5): 2.0, (6, 4): 1.0,
}


@dataclass
class Bet:
    sport: str
    stat_type: str
    player: str
    prediction: float
    line: float
    direction: str
    model_prob: float
    edge: float
    kelly_fraction: float
    bet_size: float
    entry_id: str
    legs: int = 5
    entry_type: str = "flex"


class RiskManager:
    def __init__(
        self,
        bankroll: float = 1000,
        kelly_fraction: float = 0.25,
        max_bet_pct: float = 0.03,
        max_concurrent: int = 10,
        daily_loss_limit: float = 0.10,
        min_edge: float = 0.03,
        default_legs: int = 5,
        default_entry_type: str = "flex",
    ):
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.max_bet_pct = max_bet_pct
        self.max_concurrent = max_concurrent
        self.daily_loss_limit = daily_loss_limit
        self.min_edge = min_edge
        self.default_legs = default_legs
        self.default_entry_type = default_entry_type
        self.active_bets: list[Bet] = []
        self.starting_bankroll = bankroll
        self.daily_pnl = 0.0

    def kelly_size(self, prob: float, decimal_odds: float) -> float:
        if decimal_odds <= 1:
            return 0.0
        b = decimal_odds - 1
        q = 1 - prob
        if b * prob - q <= 0:
            return 0.0
        kelly = (b * prob - q) / b
        return max(0.0, kelly * self.kelly_fraction)

    def decimal_odds(self, legs: int, entry_type: str) -> float:
        if entry_type == "flex":
            payout = FLEX_PAYOUTS.get((legs, legs), STANDARD_PAYOUTS.get(legs, 0))
        else:
            payout = STANDARD_PAYOUTS.get(legs, 0)
        return payout if payout > 0 else 0.0

    def size_bet(self, prob: float, line: float, direction: str,
                 legs: Optional[int] = None, entry_type: Optional[str] = None) -> float:
        legs = legs or self.default_legs
        entry_type = entry_type or self.default_entry_type

        decimal_odds = self.decimal_odds(legs, entry_type)
        if decimal_odds <= 0:
            return 0.0

        model_edge = prob - (1.0 / decimal_odds)
        if model_edge < self.min_edge:
            return 0.0

        kelly = self.kelly_size(prob, decimal_odds)
        capped = min(kelly, self.max_bet_pct)
        return round(capped * self.bankroll, 2)

    def approve(self, bet: Bet) -> bool:
        if bet.edge < self.min_edge:
            return False
        if bet.bet_size > self.bankroll * self.max_bet_pct:
            return False
        if len(self.active_bets) >= self.max_concurrent:
            return False
        if self.daily_pnl <= -self.starting_bankroll * self.daily_loss_limit:
            return False
        return True

    def record_result(self, bet: Bet, won: bool):
        if won:
            self.bankroll += bet.bet_size
            self.daily_pnl += bet.bet_size
        else:
            self.bankroll -= bet.bet_size
            self.daily_pnl -= bet.bet_size
        self.active_bets = [b for b in self.active_bets if b.entry_id != bet.entry_id]

    def reset_daily(self):
        self.daily_pnl = 0.0
