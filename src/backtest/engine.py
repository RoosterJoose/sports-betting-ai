from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from src.execution.risk import RiskManager, Bet
from src.execution.edge_scanner import EdgeScanner


class BacktestEngine:
    def __init__(self, risk: Optional[RiskManager] = None):
        self.risk = risk or RiskManager()
        self.trades = []
        self.equity_curve = []

    def run(
        self,
        predictions: pd.DataFrame,
        actuals: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
    ) -> dict:
        merged = predictions.merge(
            actuals, on=["player", "stat_type", "game_date"], suffixes=("_pred", "_act")
        )
        merged = merged.sort_values("game_date")

        for _, row in merged.iterrows():
            bet = Bet(
                sport=row.get("sport", "nba"),
                stat_type=row["stat_type"],
                player=row["player"],
                prediction=row["model_prob"],
                line=row["line_score"],
                direction="over" if row["model_prob"] > 0.5 else "under",
                model_prob=row["model_prob"],
                edge=abs(row["model_prob"] - 0.5),
                kelly_fraction=self.risk.kelly_fraction,
                bet_size=0,
                entry_id=f"{row['player']}_{row['stat_type']}_{row['game_date']}",
            )
            bet.bet_size = self.risk.size_bet(bet.model_prob, bet.line, bet.direction)

            if bet.bet_size <= 0 or not self.risk.approve(bet):
                continue

            actual_value = row.get("actual_value", 0)
            won = actual_value > bet.line if bet.direction == "over" else actual_value < bet.line
            self.risk.record_result(bet, won)

            self.trades.append({
                "date": row["game_date"],
                "player": bet.player,
                "stat": bet.stat_type,
                "line": bet.line,
                "direction": bet.direction,
                "prob": bet.model_prob,
                "bet_size": bet.bet_size,
                "won": won,
                "pnl": bet.bet_size if won else -bet.bet_size,
                "bankroll": self.risk.bankroll,
            })

        return self.summarize()

    def summarize(self) -> dict:
        if not self.trades:
            return {"total_trades": 0}

        df = pd.DataFrame(self.trades)
        wins = df["won"].sum()
        total = len(df)
        win_rate = wins / total if total else 0
        total_pnl = df["pnl"].sum()

        df["cumulative_pnl"] = df["pnl"].cumsum()
        peak = df["cumulative_pnl"].cummax()
        drawdown = (df["cumulative_pnl"] - peak).min()

        returns = df["pnl"] / (df["bankroll"] - df["pnl"])
        sharpe = (returns.mean() / returns.std() * (252 ** 0.5)) if returns.std() > 0 else 0

        return {
            "total_trades": total,
            "wins": int(wins),
            "losses": total - int(wins),
            "win_rate": round(win_rate, 4),
            "total_pnl": round(total_pnl, 2),
            "roi_pct": round(total_pnl / self.risk.starting_bankroll * 100, 2),
            "max_drawdown_pct": round(float(drawdown) / self.risk.starting_bankroll * 100, 2),
            "sharpe_ratio": round(sharpe, 4),
            "final_bankroll": round(self.risk.bankroll, 2),
            "avg_bet_size": round(df["bet_size"].mean(), 2),
        }

    def equity_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.equity_curve)
