import time
from datetime import datetime
from typing import Optional

import pandas as pd

from src.data.kalshi import KalshiClient
from src.execution.risk import Bet, RiskManager
from src.config.settings import settings


class KalshiTrader:
    def __init__(self, client: Optional[KalshiClient] = None, risk: Optional[RiskManager] = None):
        self.client = client or KalshiClient()
        self.risk = risk or RiskManager()

    def _kalshi_binary_kelly(self, win_prob, price_paid):
        """Full-Kelly fraction for a Kalshi binary contract.
        
        Buying NO at price P (where P = 1 - YES_price):
          Win: get $1 back on $P paid → net profit = 1-P per $P risked
          Lose: you lose $P
        """
        net_odds = (1.0 - price_paid) / max(price_paid, 0.001)
        q = 1.0 - win_prob
        if net_odds * win_prob - q <= 0:
            return 0.0
        return (net_odds * win_prob - q) / net_odds

    def safe_compounder_scan(self) -> list[dict]:
        balance = self.client.get_balance()
        self.risk.bankroll = balance
        opportunities = self.client.find_no_side_opportunities()
        if opportunities.empty:
            return []

        trades = []
        for _, row in opportunities.iterrows():
            trade = self._evaluate_no_side(row)
            if trade:
                trades.append(trade)
        return trades

    def _evaluate_no_side(self, row: pd.Series) -> Optional[dict]:
        cfg = settings.kalshi
        if row["category"] in ["Sports", "Entertainment"]:
            return None

        yes_price = row["yes_price"]
        if yes_price <= 0.0:
            return None

        no_price = 1.0 - yes_price
        best_price = int(yes_price * 100 + 1)
        if best_price >= 20:
            return None

        # Our model: NO is very likely for longshot fades → p_yes_model = yes_price (market)
        # But we believe NO has higher true prob:
        # edge = market_implied_no - yes_price (in cents)
        # For a YES at 5¢: implied NO = 95¢, true NO = 95¢ + edge
        maker_edge = (no_price * 100) - (yes_price * 100) - cfg.min_edge_cents
        if maker_edge <= 0:
            return None

        category = row.get("category", "")
        if category in ["Politics", "Policy"]:
            fee_mult = 0.0
        elif category in ["Finance"]:
            fee_mult = cfg.maker_fee_mult * 0.5
        else:
            fee_mult = cfg.maker_fee_mult

        # Kelly sizing for binary NO-side (collateral-aware)
        # True NO prob ≈ (1 - yes_price + maker_edge/100) = no_price + edge
        p_no_true = min(0.999, max(0.001, no_price + maker_edge / 100))
        p_no_market = no_price
        kelly_frac = self._kalshi_binary_kelly(p_no_true, no_price)

        # Max allowable risk per bet
        actual_balance = self.risk.bankroll
        max_risk = actual_balance * self.risk.max_bet_pct
        risk_per_contract = no_price  # max loss per NO contract
        int_count = int(max_risk / risk_per_contract)
        if int_count < 1:
            return None

        # Also cap by Kelly fraction
        kelly_count = int(actual_balance * kelly_frac / no_price) if no_price > 0 else 0
        int_count = min(int_count, max(1, kelly_count))

        return {
            "ticker": row["ticker"],
            "event": row["title"],
            "action": "buy",
            "side": "no",
            "yes_price": yes_price,
            "no_implied": no_price,
            "edge_cents": round(maker_edge, 2),
            "order_price_cents": best_price,
            "size": int_count,
            "strategy": "safe_compounder",
            "category": category,
            "fee_mult": fee_mult,
        }

    def execute(self, trade: dict):
        cost_per = trade["order_price_cents"] / 100.0 if trade["side"] == "yes" else (1 - trade["order_price_cents"] / 100.0)
        dollar_amount = float(trade["size"]) * cost_per
        # Collateral check for NO-side: need (1 - yes_price) * count reserved
        if trade["side"] == "no":
            collateral = trade["size"] * (1 - trade["order_price_cents"] / 100.0)
            if collateral > self.risk.bankroll * self.risk.max_bet_pct * 2:
                return None

        bet = Bet(
            sport="kalshi",
            stat_type="binary",
            player=trade["ticker"],
            prediction=1.0 - trade["yes_price"],
            line=trade["yes_price"],
            direction="no",
            model_prob=1.0 - trade["yes_price"],
            edge=trade["edge_cents"] / 100,
            kelly_fraction=self.risk.kelly_fraction,
            bet_size=dollar_amount,
            entry_id=trade["ticker"],
        )
        if not self.risk.approve(bet):
            return None

        order = self.client.create_order(
            ticker=trade["ticker"],
            side=trade["side"],
            yes_price=trade["order_price_cents"],
            count=f"{trade['size']:.0f}",
            order_type="limit",
        )

        trade["order_id"] = order.get("order_id", "")
        trade["executed_at"] = datetime.utcnow().isoformat()
        return trade

    def run_safe_compounder(self):
        balance = self.client.get_balance()
        print(f"[Kalshi Safe Compounder] Balance: ${balance:.2f}")
        print(f"  Min NO contract at 80¢ = $0.80 → needs ${0.80/self.risk.max_bet_pct:.0f} min bankroll")
        trades = self.safe_compounder_scan()
        print(f"  Found {len(trades)} viable opportunities (size>0)")

        executed = []
        for trade in trades[:5]:
            result = self.execute(trade)
            if result:
                executed.append(result)
                print(f"  Executed: {trade['ticker']} {trade['side']} @ {trade['order_price_cents']}¢, {trade['size']} contracts")
            else:
                print(f"  Skipped: {trade['ticker']} (risk check failed)")

        return executed

    def scan_ml_opportunities(self, predictions: pd.DataFrame) -> list[dict]:
        events = self.client.list_events(status="open")
        if events.empty:
            return []

        markets = self.client.list_sports_markets_for_events(events["ticker"].tolist()[:20])
        if markets.empty:
            return []

        trades = []
        for _, m in markets.iterrows():
            ticker = m["ticker"]
            market_price = float(m.get("yes_price", "0.5000"))

            pred_row = predictions[predictions["ticker"] == ticker]
            if pred_row.empty:
                continue
            model_prob = pred_row.iloc[0]["model_prob"]

            if model_prob > market_price:
                edge_pct = (model_prob - market_price) / market_price
                if edge_pct >= 0.05:
                    trades.append({
                        "ticker": ticker,
                        "title": m.get("title", ""),
                        "market_price": market_price,
                        "model_prob": model_prob,
                        "edge_pct": round(edge_pct * 100, 2),
                        "direction": "yes",
                    })

        return sorted(trades, key=lambda t: t["edge_pct"], reverse=True)
