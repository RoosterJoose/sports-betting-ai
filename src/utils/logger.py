import json
import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(name: str = "sports-betting-ai", log_dir: Path = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"trade_{datetime.now():%Y%m%d}.log")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


class TradeLogger:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def prediction(self, player: str, stat: str, prob: float, line: float, edge: float):
        self.logger.info(
            "PRED | %s %s | prob=%.1f%% line=%.1f edge=%.2f%%",
            player, stat, prob * 100, line, edge * 100,
        )

    def bet_placed(self, platform: str, sport: str, detail: str, amount: float):
        self.logger.info("BET  | %s/%s | $%.2f | %s", platform, sport, amount, detail)

    def bet_result(self, bet_id: int, won: bool, pnl: float):
        self.logger.info("RES  | bet#%d | %s | $%.2f", bet_id, "WON" if won else "LOST", pnl)

    def edge_found(self, count: int, platform: str):
        self.logger.info("EDGE | %d opportunities on %s", count, platform)

    def risk_alert(self, message: str):
        self.logger.warning("RISK | %s", message)

    def kalshi_trade(self, ticker: str, side: str, price: int, size: float):
        self.logger.info(
            "KALSHI | %s %s @ %d¢ | $%.2f", ticker, side.upper(), price, size
        )

    def model_trained(self, sport: str, stat: str, accuracy: float, samples: int):
        self.logger.info(
            "TRAIN | %s %s | acc=%.1f%% n=%d", sport, stat, accuracy * 100, samples
        )

    def portfolio_summary(self, bankroll: float, daily_pnl: float):
        self.logger.info(
            "BANKROLL | $%.2f | today: $%.2f (%.1f%%)",
            bankroll, daily_pnl,
            daily_pnl / bankroll * 100 if bankroll else 0,
        )
