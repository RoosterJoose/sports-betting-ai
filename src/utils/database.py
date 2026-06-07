from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text


class Database:
    def __init__(self, db_path: Path):
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        self._init_tables()

    def _init_tables(self):
        stmts = [
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sport TEXT,
                player TEXT,
                stat_type TEXT,
                line REAL,
                direction TEXT,
                model_prob REAL,
                edge_pct REAL,
                confidence_tier TEXT,
                game_date TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS bets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sport TEXT,
                stat_type TEXT,
                player TEXT,
                line REAL,
                direction TEXT,
                bet_size REAL,
                model_prob REAL,
                edge_pct REAL,
                kelly_fraction REAL,
                platform TEXT,
                status TEXT DEFAULT 'pending',
                outcome TEXT,
                pnl REAL,
                placed_at TEXT DEFAULT (datetime('now')),
                settled_at TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sport TEXT,
                stat_type TEXT,
                model_version TEXT,
                accuracy REAL,
                brier_score REAL,
                calibrated_brier REAL,
                n_train INTEGER,
                n_test INTEGER,
                n_features INTEGER,
                trained_at TEXT DEFAULT (datetime('now'))
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                platform TEXT,
                sport TEXT,
                player TEXT,
                stat_type TEXT,
                line REAL,
                board_time TEXT,
                updated_at TEXT,
                fetched_at TEXT DEFAULT (datetime('now'))
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS bankroll_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bankroll REAL,
                daily_pnl REAL,
                open_positions INTEGER,
                snapshot_at TEXT DEFAULT (datetime('now'))
            )
            """,
        ]
        with self.engine.begin() as conn:
            for stmt in stmts:
                conn.execute(text(stmt))

    def save_predictions(self, df: pd.DataFrame):
        if not df.empty:
            df.to_sql("predictions", self.engine, if_exists="append", index=False)

    def save_bet(self, bet: dict) -> int:
        with self.engine.begin() as conn:
            result = conn.execute(
                text("""
                    INSERT INTO bets (sport, stat_type, player, line, direction,
                                      bet_size, model_prob, edge_pct, kelly_fraction,
                                      platform, status)
                    VALUES (:sport, :stat_type, :player, :line, :direction,
                            :bet_size, :model_prob, :edge_pct, :kelly_fraction,
                            :platform, 'pending')
                """),
                bet,
            )
            return result.lastrowid

    def settle_bet(self, bet_id: int, outcome: str, pnl: float):
        with self.engine.begin() as conn:
            conn.execute(
                text("""
                    UPDATE bets
                    SET outcome = :outcome, pnl = :pnl, status = 'settled',
                        settled_at = datetime('now')
                    WHERE id = :bet_id
                """),
                {"bet_id": bet_id, "outcome": outcome, "pnl": pnl},
            )

    def save_performance(self, metrics: dict):
        pd.DataFrame([metrics]).to_sql(
            "model_performance", self.engine, if_exists="append", index=False
        )

    def snapshot_bankroll(self, bankroll: float, daily_pnl: float, open_positions: int):
        with self.engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO bankroll_snapshots (bankroll, daily_pnl, open_positions)
                    VALUES (:bankroll, :daily_pnl, :open_positions)
                """),
                {"bankroll": bankroll, "daily_pnl": daily_pnl, "open_positions": open_positions},
            )

    def get_performance_summary(self, sport: str = None, days: int = 30) -> pd.DataFrame:
        query = """
            SELECT sport, stat_type, COUNT(*) as n_bets,
                   SUM(CASE WHEN outcome = 'won' THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN outcome = 'lost' THEN 1 ELSE 0 END) as losses,
                   ROUND(AVG(edge_pct), 2) as avg_edge,
                   ROUND(SUM(pnl), 2) as total_pnl,
                   ROUND(SUM(CASE WHEN outcome = 'won' THEN bet_size ELSE 0 END) * 100.0 /
                         NULLIF(SUM(bet_size), 0), 2) as roi_pct
            FROM bets
            WHERE settled_at IS NOT NULL
        """
        params = {}
        if sport:
            query += " AND sport = :sport"
            params["sport"] = sport
        query += " GROUP BY sport, stat_type ORDER BY total_pnl DESC"
        return pd.read_sql(query, self.engine, params=params)

    def get_open_bets(self) -> pd.DataFrame:
        return pd.read_sql(
            "SELECT * FROM bets WHERE status = 'pending'", self.engine
        )
