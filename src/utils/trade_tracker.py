import json
import sqlite3
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "trade_tracker.db"


class TradeTracker:
    def __init__(self, db_path: str = str(DB_PATH)):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                sport TEXT,
                model_name TEXT,
                ticker TEXT,
                title TEXT,
                side TEXT,
                price_cents INTEGER,
                size INTEGER,
                model_prob REAL,
                market_prob REAL,
                edge REAL,
                live INTEGER DEFAULT 0,
                status TEXT DEFAULT 'pending',
                resolved_price REAL,
                pnl REAL DEFAULT 0.0,
                notes TEXT,
                UNIQUE(ticker, timestamp)
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_sport_model
            ON trades(sport, model_name)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_status
            ON trades(status)
        """)
        self._conn.commit()

    def log_trade(self, sport: str, model_name: str, ticker: str, title: str,
                  side: str, price_cents: int, size: int, model_prob: float,
                  market_prob: float, edge: float, live: bool = False,
                  notes: str = ""):
        ts = datetime.now().isoformat(timespec="seconds")
        self._conn.execute("""
            INSERT OR IGNORE INTO trades
            (timestamp, sport, model_name, ticker, title, side, price_cents, size,
             model_prob, market_prob, edge, live, notes)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (ts, sport, model_name, ticker, title, side, price_cents, size,
              round(model_prob, 4), round(market_prob, 4), round(edge, 4),
              1 if live else 0, notes))
        self._conn.commit()

    def log_batch(self, trades: list[dict]):
        for t in trades:
            self.log_trade(
                sport=t.get("sport", "unknown"),
                model_name=t.get("model_name", t.get("type", "unknown")),
                ticker=t.get("ticker", ""),
                title=t.get("title", t.get("label", "")),
                side=t.get("side", "yes"),
                price_cents=int(t.get("price_cents", 0)),
                size=int(t.get("contracts", t.get("size", 1))),
                model_prob=t.get("model_prob", 0.5),
                market_prob=t.get("market_prob", 0.5),
                edge=t.get("edge", 0),
                live=t.get("live", False),
                notes=t.get("notes", ""),
            )

    def resolve_trade(self, ticker: str, resolved_price: float, status: str = "won"):
        pnl = 0.0
        rows = self._conn.execute(
            "SELECT id, side, price_cents, size FROM trades WHERE ticker=? AND status='pending'",
            (ticker,)
        ).fetchall()
        for row in rows:
            tid, side, price_cents, size = row
            if status == "won":
                if side == "yes":
                    pnl = size * (resolved_price - price_cents / 100.0)
                else:
                    pnl = size * (price_cents / 100.0 - resolved_price)
            elif status == "lost":
                if side == "yes":
                    pnl = -size * (price_cents / 100.0)
                else:
                    pnl = -size * (1.0 - price_cents / 100.0)
            self._conn.execute(
                "UPDATE trades SET status=?, resolved_price=?, pnl=? WHERE id=?",
                (status, resolved_price, round(pnl, 2), tid)
            )
        self._conn.commit()
        return pnl

    def get_analytics(self, sport: str = None, model_name: str = None,
                      min_sample: int = 10) -> dict:
        where = ["status IN ('won','lost')"]
        params = []
        if sport:
            where.append("sport=?")
            params.append(sport)
        if model_name:
            where.append("model_name=?")
            params.append(model_name)

        q = f"""
            SELECT sport, model_name,
                   COUNT(*) as n,
                   SUM(CASE WHEN status='won' THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN status='lost' THEN 1 ELSE 0 END) as losses,
                   ROUND(AVG(edge), 4) as avg_edge,
                   ROUND(AVG(CASE WHEN status='won' THEN edge ELSE NULL END), 4) as avg_win_edge,
                   ROUND(AVG(CASE WHEN status='lost' THEN edge ELSE NULL END), 4) as avg_loss_edge,
                   ROUND(SUM(pnl), 2) as total_pnl,
                   ROUND(AVG(pnl), 4) as avg_pnl,
                   ROUND(SUM(price_cents * size) / 100.0, 2) as total_volume,
                   ROUND(AVG(live), 1) as is_live
            FROM trades
            WHERE {' AND '.join(where)}
            GROUP BY sport, model_name
            HAVING n >= ?
            ORDER BY total_pnl DESC
        """
        params.append(min_sample)
        df = pd.read_sql_query(q, self._conn, params=params)
        if df.empty:
            return {}
        df["win_rate"] = (df["wins"] / df["n"]).round(3)
        df["roi"] = (df["total_pnl"] / df["total_volume"].clip(lower=1)).round(3)
        return df.to_dict(orient="records")

    def get_calibration(self, sport: str = None, model_name: str = None,
                        bins: int = 10) -> pd.DataFrame:
        where = ["status IN ('won','lost')"]
        params = []
        if sport:
            where.append("sport=?")
            params.append(sport)
        if model_name:
            where.append("model_name=?")
            params.append(model_name)

        q = f"""
            SELECT model_prob, CASE WHEN status='won' THEN 1.0 ELSE 0.0 END as outcome
            FROM trades
            WHERE {' AND '.join(where)}
        """
        df = pd.read_sql_query(q, self._conn, params=params)
        if df.empty:
            return pd.DataFrame()
        df["bin"] = pd.cut(df["model_prob"], bins=bins, labels=False)
        cal = df.groupby("bin").agg(
            pred_prob=("model_prob", "mean"),
            actual_rate=("outcome", "mean"),
            count=("outcome", "count"),
        ).reset_index()
        cal["brier"] = (cal["pred_prob"] - cal["actual_rate"]) ** 2
        return cal

    def summary(self, min_sample: int = 5) -> str:
        lines = []
        lines.append(f"{'='*70}")
        lines.append(f"  TRADE TRACKER — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"{'='*70}")

        # Pending trades
        pending = self._conn.execute(
            "SELECT COUNT(*) FROM trades WHERE status='pending'"
        ).fetchone()[0]
        live_pending = self._conn.execute(
            "SELECT COUNT(*) FROM trades WHERE status='pending' AND live=1"
        ).fetchone()[0]
        lines.append(f"  Pending: {pending} ({live_pending} live)")

        # Per-model stats
        analytics = self.get_analytics(min_sample=min_sample)
        if analytics:
            lines.append(f"\n  {'Model':25s} {'N':>5s} {'Wins':>5s} {'WR%':>6s} {'ROI%':>6s} {'Avg Edge':>8s}")
            lines.append(f"  {'-'*25} {'-'*5} {'-'*5} {'-'*6} {'-'*6} {'-'*8}")
            for a in analytics:
                label = f"{a['sport']}/{a['model_name']}"
                lines.append(
                    f"  {label:25s} {a['n']:5d} {a['wins']:5d} "
                    f"{a['win_rate']*100:5.1f}% {a['roi']*100:5.1f}% "
                    f"{a['avg_edge']*100:6.1f}%"
                )

        # Recent trades
        recent = pd.read_sql_query(
            "SELECT timestamp, sport, model_name, title, side, edge, live, status, pnl "
            "FROM trades ORDER BY id DESC LIMIT 10",
            self._conn
        )
        if not recent.empty:
            lines.append(f"\n  Recent trades:")
            for _, r in recent.iterrows():
                icon = "🔴" if r["status"] == "lost" else "🟢" if r.get("pnl", 0) > 0 else "⏳"
                lines.append(
                    f"  {icon} {r['timestamp'][:10]} {r['sport']:5s}/{r['model_name']:12s} "
                    f"{r['title'][:20]:20s} edge={r['edge']:.0%} "
                    f"{'LIVE' if r['live'] else 'PAP':4s} {r['status']:8s} "
                    f"pnl=${r.get('pnl',0):+.2f}"
                )

        return "\n".join(lines)

    def close(self):
        self._conn.close()
