#!/usr/bin/env python3
"""Staleness validator: per-sport season windows + cache freshness check.

For every sport in season, the latest game_date in the player-game-logs
parquet must be within `threshold_days` (default 7) of today. If not, the
scan is aborted with a loud error.

The check that would have caught the June 9, 2026 NBA Finals data bug:
the cache had a 2025-04-13 max game_date, but the model was being used
to suggest bets for a 2026-06-08 game. This guard refuses to start a
scan in that state.

Sports with player game logs:
  NBA   Oct 1 - Jun 30   data/nba_cache/game_logs_v14.parquet
  MLB   Apr 1 - Oct 31   data/mlb_cache/*.parquet
  NFL   Sep 1 - Feb 28    data/nfl_cache/*.parquet
  NHL   Oct 1 - Jun 30   data/nhl_cache/*.parquet
  WNBA  May 1 - Oct 31    data/wnba_cache/wnba_games.parquet
  CFB   Aug 15 - Jan 31   data/cfb_cache/*.parquet

Wired into:
  - scripts/hooks/pre-commit  (bash, calls `python -m src.utils.staleness`)
  - src/scripts/morning_scan.py (top of morning_scan(), sys.exit(1) on stale)

Override: set `STALENESS_OVERRIDE=1` in env to skip the check
(only honored in morning_scan, NOT in pre-commit).
"""
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# (sport, season_start_md, season_end_md, parquet_glob, date_col)
SPORT_CONFIG = [
    ("nba",  (10, 1), (6, 30),  "data/nba_cache/game_logs_v14.parquet", "game_date"),
    ("mlb",  (3, 20), (11, 15), "data/cache/mlb/game_logs*.parquet",    "game_date"),
    ("nfl",  (9, 1),  (2, 28),  "data/nfl_cache/*.parquet",             "game_date"),
    ("nhl",  (10, 1), (6, 30),  "data/nhl_cache/*.parquet",             "game_date"),
    ("wnba", (5, 1),  (10, 31), "data/wnba_cache/wnba_games.parquet",   "game_date"),
    ("cfb",  (8, 15), (1, 31),  "data/cfb_cache/*.parquet",             "game_date"),
]


def in_season(today: datetime, start_md: tuple, end_md: tuple) -> bool:
    """True if today is within the season window.

    Handles wrap-around seasons (e.g. NFL Sep-Feb: start=Sep, end=Feb).
    """
    month, day = today.month, today.day
    sm, sd = start_md
    em, ed = end_md
    today_ord = month * 100 + day
    start_ord = sm * 100 + sd
    end_ord = em * 100 + ed
    if start_ord <= end_ord:
        return start_ord <= today_ord <= end_ord
    # Wrap-around (e.g. Sep 1 → Feb 28)
    return today_ord >= start_ord or today_ord <= end_ord


def latest_game_date(parquet_path: Path, date_col: str):
    """Return the most recent date in the parquet, or None if missing/empty.

    Robust against:
      - column name variations (case-insensitive match)
      - missing pyarrow (returns None, caller treats as "no cache")
      - all-NaT values (returns None)
    """
    if not parquet_path.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"    [debug] {parquet_path.name}: read_parquet failed: {e}")
        return None
    if df.empty:
        return None
    # Find date column (case-insensitive)
    actual_col = None
    for c in df.columns:
        if c.lower() == date_col.lower():
            actual_col = c
            break
    if actual_col is None:
        print(f"    [debug] {parquet_path.name}: no '{date_col}' column (have: {list(df.columns)[:5]})")
        return None
    s = pd.to_datetime(df[actual_col], errors="coerce").dropna()
    if s.empty:
        return None
    return s.max().to_pydatetime()


def check_sport(sport: str, start_md, end_md, parquet_glob: str, date_col: str,
                threshold_days: int = 7) -> dict:
    """Check a single sport. Returns dict with is_ok, message, latest_date, age_days."""
    today = datetime.now()
    in_window = in_season(today, start_md, end_md)
    parquet_path = PROJECT_ROOT / parquet_glob
    # Glob expansion for sports with multiple parquet files (MLB, NHL, etc.)
    if "*" in parquet_glob:
        candidates = sorted(PROJECT_ROOT.glob(parquet_glob))
        if not candidates:
            return {
                "sport": sport, "is_ok": True, "in_season": in_window,
                "latest_date": None, "age_days": None,
                "message": f"{sport.upper()}: off-season / no cache found ({parquet_glob})",
            }
        # Use the largest parquet (most likely the player game logs)
        parquet_path = max(candidates, key=lambda p: p.stat().st_size)
    latest = latest_game_date(parquet_path, date_col)
    if latest is None:
        return {
            "sport": sport, "is_ok": True, "in_season": in_window,
            "latest_date": None, "age_days": None,
            "message": f"{sport.upper()}: no cache found at {parquet_path.name}",
        }
    age_days = (today - latest).days
    # Off-season: only flag if data is absurdly old (e.g. >180 days)
    if not in_window:
        is_ok = age_days <= 180
        msg = (f"{sport.upper()}: off-season, latest game {latest:%Y-%m-%d} "
               f"({age_days}d ago) — {'OK' if is_ok else 'STALE (off-season > 180d)'}")
    else:
        is_ok = age_days <= threshold_days
        if is_ok:
            msg = f"{sport.upper()}: OK — latest game {latest:%Y-%m-%d} ({age_days}d ago)"
        else:
            msg = (f"{sport.upper()}: STALE — latest game {latest:%Y-%m-%d} is "
                   f"{age_days}d old (threshold {threshold_days}d). "
                   f"Run: python -m src.scripts.refresh_{sport} or "
                   f"src.data.{sport}.NBADataSource.fetch_player_game_logs()")
    return {
        "sport": sport, "is_ok": is_ok, "in_season": in_window,
        "latest_date": latest, "age_days": age_days, "message": msg,
    }


def check_all_sports(threshold_days: int = 7) -> list:
    """Check every sport. Returns list of result dicts."""
    return [check_sport(sport, sm, em, glob, col, threshold_days)
            for sport, sm, em, glob, col in SPORT_CONFIG]


def main() -> int:
    results = check_all_sports()
    failed = [r for r in results if not r["is_ok"]]
    print()
    for r in results:
        print(f"  [staleness] {r['message']}")
    print()
    if failed:
        print(f"  STALENESS CHECK FAILED: {len(failed)} sport(s) stale")
        print("  " + "!" * 66)
        for r in failed:
            print(f"  !!! {r['sport'].upper()}: latest {r['latest_date']:%Y-%m-%d if r['latest_date'] else 'N/A'} "
                  f"({r['age_days']}d old)")
            print(f"  !!!   {r['message']}")
        print("  " + "!" * 66)
        print("  Refusing to suggest bets with stale data. Refresh first.")
        return 1
    print(f"  [staleness] all sports fresh ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
