from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data.base import DataSource

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "nfl_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

STAT_COL_MAP = {
    "pass_yds": "passing_yards",
    "rush_yds": "rushing_yards",
    "rec_yds": "receiving_yards",
    "rec": "receptions",
    "td": "touchdowns",
    "pass_td": "passing_tds",
    "rush_td": "rushing_tds",
    "rec_td": "receiving_tds",
    "int": "interceptions",
    "pass_att": "attempts",
    "rush_att": "carries",
    "tgt": "targets",
    "cmp": "completions",
}


class NFLDataSource(DataSource):
    def __init__(self):
        self._cache = {}

    def fetch_player_stats(self, player_id: str, start_date, end_date) -> pd.DataFrame:
        return pd.DataFrame()

    def fetch_team_stats(self, team_id: str, start_date, end_date) -> pd.DataFrame:
        return pd.DataFrame()

    def fetch_schedule(self, season: str) -> pd.DataFrame:
        import nfl_data_py as nfl
        return nfl.import_schedules([int(season)])

    def fetch_player_game_logs(self, seasons: list[str]) -> pd.DataFrame:
        if self._cache.get("weekly") is not None:
            return self._cache["weekly"]

        cache_path = CACHE_DIR / "weekly.parquet"
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            self._cache["weekly"] = df
            print(f"  NFL: {len(df)} rows from cache")
            return df

        import nfl_data_py as nfl

        candidate_years = set()
        for s in seasons:
            try:
                candidate_years.add(int(s))
            except ValueError:
                pass
        # NFL data availability: current season usually ready ~mid-season
        max_year = datetime.now().year
        candidate_years = sorted([y for y in candidate_years if 2000 < y <= max_year], reverse=True)

        if not candidate_years:
            candidate_years = [max_year]

        successful = []
        for y in candidate_years:
            try:
                df_y = nfl.import_weekly_data([y], downcast=True)
                if df_y is not None and not df_y.empty:
                    successful.append(df_y)
                    print(f"    {y}: {len(df_y)} rows")
            except Exception:
                print(f"    {y}: unavailable")
                continue

        if not successful:
            print("  No NFL data available")
            return pd.DataFrame()

        weekly = pd.concat(successful, ignore_index=True)
        weekly["player_id"] = weekly["player_id"].astype(str)
        weekly["game_date"] = pd.to_datetime(
            weekly["season"].astype(str) + "-W" + weekly["week"].astype(str).str.zfill(2) + "-5",
            format="%Y-W%W-%w",
            errors="coerce",
        )
        weekly["touchdowns"] = (
            weekly.get("passing_tds", 0).fillna(0)
            + weekly.get("rushing_tds", 0).fillna(0)
            + weekly.get("receiving_tds", 0).fillna(0)
        )
        weekly["pass_attempts"] = weekly.get("attempts", 0).fillna(0)
        weekly["rush_attempts"] = weekly.get("carries", 0).fillna(0)
        weekly["pass_yds+td"] = weekly.get("passing_yards", 0).fillna(0) + weekly.get("passing_tds", 0).fillna(0)
        weekly["rush+rec_yds"] = weekly.get("rushing_yards", 0).fillna(0) + weekly.get("receiving_yards", 0).fillna(0)

        # Add PrizePicks-compatible column aliases for pipeline matching
        for alias, internal in STAT_COL_MAP.items():
            if internal in weekly.columns and alias != internal:
                weekly[alias] = weekly[internal]

        weekly.to_parquet(cache_path)
        self._cache["weekly"] = weekly
        print(f"  NFL: {len(weekly)} rows, {weekly.player_id.nunique()} players")
        return weekly
