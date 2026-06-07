from datetime import datetime
from pathlib import Path
import time

import pandas as pd

from src.data.base import DataSource

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "nba_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Fixed SEASON_MAP — maps pipeline year to correct NBA season notation
SEASON_MAP = {"2022": "2021-22", "2023": "2022-23", "2024": "2023-24", "2025": "2024-25", "2026": "2025-26"}


class NBADataSource(DataSource):
    def __init__(self):
        self._cached_raw_games = None

    def fetch_player_stats(self, player_id: str, start_date, end_date) -> pd.DataFrame:
        return pd.DataFrame()

    def fetch_team_stats(self, team_id: str, start_date, end_date) -> pd.DataFrame:
        return pd.DataFrame()

    def fetch_schedule(self, season: str) -> pd.DataFrame:
        return pd.DataFrame()

    def fetch_player_game_logs(self, seasons: list[str]) -> pd.DataFrame:
        cache_path = CACHE_DIR / "game_logs_v14.parquet"
        if cache_path.exists():
            print(f"  Loading NBA from cache: {cache_path}")
            df = pd.read_parquet(cache_path)
            self._cached_raw_games = df.copy()
            return df

        api_seasons = list(dict.fromkeys(  # preserve order, dedup
            SEASON_MAP.get(str(s)[:4]) for s in seasons
        ))
        api_seasons = [s for s in api_seasons if s]
        if not api_seasons:
            api_seasons = ["2024-25"]

        from nba_api.stats.endpoints import playergamelogs

        frames = []
        for api_season in api_seasons:
            print(f"  Fetching NBA {api_season} via PlayerGameLogs...")
            logs = playergamelogs.PlayerGameLogs(season_nullable=api_season, timeout=30)
            df = logs.get_data_frames()[0]
            if df.empty:
                print(f"  No data for {api_season}")
                continue
            df.columns = [c.lower() for c in df.columns]
            df["season"] = api_season
            if "game_date" in df.columns:
                df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
            if "player_id" in df.columns:
                df["player_id"] = df["player_id"].astype(str)
            frames.append(df)
            print(f"  {api_season}: {len(df)} rows")
            time.sleep(1)

        if frames:
            result = pd.concat(frames, ignore_index=True)
            print(f"  NBA total: {len(result)} rows across {len(frames)} seasons")
            result.to_parquet(cache_path)
            self._cached_raw_games = result.copy()
            return result

        print("  No NBA data available")
        return pd.DataFrame()

    def get_raw_game_logs(self) -> pd.DataFrame:
        if self._cached_raw_games is not None:
            return self._cached_raw_games
        cache_path = CACHE_DIR / "game_logs_v14.parquet"
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            self._cached_raw_games = df.copy()
            return df
        return pd.DataFrame()
