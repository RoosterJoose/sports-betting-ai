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

    def fetch_player_game_logs(self, seasons: list[str] = None) -> pd.DataFrame:
        cache_path = CACHE_DIR / "game_logs_v14.parquet"
        if cache_path.exists():
            print(f"  Loading NBA from cache: {cache_path}")
            df = pd.read_parquet(cache_path)
            self._cached_raw_games = df.copy()
            return df

        # Default seasons: CURRENT season only (2025-26).
        # Per user direction (June 9, 2026): for betting on the 2026 NBA
        # Finals, only the current season is relevant. Older seasons
        # (2023-24, 2024-25) are noise that could mislead the model
        # (different rosters, roles, form). Set explicitly to a list of
        # pipeline years if you need historical data.
        # IMPORTANT: pass pipeline years (e.g. "2026") not API format
        # ("2025-26"). SEASON_MAP looks up str(s)[:4] so passing
        # "2025-26" maps to "2024-25" via the first 4 chars.
        if not seasons:
            seasons = ["2026"]

        api_seasons = list(dict.fromkeys(  # preserve order, dedup
            SEASON_MAP.get(str(s)[:4]) for s in seasons
        ))
        api_seasons = [s for s in api_seasons if s]
        if not api_seasons:
            api_seasons = ["2024-25", "2025-26"]

        # Pull both Regular Season and Playoffs for each season. Without
        # season_type_nullable, nba_api defaults to Regular Season only —
        # the bug behind the 2026 NBA Finals data gap (June 9, 2026).
        season_types = ["Regular Season", "Playoffs"]

        from nba_api.stats.endpoints import playergamelogs

        frames = []
        for api_season in api_seasons:
            for season_type in season_types:
                print(f"  Fetching NBA {api_season} ({season_type}) via PlayerGameLogs...")
                try:
                    logs = playergamelogs.PlayerGameLogs(
                        season_nullable=api_season,
                        season_type_nullable=season_type,
                        timeout=30,
                    )
                    df = logs.get_data_frames()[0]
                except Exception as e:
                    print(f"  Error fetching {api_season} {season_type}: {e}")
                    continue
                if df.empty:
                    print(f"  No data for {api_season} {season_type}")
                    continue
                df.columns = [c.lower() for c in df.columns]
                df["season"] = api_season
                df["season_type"] = season_type
                if "game_date" in df.columns:
                    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
                if "player_id" in df.columns:
                    df["player_id"] = df["player_id"].astype(str)
                frames.append(df)
                print(f"  {api_season} {season_type}: {len(df)} rows")
                time.sleep(0.6)

        if frames:
            result = pd.concat(frames, ignore_index=True)
            print(f"  NBA total: {len(result)} rows across {len(frames)} season/season_type combos")
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
