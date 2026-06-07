from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data.base import DataSource

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "wnba_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class WNBADataSource(DataSource):
    def __init__(self):
        self._cache = {}

    def fetch_player_stats(self, player_id: str, start_date, end_date) -> pd.DataFrame:
        return pd.DataFrame()

    def fetch_team_stats(self, team_id: str, start_date, end_date) -> pd.DataFrame:
        return pd.DataFrame()

    def fetch_schedule(self, season: str) -> pd.DataFrame:
        return pd.DataFrame()

    def fetch_player_game_logs(self, seasons: list[str]) -> pd.DataFrame:
        cache_path = CACHE_DIR / "wnba_games.parquet"
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            self._cache["games"] = df
            print(f"  WNBA: {len(df)} rows from cache")
            return df

        try:
            from nba_api.stats.endpoints import leaguegamefinder
        except ImportError:
            raise ImportError("pip install nba_api")

        target_seasons = set()
        for s in seasons:
            try:
                (int(s))
                target_seasons.add(f"2{int(s)}")
            except ValueError:
                pass

        finder = leaguegamefinder.LeagueGameFinder(
            timeout=60,
            league_id_nullable="10",
        )
        try:
            df = finder.get_data_frames()[0]
        except Exception as e:
            print(f"  WNBA fetch failed: {e}")
            return pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()

        df.columns = [c.lower() for c in df.columns]
        df["game_date"] = pd.to_datetime(df["game_date"])
        df["player_id"] = df["team_id"].astype(str)

        # Filter to requested seasons
        if target_seasons:
            df = df[df["season_id"].isin(target_seasons)]

        df.to_parquet(cache_path)
        self._cache["games"] = df
        print(f"  WNBA: {len(df)} rows, {df.player_id.nunique()} teams, seasons={sorted(df['season_id'].unique())}")
        return df
