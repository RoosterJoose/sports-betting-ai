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

    def fetch_player_game_logs(self, seasons: list[str] = None) -> pd.DataFrame:
        cache_path = CACHE_DIR / "wnba_games.parquet"
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            self._cache["games"] = df
            if "player_name" in df.columns:
                print(f"  WNBA: {len(df)} rows, {df.player_id.nunique()} players from cache")
            else:
                print(f"  WNBA: {len(df)} rows from cache (no player_name — likely old team-level data)")
            return df

        try:
            from nba_api.stats.endpoints import playergamelogs
        except ImportError:
            raise ImportError("pip install nba_api")

        # Map pipeline seasons to WNBA season years
        # WNBA season year = the year the season starts (e.g. 2025 for 2025 season)
        import time
        target_years = set()
        for s in seasons:
            try:
                sy = int(str(s)[:4])
                # For WNBA, season is the starting year
                target_years.add(str(sy))
            except ValueError:
                pass

        if not target_years:
            target_years = {"2025", "2026"}

        frames = []
        for year in sorted(target_years):
            api_season = year  # WNBA seasons are single-year (e.g., "2025")
            print(f"  Fetching WNBA {api_season} via PlayerGameLogs...", flush=True)
            try:
                logs = playergamelogs.PlayerGameLogs(
                    league_id_nullable="10",
                    season_nullable=api_season,
                    season_type_nullable="Regular Season",
                    timeout=60,
                )
                df = logs.get_data_frames()[0]
                if df is None or df.empty:
                    print(f"  No data for {api_season}")
                    continue
                df.columns = [c.lower() for c in df.columns]
                df["season"] = api_season
                if "game_date" in df.columns:
                    df["game_date"] = pd.to_datetime(df["game_date"])
                if "player_id" in df.columns:
                    df["player_id"] = df["player_id"].astype(str)
                # Ensure standard columns exist
                for col in ["pts", "reb", "ast", "stl", "blk", "tov", "fg3m", "fg3a", "fgm", "fga", "ftm", "fta", "min"]:
                    if col not in df.columns:
                        df[col] = 0
                frames.append(df)
                print(f"  WNBA {api_season}: {len(df)} rows, {df.player_id.nunique()} players", flush=True)
                time.sleep(1.5)
            except Exception as e:
                print(f"  WNBA {api_season} fetch failed: {e}", flush=True)

        if not frames:
            print("  No WNBA data available")
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        print(f"  WNBA total: {len(result)} rows, {result.player_id.nunique()} players across {len(frames)} seasons", flush=True)
        result.to_parquet(cache_path)
        self._cache["games"] = result.copy()
        return result
