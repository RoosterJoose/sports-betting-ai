from datetime import datetime
from typing import Optional

import pandas as pd

from src.data.base import DataSource


class SoccerDataSource(DataSource):
    def __init__(self):
        self._cache = {}

    def fetch_player_stats(
        self, player_id: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        try:
            from understat import Understat
            import asyncio
        except ImportError:
            raise ImportError("pip install understat")

        async def _fetch():
            async with Understat() as us:
                player = await us.get_player_stats(int(player_id))
                return player
        return pd.DataFrame(asyncio.run(_fetch()))

    def fetch_team_stats(
        self, team_id: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        try:
            import requests
        except ImportError:
            raise ImportError("pip install requests")

        league_id = self._team_to_league(team_id)
        season = start_date.year
        url = f"https://understat.com/league/{league_id}/{season}"
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        tables = pd.read_html(resp.text)
        return tables[0] if tables else pd.DataFrame()

    def fetch_fixtures(self, league_id: str, season: str) -> pd.DataFrame:
        try:
            from understat import Understat
            import asyncio
        except ImportError:
            raise ImportError("pip install understat")

        async def _fetch():
            async with Understat() as us:
                fixtures = await us.get_league_fixtures(league_id, int(season))
                return fixtures
        return pd.DataFrame(asyncio.run(_fetch()))

    def fetch_schedule(self, season: str) -> pd.DataFrame:
        return pd.DataFrame()

    @staticmethod
    def _team_to_league(team_id: str) -> int:
        league_map = {
            "EPL": 1, "La_Liga": 2, "Bundesliga": 3,
            "Serie_A": 4, "Ligue_1": 5, "RFPL": 6,
            "MLS": 7, "EPL_23": 8,
        }
        return league_map.get(team_id, 1)


class SoccerFBRefScraper(DataSource):
    def __init__(self):
        self._base = "https://fbref.com/en/comps"

    def fetch_player_stats(
        self, player_id: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        try:
            import requests
        except ImportError:
            raise ImportError("pip install requests")

        league_code = self._league_for_date(start_date)
        season = self._season_str(start_date)
        url = f"{self._base}/{league_code}/{season}/stats/{season}-{league_code}-stats"
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        tables = pd.read_html(resp.text)
        if not tables:
            return pd.DataFrame()
        df = tables[0]
        df.columns = ["_".join(col).strip() for col in df.columns]
        return df

    def fetch_team_stats(
        self, team_id: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        return pd.DataFrame()

    def fetch_schedule(self, season: str) -> pd.DataFrame:
        return pd.DataFrame()

    @staticmethod
    def _league_for_date(d: datetime) -> str:
        return "9"  # Premier League

    @staticmethod
    def _season_str(d: datetime) -> str:
        return f"{d.year}-{d.year + 1}"
