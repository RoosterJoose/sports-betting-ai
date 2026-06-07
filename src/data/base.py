from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional
import pandas as pd


class DataSource(ABC):
    @abstractmethod
    def fetch_player_stats(
        self, player_id: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        ...

    @abstractmethod
    def fetch_team_stats(
        self, team_id: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        ...

    @abstractmethod
    def fetch_schedule(self, season: str) -> pd.DataFrame:
        ...

    def fetch_player_game_logs(self, seasons: list[str]) -> pd.DataFrame:
        """
        Return per-player per-game stat lines for training.
        Override in each sport with sport-specific data fetching.
        Default: calls fetch_schedule for each season and concatenates.
        """
        frames = []
        for s in seasons:
            df = self.fetch_schedule(s)
            if isinstance(df, pd.DataFrame) and not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True)
        return result


class BookDataSource(ABC):
    @abstractmethod
    def fetch_lines(self, sport: str) -> pd.DataFrame:
        ...

    @abstractmethod
    def fetch_settlements(self, sport: str, date: datetime) -> pd.DataFrame:
        ...
