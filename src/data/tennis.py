from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.base import DataSource

SACKMANN_ATP = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"
SACKMANN_WTA = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master"


class TennisDataSource(DataSource):
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir
        self._cache = {}

    def fetch_player_stats(
        self, player_id: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        try:
            import requests
        except ImportError:
            raise ImportError("pip install requests")

        player_id_str = str(player_id)
        atp_url = f"{SACKMANN_ATP}/atp_matches_{start_date.year}.csv"
        wta_url = f"{SACKMANN_WTA}/wta_matches_{start_date.year}.csv"

        frames = []
        for url in [atp_url, wta_url]:
            try:
                df = pd.read_csv(url)
                mask = (
                    (df["winner_id"].astype(str) == player_id_str) |
                    (df["loser_id"].astype(str) == player_id_str)
                )
                matches = df[mask].copy()
                if not matches.empty:
                    matches["player_id"] = player_id_str
                    frames.append(matches)
            except Exception:
                continue

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined["tourney_date"] = pd.to_datetime(combined["tourney_date"], format="%Y%m%d", errors="coerce")
        if "tourney_date" in combined.columns:
            mask = (combined["tourney_date"] >= start_date) & (combined["tourney_date"] <= end_date)
            combined = combined[mask]
        return combined

    def fetch_team_stats(
        self, team_id: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        return pd.DataFrame()

    def fetch_schedule(self, season: str) -> pd.DataFrame:
        try:
            atp = pd.read_csv(f"{SACKMANN_ATP}/atp_matches_{season}.csv")
            wta = pd.read_csv(f"{SACKMANN_WTA}/wta_matches_{season}.csv")
            return pd.concat([atp, wta], ignore_index=True)
        except Exception:
            return pd.DataFrame()

    def fetch_player_rankings(self, date: datetime) -> pd.DataFrame:
        try:
            rankings = pd.read_csv(f"{SACKMANN_ATP}/atp_rankings_current.csv")
            rankings["ranking_date"] = pd.to_datetime(rankings["ranking_date"], format="%Y%m%d")
            return rankings[rankings["ranking_date"] <= date].head(100)
        except Exception:
            return pd.DataFrame()

    def fetch_atp_matches(self, year: int) -> pd.DataFrame:
        return pd.read_csv(f"{SACKMANN_ATP}/atp_matches_{year}.csv")

    def fetch_wta_matches(self, year: int) -> pd.DataFrame:
        return pd.read_csv(f"{SACKMANN_WTA}/wta_matches_{year}.csv")
