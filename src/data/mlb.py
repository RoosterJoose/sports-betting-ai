from datetime import datetime, timedelta
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests

from src.data.base import DataSource

MLB_API = "https://statsapi.mlb.com/api/v1"
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache" / "mlb"

# Map MLB Stats API field names to internal feature names
HIT_MAP = {
    "hits": "h", "atBats": "ab", "runs": "r", "rbi": "rbi",
    "baseOnBalls": "bb", "strikeOuts": "so",
    "doubles": "2b", "triples": "3b", "homeRuns": "hr",
    "stolenBases": "sb", "caughtStealing": "cs",
    "avg": "avg", "obp": "obp", "slg": "slg", "ops": "ops",
    "totalBases": "tb", "babip": "babip",
    "hitByPitch": "hbp", "sacFlies": "sf",
    "groundIntoDoublePlay": "gidp",
}

PITCH_MAP = {
    "inningsPitched": "ip", "earnedRuns": "er",
    "strikeOuts": "so", "baseOnBalls": "bb",
    "hits": "h", "homeRuns": "hr",
    "wins": "w", "losses": "l", "saves": "sv",
    "era": "era", "whip": "whip",
    "blownSaves": "bsv", "holds": "hld",
    "gamesPitched": "g", "gamesStarted": "gs",
    "completeGames": "cg", "shutouts": "sho",
    "battersFaced": "bf", "outs": "outs",
    "wildPitches": "wp", "balks": "bk",
}


class MLBDataSource(DataSource):
    def __init__(self):
        self._cache = {}
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _api_get(self, path: str, params: dict = None) -> dict:
        url = f"{MLB_API}/{path}"
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def fetch_player_game_logs(self, seasons: list[str]) -> pd.DataFrame:
        cache_file = CACHE_DIR / f"game_logs_{'_'.join(seasons)}.parquet"
        if cache_file.exists():
            return pd.read_parquet(cache_file)

        # Try loading individual season cache files
        frames = []
        for s in seasons:
            f = CACHE_DIR / f"game_logs_{s}.parquet"
            if f.exists():
                frames.append(pd.read_parquet(f))
        frames = [f for f in frames if isinstance(f, pd.DataFrame) and not f.empty]
        if frames:
            return pd.concat(frames, ignore_index=True)

        frames = []
        for season in seasons:
            data = self._api_get("sports/1/players", {"season": season})
            people = data.get("people", [])
            print(f"  MLB {season}: {len(people)} players found")
            for i, person in enumerate(people):
                pid = person["id"]
                pos = person.get("primaryPosition", {}).get("abbreviation", "")
                group = "pitching" if pos in ("P", "RP", "SP", "CP") else "hitting"
                try:
                    gl = self._api_get(f"people/{pid}/stats", {
                        "stats": "gameLog", "season": season, "group": group,
                    })
                except Exception:
                    continue
                if not gl.get("stats"):
                    continue
                splits = gl["stats"][0].get("splits", [])
                column_map = PITCH_MAP if group == "pitching" else HIT_MAP
                for split in splits:
                    stat = split.get("stat", {})
                    if stat.get("gamesPlayed", 0) == 0:
                        continue
                    row = {}
                    for api_name, internal_name in column_map.items():
                        val = stat.get(api_name)
                        if val is not None:
                            if isinstance(val, str) and val in (".---", ".--- ", "-.--", "-.-- ", "", " ", ".---"):
                                continue
                            try:
                                row[internal_name] = float(val)
                            except (ValueError, TypeError):
                                if val not in (".---", ".--- ", "", None):
                                    row[internal_name] = val
                    if not row:
                        continue
                    row["player_id"] = str(pid)
                    row["player_name"] = person.get("fullName", "")
                    row["game_date"] = split.get("date", "")
                    row["season"] = season
                    row["position"] = pos
                    row["team_id"] = str(split.get("team", {}).get("id", ""))
                    row["opponent"] = split.get("opponent", {}).get("name", "")
                    row["opponent_id"] = str(split.get("opponent", {}).get("id", ""))
                    row["game_pk"] = split.get("game", {}).get("gamePk", 0)
                    row["home_or_away"] = "H" if split.get("isHome", False) else "A"
                    frames.append(row)
                if (i + 1) % 200 == 0:
                    print(f"    {i+1}/{len(people)} players processed")

        if not frames:
            return pd.DataFrame()
        df = pd.DataFrame(frames)
        # Compute singles (1b = h - 2b - 3b - hr)
        if "h" in df.columns and "2b" in df.columns and "3b" in df.columns and "hr" in df.columns:
            df["1b"] = df["h"] - df["2b"] - df["3b"] - df["hr"]
        df.to_parquet(cache_file, index=False)
        print(f"  MLB cached: {len(df)} player-game rows, {df['player_id'].nunique()} players")
        return df

    def fetch_schedule(self, season: str) -> pd.DataFrame:
        return self.fetch_player_game_logs([season])

    def fetch_player_stats(self, player_id: str, start_date, end_date) -> pd.DataFrame:
        return pd.DataFrame()

    def fetch_team_stats(self, team_id: str, start_date, end_date) -> pd.DataFrame:
        return pd.DataFrame()
