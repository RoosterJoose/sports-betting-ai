import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

import pandas as pd
import requests

from src.data.base import DataSource

# Map config stat types to DataFrame column names
GOLF_STAT_MAP = {
    "scoring_avg": "scoring_avg_avg",
    "driving_dist": "driving_dist_avg",
    "gir": "gir_avg",
    "scrambling": "scrambling_avg",
    "putting_avg": "putting_avg_avg",
    "sg_ott": "sg_ott_points",
    "sg_app": "sg_app_points",
}


CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "golf_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

STAT_IDS: dict[str, str] = {
    "scoring_avg": "02568",
    "driving_dist": "02674",
    "driving_acc": "02675",
    "gir": "02568",
    "scrambling": "02569",
    "putting_avg": "02567",
    "sg_putt": "02754",
    "sg_ott": "02708",
    "sg_app": "02709",
    "sg_t2g": "02705",
    "sg_total": "02677",
    "birdie_avg": "02702",
    "par_breakers": "02678",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}


def _fetch_stat_page(stat_id: str, year: str) -> list[dict[str, Any]]:
    r = requests.get(
        f"https://www.pgatour.com/stats/stat.{stat_id}.html",
        headers=HEADERS,
        timeout=15,
    )
    if r.status_code != 200:
        return []
    match = re.search(
        r'<script id="__NEXT_DATA__" type="application/json"[^>]*>(.*?)</script>',
        r.text,
        re.DOTALL,
    )
    if not match:
        return []
    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        return []
    queries = data.get("props", {}).get("pageProps", {}).get("dehydratedState", {}).get("queries", [])
    for q in queries:
        qk = q.get("queryKey", [])
        if isinstance(qk, list) and len(qk) > 1 and isinstance(qk[1], dict):
            params = qk[1]
            if (
                params.get("statId") == stat_id
                and params.get("year") == int(year)
                and params.get("eventQuery") is None
            ):
                return q["state"]["data"].get("rows", [])
    return []


class GolfDataSource(DataSource):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def fetch_player_stats(self, player_id: str, start_date, end_date) -> pd.DataFrame:
        return pd.DataFrame()

    def fetch_team_stats(self, team_id, start_date, end_date) -> pd.DataFrame:
        return pd.DataFrame()

    def fetch_schedule(self, season: str) -> pd.DataFrame:
        return self.fetch_player_season_stats([season])

    def fetch_player_game_logs(self, seasons: list[str]) -> pd.DataFrame:
        return self.fetch_player_season_stats(seasons)

    def fetch_player_season_stats(self, seasons: list[str]) -> pd.DataFrame:
        cache_path = CACHE_DIR / "season_stats_v4.parquet"
        if cache_path.exists():
            print(f"  Loading Golf from cache: {cache_path}")
            df = pd.read_parquet(cache_path)
            # Add PrizePicks-compatible column aliases
            df["player_id"] = df["player_id"].astype(str)
            for alias, col in GOLF_STAT_MAP.items():
                if col in df.columns and alias not in df.columns:
                    df[alias] = df[col]
            return df

        if not seasons:
            seasons = ["2026"]

        print(f"  Fetching PGA Tour stats for {len(seasons)} seasons...")

        frames = []
        for season in seasons:
            year = str(season)[:4]
            season_data: dict[str, dict] = {}
            player_names: dict[str, str] = {}

            for stat_name, stat_id in STAT_IDS.items():
                rows = _fetch_stat_page(stat_id, year)
                if not rows:
                    continue
                count = 0
                for row in rows:
                    pid = row.get("playerId") or row.get("player_id")
                    if not pid:
                        continue
                    pid_str = str(pid)
                    player_names[pid_str] = row.get("playerName", "")
                    if pid_str not in season_data:
                        season_data[pid_str] = {"player_id": pid_str, "season": year}
                    for s in row.get("stats", []):
                        col = f"{stat_name}_{s.get('statName', 'value').lower()}"
                        raw = s.get("statValue", "")
                        try:
                            val = float(raw.replace(",", "").replace("$", "").replace("%", ""))
                        except (ValueError, AttributeError):
                            val = raw
                        season_data[pid_str][col] = val
                    count += 1
                print(f"    {stat_name}: {count} players")

            if season_data:
                df = pd.DataFrame(list(season_data.values()))
                df["player_name"] = df["player_id"].map(player_names)
                df["game_date"] = pd.to_datetime(f"{year}-07-01")
                frames.append(df)
                print(f"  {year}: {len(df)} players, {len(df.columns)} columns")

        if frames:
            result = pd.concat(frames, ignore_index=True)
            print(f"  Golf: {len(result)} player-season rows, {len(result.columns)} stat columns")
            result.to_parquet(cache_path)
            return result

        print("  No Golf data available")
        return pd.DataFrame()
