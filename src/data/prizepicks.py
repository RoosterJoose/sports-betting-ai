import hashlib
import json
import os
import time
import uuid
from datetime import datetime
from typing import Optional

import httpx
import pandas as pd

from src.data.base import BookDataSource

LEAGUE_IDS: dict[str, int] = {
    "nfl": 9,
    "mlb": 2,
    "nba": 7,
    "nhl": 8,
    "cfb": 15,
    "cbb": 20,
    "golf": 1,
    "soccer": 82,
    "nascar": 4,
    "tennis": 5,
    "wnba": 3,
    "ufc": 12,
    "epl": 14,
    "liv": 228,
    "lpga": 256,
}

SUB_GAME_LEAGUES = {84: "nba_1h", 192: "nba_1q", 80: "nba_2h",
                    35: "nfl_1h", 25: "nfl_2h", 245: "nfl_1q", 152: "nfl_4q",
                    227: "nhl_1p", 234: "nhl_2p", 226: "nhl_3p",
                    242: "soccer_1h", 243: "soccer_2h",
                    193: "wnba_1h", 194: "wnba_2h", 195: "wnba_4q", 308: "wnba_1q"}

BASE_URL = "https://api.prizepicks.com"
DEFAULT_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Referer": "https://app.prizepicks.com/",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Dest": "empty",
}


def _device_id() -> str:
    return str(uuid.uuid4())


class PrizePicksScraper(BookDataSource):
    def __init__(self, proxy: Optional[str] = None):
        kwargs = dict(
            headers=DEFAULT_HEADERS,
            http2=True,
            timeout=30,
        )
        if proxy:
            kwargs["proxies"] = {"all://": proxy}
        self.client = httpx.Client(**kwargs)
        self.client.headers["X-Device-ID"] = _device_id()
        self._last_request = 0.0

    def _request(self, url: str, **kwargs) -> httpx.Response:
        now = time.time()
        elapsed = now - self._last_request
        if elapsed < 1.5:
            time.sleep(1.5 - elapsed)
        self._last_request = time.time()

        for attempt in range(3):
            resp = self.client.get(url, **kwargs)
            if resp.status_code == 403:
                raise RuntimeError(
                    "PrizePicks blocked the request (403). "
                    "Try: residential IP, HTTP/2 client, or rotating headers. "
                    "Or use a data provider like dailyfantasyapi.io or SharpAPI."
                )
            if resp.status_code == 429:
                wait = 5 * (attempt + 1)
                print(f"  Rate limited (429), waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        raise RuntimeError(f"Failed after 3 retries, last status: {resp.status_code}")

    def fetch_lines(self, sport: str, league_id: int = None) -> pd.DataFrame:
        if league_id is None:
            league_id = LEAGUE_IDS.get(sport)
        if not league_id:
            raise ValueError(f"Unknown sport: {sport}. Known: {list(LEAGUE_IDS.keys())}")

        projections, included = self._fetch_projections(league_id)
        if projections.empty:
            return projections

        df = self._normalize(projections)

        # Resolve player names from included resources
        player_map = {}
        for item in included:
            if item.get("type") == "new_player":
                attrs = item.get("attributes", {})
                player_map[item["id"]] = attrs.get("display_name", "")
        df["player_name"] = df.get("new_player_data_id", "").astype(str).map(player_map)

        # Resolve league names
        league_map = {}
        for item in included:
            if item.get("type") == "league":
                attrs = item.get("attributes", {})
                league_map[item["id"]] = attrs.get("name", "")
        df["league_name"] = df.get("league_data_id", "").astype(str).map(league_map)

        return df

    def fetch_settlements(self, sport: str, date: datetime) -> pd.DataFrame:
        raise NotImplementedError(
            "PrizePicks does not provide settlement data via API. "
            "Use nba_api / pybaseball / nfl_data_py to get actual box scores "
            "and compare against historical projections."
        )

    def fetch_all_sports(self) -> dict[str, pd.DataFrame]:
        results = {}
        for sport_name in LEAGUE_IDS:
            try:
                results[sport_name] = self.fetch_lines(sport_name)
            except Exception as e:
                print(f"Failed to fetch {sport_name}: {e}")
        return results

    def _fetch_projections(self, league_id: int) -> tuple[pd.DataFrame, list]:
        params = {
            "league_id": league_id,
            "per_page": 250,
            "single_stat": "true",
            "game_mode": "pickem",
        }
        all_data = []
        all_included = []
        page = 1
        while True:
            resp = self._request(f"{BASE_URL}/projections", params={**params, "page": page})
            payload = resp.json()
            batch = payload.get("data", [])
            if not batch:
                break
            all_data.extend(batch)
            if not all_included:
                all_included = payload.get("included", [])
            meta = payload.get("meta", {})
            if page >= meta.get("total_pages", 1):
                break
            page += 1
            time.sleep(0.5)

        return pd.json_normalize(all_data, sep="_"), all_included

    def _normalize(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()
        for col in df.columns:
            df.rename(columns={col: col.replace("attributes_", "").replace("relationships_", "")}, inplace=True)

        str_cols = ["board_time", "updated_at"]
        for c in str_cols:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")

        float_cols = ["line_score", "flash_sale_line_score", "discount_percentage"]
        for c in float_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        df["fetched_at"] = datetime.utcnow()
        return df


def get_prizepicks_client() -> BookDataSource:
    """Factory: returns a PrizePicks scraper instance.

    SharpAPI doesn't support PrizePicks data (only sportsbook odds),
    so direct scraping is our only option for PrizePicks lines.
    """
    return PrizePicksScraper()
