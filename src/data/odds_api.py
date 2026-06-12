"""The Odds API client for fetching moneyline odds across UFC, MLB, and NFL.

The Odds API (https://the-odds-api.com/) is a REST API that aggregates
odds from major US sportsbooks including DraftKings, FanDuel, BetMGM.

**Free tier**: 500 credits/month. Each `/odds` request costs 1 credit per
bookmaker-market combination returned. Sufficient for dev/testing.

**Limitation for UFC props**: The Odds API exposes UFC main market odds
(moneyline) but does NOT cover method-of-victory, round-of-finish, or
fight-goes-to-distance props. Use `dk_props_scraper.py` for those.

**MLB + NFL**: Standard moneyline (h2h) is supported. Spreads, totals, and
player props are also available but not exposed by this client (we use the
h2h market only for the model + paper-trading pipeline).

Usage:
    from src.data.odds_api import OddsAPIClient

    client = OddsAPIClient(api_key=os.environ.get("ODDS_API_KEY"))
    # UFC
    events = client.get_ufc_events()           # List of upcoming UFC events
    odds = client.get_ufc_odds(event_id)       # Moneyline odds for one event
    # odds = [{"sportsbook": "draftkings", "red_odds": -150, "blue_odds": +130, ...}, ...]
    # MLB
    mlb_events = client.get_mlb_events()       # List of upcoming MLB events
    mlb_odds = client.get_mlb_odds(event_id)   # Moneyline odds for one MLB game
    # mlb_odds = [{"sportsbook": "draftkings", "home_odds": -150, "away_odds": +130, ...}, ...]
    # NFL
    nfl_events = client.get_nfl_events()
    nfl_odds = client.get_nfl_odds(event_id)
"""

import json
import os
import time
import warnings
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

warnings.filterwarnings("ignore")

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEY_UFC = "mma_mixed_martial_arts"
SPORT_KEY_MLB = "baseball_mlb"
SPORT_KEY_NFL = "americanfootball_nfl"
DEFAULT_REGIONS = "us"
DEFAULT_MARKETS = "h2h"  # head-to-head = moneyline; the only UFC market The Odds API supports
DEFAULT_BOOKMAKERS = "draftkings,fanduel,betmgm"
DEFAULT_ODDS_FORMAT = "american"


class OddsAPIError(Exception):
    """Raised when The Odds API returns an error or is unreachable."""


class OddsAPIClient:
    """Thin client for The Odds API v4.

    No external dependencies — uses urllib only. Set ODDS_API_KEY env var
    or pass api_key to the constructor.

    Credit tracking: every call to get_ufc_odds() consumes 1 credit per
    bookmaker returned. The free tier is 500 credits/month.
    """

    def __init__(self, api_key: str | None = None, timeout: float = 15.0):
        self.api_key = api_key or os.environ.get("ODDS_API_KEY", "")
        if not self.api_key:
            raise OddsAPIError(
                "ODDS_API_KEY not set. Get a free key at https://the-odds-api.com/ "
                "and set ODDS_API_KEY env var, or pass api_key=... to the constructor."
            )
        self.timeout = timeout
        # Track credits remaining across calls so the caller can warn when low
        self.credits_remaining: int | None = None
        self.credits_used: int | None = None

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        """Make a GET request to The Odds API and return parsed JSON.

        Updates `credits_remaining` and `credits_used` from response headers.
        Raises OddsAPIError on non-2xx status or network failure.
        """
        params = {**params, "apiKey": self.api_key}
        url = f"{BASE_URL}{path}?{urlencode(params)}"
        try:
            req = Request(url, headers={"User-Agent": "sports-betting-ai/1.0"})
            with urlopen(req, timeout=self.timeout) as resp:
                # Track credit usage from response headers
                if "x-requests-remaining" in resp.headers:
                    self.credits_remaining = int(resp.headers["x-requests-remaining"])
                if "x-requests-used" in resp.headers:
                    self.credits_used = int(resp.headers["x-requests-used"])
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            raise OddsAPIError(f"GET {path} failed: {e}") from e

    def get_ufc_events(self) -> list[dict]:
        """Return list of upcoming UFC events with id, sport_title, commence_time, home/away teams.

        Each dict has shape:
            {
                "id": "abc123...",
                "sport_key": "mma_mixed_martial_arts",
                "sport_title": "MMA",
                "commence_time": "2026-06-14T23:00:00Z",
                "home_team": "Ilia Topuria",
                "away_team": "Justin Gaethje",
            }
        Note: for MMA, "home_team"/"away_team" are just the two fighters (no home/away concept).
        """
        return self._get("/sports/mma_mixed_martial_arts/events", {})

    def get_ufc_odds(
        self,
        event_id: str,
        regions: str = DEFAULT_REGIONS,
        bookmakers: str = DEFAULT_BOOKMAKERS,
        odds_format: str = DEFAULT_ODDS_FORMAT,
    ) -> list[dict]:
        """Return list of bookmaker odds for one UFC event.

        Each dict has shape:
            {
                "sportsbook": "draftkings",   # one of bookmakers filter
                "commence_time": "2026-06-14T23:00:00Z",
                "home_team": "Ilia Topuria",
                "away_team": "Justin Gaethje",
                "red_odds": -150,             # American odds for red/first fighter
                "blue_odds": +130,            # American odds for blue/second fighter
            }

        If a bookmaker doesn't offer odds for the event, it's omitted from
        the list (the API only returns bookmakers that have active odds).
        """
        raw = self._get(
            f"/sports/mma_mixed_martial_arts/events/{event_id}/odds",
            {
                "regions": regions,
                "markets": DEFAULT_MARKETS,  # h2h only — no UFC props in The Odds API
                "bookmakers": bookmakers,
                "oddsFormat": odds_format,
                "dateFormat": "iso",
            },
        )
        return [_parse_h2h_outcome(book) for book in raw]

    def get_mlb_events(self) -> list[dict]:
        """Return list of upcoming MLB events with id, sport_title, commence_time, home/away teams.

        Each dict has shape:
            {
                "id": "abc123...",
                "sport_key": "baseball_mlb",
                "sport_title": "MLB",
                "commence_time": "2026-06-12T23:10:00Z",
                "home_team": "Baltimore Orioles",
                "away_team": "Tampa Bay Rays",
            }
        """
        return self._get(f"/sports/{SPORT_KEY_MLB}/events", {})

    def get_mlb_odds(
        self,
        event_id: str,
        regions: str = DEFAULT_REGIONS,
        markets: str = DEFAULT_MARKETS,
        bookmakers: str = DEFAULT_BOOKMAKERS,
        odds_format: str = DEFAULT_ODDS_FORMAT,
    ) -> list[dict]:
        """Return list of bookmaker odds for one MLB event.

        Each dict has shape:
            {
                "sportsbook": "draftkings",
                "commence_time": "2026-06-12T23:10:00Z",
                "home_team": "Baltimore Orioles",
                "away_team": "Tampa Bay Rays",
                "home_odds": -150,             # American odds for home team
                "away_odds": +130,             # American odds for away team
            }

        If a bookmaker doesn't offer odds for the event, it's omitted from
        the list (the API only returns bookmakers that have active odds).

        `markets` defaults to "h2h" (moneyline). The Odds API also supports
        "spreads" and "totals" for MLB — pass a comma-separated string to
        fetch multiple markets in one call.
        """
        raw = self._get(
            f"/sports/{SPORT_KEY_MLB}/events/{event_id}/odds",
            {
                "regions": regions,
                "markets": markets,
                "bookmakers": bookmakers,
                "oddsFormat": odds_format,
                "dateFormat": "iso",
            },
        )
        return [_parse_h2h_team_outcome(book) for book in raw]

    def get_nfl_events(self) -> list[dict]:
        """Return list of upcoming NFL events with id, sport_title, commence_time, home/away teams.

        Each dict has shape:
            {
                "id": "abc123...",
                "sport_key": "americanfootball_nfl",
                "sport_title": "NFL",
                "commence_time": "2026-09-10T00:20:00Z",
                "home_team": "Kansas City Chiefs",
                "away_team": "Baltimore Ravens",
            }
        """
        return self._get(f"/sports/{SPORT_KEY_NFL}/events", {})

    def get_nfl_odds(
        self,
        event_id: str,
        regions: str = DEFAULT_REGIONS,
        markets: str = DEFAULT_MARKETS,
        bookmakers: str = DEFAULT_BOOKMAKERS,
        odds_format: str = DEFAULT_ODDS_FORMAT,
    ) -> list[dict]:
        """Return list of bookmaker odds for one NFL event.

        Each dict has shape:
            {
                "sportsbook": "draftkings",
                "commence_time": "2026-09-10T00:20:00Z",
                "home_team": "Kansas City Chiefs",
                "away_team": "Baltimore Ravens",
                "home_odds": -150,             # American odds for home team
                "away_odds": +130,             # American odds for away team
            }

        If a bookmaker doesn't offer odds for the event, it's omitted from
        the list (the API only returns bookmakers that have active odds).

        `markets` defaults to "h2h" (moneyline). The Odds API also supports
        "spreads" and "totals" for NFL — pass a comma-separated string to
        fetch multiple markets in one call.
        """
        raw = self._get(
            f"/sports/{SPORT_KEY_NFL}/events/{event_id}/odds",
            {
                "regions": regions,
                "markets": markets,
                "bookmakers": bookmakers,
                "oddsFormat": odds_format,
                "dateFormat": "iso",
            },
        )
        return [_parse_h2h_team_outcome(book) for book in raw]


def _parse_h2h_outcome(book: dict) -> dict:
    """Parse one bookmaker's h2h (moneyline) response into a flat dict.

    Input shape (from The Odds API):
        {
            "key": "draftkings",
            "title": "DraftKings",
            "markets": [{"key": "h2h", "outcomes": [
                {"name": "Ilia Topuria", "price": -150},
                {"name": "Justin Gaethje", "price": +130},
            ]}],
            "commence_time": "2026-06-14T23:00:00Z",
        }
    """
    market = book.get("markets", [{}])[0]
    outcomes = market.get("outcomes", [])
    if len(outcomes) < 2:
        return {
            "sportsbook": book.get("key", "unknown"),
            "commence_time": book.get("commence_time", ""),
            "home_team": book.get("home_team", ""),
            "away_team": book.get("away_team", ""),
            "red_odds": None,
            "blue_odds": None,
        }
    # The API doesn't guarantee which is "red" vs "blue" — convention is
    # the first outcome is the favorite (or the alphabetically first team).
    # For UFC, there's no red/blue in the API; we map home/away to red/blue
    # for downstream consistency with the rest of our pipeline.
    return {
        "sportsbook": book.get("key", "unknown"),
        "commence_time": book.get("commence_time", ""),
        "home_team": book.get("home_team", ""),
        "away_team": book.get("away_team", ""),
        "red_odds": outcomes[0].get("price"),   # home/first listed
        "blue_odds": outcomes[1].get("price"),  # away/second listed
    }


def _parse_h2h_team_outcome(book: dict) -> dict:
    """Parse one bookmaker's h2h response into a flat dict for team sports (MLB, NFL).

    Unlike UFC, team sports have a real home/away structure. We use a
    name-to-odds lookup so the parser is robust to outcome ordering (the
    API doesn't guarantee home_team appears first in the outcomes list).

    Input shape (from The Odds API):
        {
            "key": "draftkings",
            "title": "DraftKings",
            "markets": [{"key": "h2h", "outcomes": [
                {"name": "Baltimore Orioles", "price": -150},
                {"name": "Tampa Bay Rays", "price": +130},
            ]}],
            "commence_time": "2026-06-12T23:10:00Z",
            "home_team": "Baltimore Orioles",
            "away_team": "Tampa Bay Rays",
        }
    """
    market = book.get("markets", [{}])[0]
    outcomes = market.get("outcomes", [])

    home_team = book.get("home_team", "")
    away_team = book.get("away_team", "")
    name_to_price = {o.get("name", ""): o.get("price") for o in outcomes}

    if len(outcomes) < 2:
        return {
            "sportsbook": book.get("key", "unknown"),
            "commence_time": book.get("commence_time", ""),
            "home_team": home_team,
            "away_team": away_team,
            "home_odds": None,
            "away_odds": None,
        }

    return {
        "sportsbook": book.get("key", "unknown"),
        "commence_time": book.get("commence_time", ""),
        "home_team": home_team,
        "away_team": away_team,
        "home_odds": name_to_price.get(home_team),
        "away_odds": name_to_price.get(away_team),
    }


def american_odds_to_implied_prob(odds: int | float | None) -> float | None:
    """Convert American odds to implied probability (vig included).

    Positive odds:  p = 100 / (odds + 100)
    Negative odds:  p = -odds / (-odds + 100)
    Returns None if odds is None or 0.
    """
    if odds is None or odds == 0:
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return -odds / (-odds + 100.0)


def american_odds_to_fair_prob(red_odds: int | float, blue_odds: int | float) -> tuple[float, float]:
    """Convert a 2-outcome American odds pair to fair probabilities (vig removed).

    Uses the standard "remove the vig" approach: scale each implied
    probability so they sum to 1.0. This approximates a fair line where
    the bookmaker's edge (vig/overround) is removed.
    """
    p_red = american_odds_to_implied_prob(red_odds) or 0.5
    p_blue = american_odds_to_implied_prob(blue_odds) or 0.5
    total = p_red + p_blue
    if total <= 0:
        return 0.5, 0.5
    return p_red / total, p_blue / total


if __name__ == "__main__":
    # Smoke test: print upcoming UFC events + moneyline odds
    try:
        client = OddsAPIClient()
        events = client.get_ufc_events()
        print(f"Found {len(events)} upcoming UFC events")
        for ev in events[:3]:
            print(f"  {ev.get('commence_time', '?')}  {ev.get('home_team', '?')} vs {ev.get('away_team', '?')}")
        if events:
            print(f"\nOdds for first event ({events[0]['id']}):")
            odds = client.get_ufc_odds(events[0]["id"])
            for o in odds:
                print(f"  {o['sportsbook']:12s}  red={o['red_odds']}  blue={o['blue_odds']}")
        print(f"\nCredits remaining: {client.credits_remaining}")
    except OddsAPIError as e:
        print(f"OddsAPIError: {e}")
        print("(Set ODDS_API_KEY env var to enable. Free key at https://the-odds-api.com/)")
