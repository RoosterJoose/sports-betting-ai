"""FotMob lineup scraper + JSON cache for World Cup 2026 starting XIs.

Addresses Q2d from research/world_cup_nfl_research.md: "Player impact: For
World Cup specifically, how important are individual star players vs
team-level metrics? Is there a published approach for incorporating player
availability (injuries, suspensions)?"

Usage:
    from src.data.fotmob import FotMobScraper, LineupCache

    cache = LineupCache()  # data/cache/worldcup/lineups.json
    scraper = FotMobScraper()
    try:
        lineups = scraper.fetch_lineups(fotmob_match_id=4213275)
        # lineups = {"home": [...], "away": [...], "fetched_at": "..."}
        cache.set("KXWCGAME-26JUN11MEXCUB-MEX", lineups)
    finally:
        scraper.close()

    cached = cache.get("KXWCGAME-26JUN11MEXCUB-MEX")

The cache key is the Kalshi WC ticker (e.g. "KXWCGAME-26JUN11MEXCUB-MEX").
FotMob numeric match IDs are mapped separately (data/cache/worldcup/fotmob_ids.json).

Note: Per NotebookLM research, FotMob publishes confirmed lineups ~60 min
before kickoff. Scraping earlier returns the previous match's lineup or empty.
"""
import json
import re
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "worldcup"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LINEUP_CACHE_FILE = CACHE_DIR / "lineups.json"
FOTMOB_ID_MAP_FILE = CACHE_DIR / "fotmob_ids.json"
STAR_PLAYERS_FILE = PROJECT_ROOT / "data" / "wc_star_players.json"

# Default cache TTL: 6 hours (lineups confirmed ~60 min before kickoff,
# match completes within 2 hours, so 6h covers the full match lifecycle)
DEFAULT_CACHE_TTL_HOURS = 6


class LineupCache:
    """JSON-file cache for confirmed lineups, keyed by match ticker.

    File format: {"TICKER": {home: [...], away: [...], fetched_at: ISO, kickoff: ISO}}
    """

    def __init__(self, path: Path = LINEUP_CACHE_FILE):
        self.path = path
        self._data: dict = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        if self.path.exists():
            try:
                with open(self.path) as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._data = {}
        self._loaded = True

    def _save(self) -> None:
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)

    def get(self, ticker: str) -> Optional[dict]:
        self._load()
        return self._data.get(ticker)

    def set(self, ticker: str, lineups: dict) -> None:
        self._load()
        if "fetched_at" not in lineups:
            lineups["fetched_at"] = datetime.now().isoformat()
        self._data[ticker] = lineups
        self._save()

    def is_fresh(self, ticker: str, max_age_hours: float = DEFAULT_CACHE_TTL_HOURS) -> bool:
        """Return True if cached lineup is younger than max_age_hours."""
        entry = self.get(ticker)
        if not entry:
            return False
        fetched_at = entry.get("fetched_at")
        if not fetched_at:
            return False
        try:
            t = datetime.fromisoformat(fetched_at)
        except ValueError:
            return False
        return (datetime.now() - t) < timedelta(hours=max_age_hours)

    def all(self) -> dict:
        self._load()
        return dict(self._data)


def load_star_players() -> dict[str, list[str]]:
    """Load the top WC 2026 star players from data/wc_star_players.json.

    Returns: {team_code: [player_name, ...]}
    Strips the _comment/_version/_updated metadata keys.
    """
    if not STAR_PLAYERS_FILE.exists():
        return {}
    with open(STAR_PLAYERS_FILE) as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def compute_key_player_out(
    home_team: str,
    away_team: str,
    lineups: dict,
    star_players: Optional[dict[str, list[str]]] = None,
) -> int:
    """Return 1 if any star player from either team is missing from the confirmed XI.

    Parameters
    ----------
    home_team, away_team : str
        3-letter team codes matching WC2026_TEAMS (e.g. "ARG", "BRA", "USA").
    lineups : dict
        {"home": [{"name": "Messi", ...}, ...], "away": [...]} from the cache.
    star_players : dict, optional
        Override the default star players list. Format: {team: [name, ...]}.

    Returns
    -------
    int
        1 if a star is missing from either team's XI, else 0.

    Notes
    -----
    "Missing" means the star's name does not appear in the lineup list.
    Name matching is case-insensitive and uses a contains-substring match
    (so "Messi" matches "Lionel Messi", "Mbappé" matches "Kylian Mbappé").
    """
    if not lineups:
        return 0
    stars = star_players if star_players is not None else load_star_players()
    if not stars:
        return 0

    flag = 0
    for team, key in [(home_team, "home"), (away_team, "away")]:
        team_stars = stars.get(team, [])
        if not team_stars:
            continue
        xi_names = [p.get("name", "") for p in lineups.get(key, [])]
        xi_lower = [n.lower() for n in xi_names]
        for star in team_stars:
            star_lower = star.lower()
            # Substring match: "Messi" in "Lionel Messi" → True
            if not any(star_lower in n for n in xi_lower):
                flag = 1
                break
    return flag


def map_kalshi_to_fotmob_id(ticker: str) -> Optional[int]:
    """Look up a FotMob match ID by Kalshi ticker.

    Map stored in data/cache/worldcup/fotmob_ids.json as
    {"KXWCGAME-26JUN11MEXCUB-MEX": 4213275, ...}
    """
    if not FOTMOB_ID_MAP_FILE.exists():
        return None
    with open(FOTMOB_ID_MAP_FILE) as f:
        mapping = json.load(f)
    return mapping.get(ticker)


class FotMobScraper:
    """Playwright-based FotMob lineup scraper.

    Tries the public matchDetails API first (fast, returns JSON). Falls
    back to rendering the match page and reading the lineup section.
    The API approach is preferred because it's faster and less likely
    to be blocked by Cloudflare.
    """

    # /api/matchDetails returns 404; the live endpoint is /api/data/matchDetails
    # (discovered June 10 via Playwright network interception against
    # https://www.fotmob.com/matches?date=20260608)
    API_URL = "https://www.fotmob.com/api/data/matchDetails"
    BASE_URL = "https://www.fotmob.com"
    USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    )

    def __init__(self, headless: bool = True, rate_limit_sec: float = 1.0):
        self.headless = headless
        self.rate_limit_sec = rate_limit_sec
        self._playwright = None
        self._browser = None
        self._context = None
        self._last_request = 0.0

    def _ensure_browser(self):
        if self._browser is not None:
            return
        from playwright.sync_api import sync_playwright
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=self.headless)
        self._context = self._browser.new_context(
            user_agent=self.USER_AGENT,
            viewport={"width": 1280, "height": 800},
        )

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request
        if elapsed < self.rate_limit_sec:
            time.sleep(self.rate_limit_sec - elapsed)
        self._last_request = time.time()

    def fetch_lineups(self, fotmob_match_id: int) -> dict:
        """Fetch confirmed lineups for a FotMob match.

        Returns: {"home": [{"name": ..., "position": ..., "shirt": ...}, ...],
                  "away": [...], "kickoff": ISO, "fetched_at": ISO, "status": str}

        If lineups aren't published yet (~60 min before kickoff), returns
        an empty dict with status="not_published".
        """
        self._ensure_browser()
        self._rate_limit()

        page = self._context.new_page()
        try:
            # Use the matchDetails API endpoint (returns JSON with lineups)
            api_url = f"{self.API_URL}?matchId={fotmob_match_id}"
            response = page.goto(api_url, wait_until="domcontentloaded", timeout=20000)
            if response is None or not response.ok:
                return {
                    "home": [], "away": [],
                    "status": f"http_{response.status if response else 'none'}",
                    "fetched_at": datetime.now().isoformat(),
                }

            # The API returns JS that sets window.__INITIAL_STATE__ — read it
            try:
                state = page.evaluate("() => window.__INITIAL_STATE__")
            except Exception:
                state = None

            if not state:
                # Try JSON body directly
                try:
                    body = page.evaluate("() => document.body.innerText")
                    if body and body.strip().startswith("{"):
                        state = json.loads(body)
                except Exception:
                    pass

            return self._parse_api_state(state, fotmob_match_id)
        except Exception as e:
            return {
                "home": [], "away": [],
                "status": f"error:{type(e).__name__}",
                "error": str(e)[:200],
                "fetched_at": datetime.now().isoformat(),
            }
        finally:
            page.close()

    def _parse_api_state(self, state, match_id: int) -> dict:
        """Extract home/away lineups from FotMob's matchDetails JSON.

        Two known structures (auto-detected, June 10 2026):
          A) /api/data/matchDetails — flat: {general, content, ...}
             -> content.lineup.{homeTeam, awayTeam} (note camelCase!)
                Each side has keys: starting, bench, formation, coach, ...
                Each item in starting is a player dict directly.
          B) /__INITIAL_STATE__/matchDetails — nested:
             state.matchDetails.matchId[matchId].content.lineup.home.starting[].players
        """
        if not state or not isinstance(state, dict):
            return {"home": [], "away": [], "status": "no_state",
                    "fetched_at": datetime.now().isoformat()}

        try:
            # ── STRUCTURE A: flat /api/data/matchDetails ─────────────
            if "content" in state and "general" in state:
                content = state.get("content", {})
                lineup = content.get("lineup", {})
                if not lineup:
                    return {"home": [], "away": [], "status": "not_published",
                            "fetched_at": datetime.now().isoformat()}

                result = {"home": [], "away": [], "kickoff": None, "status": "ok",
                          "fetched_at": datetime.now().isoformat()}

                for src_key, dst_key in (("homeTeam", "home"), ("awayTeam", "away")):
                    side_data = lineup.get(src_key, {})
                    starting = side_data.get("starting", [])
                    if not isinstance(starting, list):
                        starting = []
                    result[dst_key] = [
                        {"name": p.get("name", ""),
                         "position": p.get("positionString", p.get("position", "")),
                         "shirt": p.get("shirtNumber", p.get("jerseyNumber", ""))}
                        for p in starting if isinstance(p, dict) and p.get("name")
                    ]

                # Kickoff from top-level general
                general = state.get("general", {})
                if isinstance(general, dict):
                    kickoff = general.get("matchTimeUTC") or general.get("matchTimeUTCDate")
                    if kickoff:
                        result["kickoff"] = kickoff

                return result

            # ── STRUCTURE B: nested __INITIAL_STATE__ ────────────────
            match_details = state.get("matchDetails", {})
            by_id = match_details.get("matchId", {})
            match_data = by_id.get(str(match_id)) or by_id.get(int(match_id))
            if not match_data:
                return {"home": [], "away": [], "status": "no_match",
                        "fetched_at": datetime.now().isoformat()}

            content = match_data.get("content", {})
            lineup = content.get("lineup", {})
            if not lineup:
                return {"home": [], "away": [], "status": "not_published",
                        "fetched_at": datetime.now().isoformat()}

            result = {"home": [], "away": [], "kickoff": None, "status": "ok",
                      "fetched_at": datetime.now().isoformat()}

            for side in ("home", "away"):
                side_data = lineup.get(side, {})
                starting = side_data.get("starting", [])
                players = []
                for entry in starting:
                    if isinstance(entry, dict) and "players" in entry:
                        players.extend(entry["players"])
                    elif isinstance(entry, dict) and "name" in entry:
                        players.append(entry)
                result[side] = [
                    {"name": p.get("name", ""), "position": p.get("positionString", ""),
                     "shirt": p.get("shirtNumber", "")}
                    for p in players if p.get("name")
                ]

            match_header = content.get("matchFacts", {}) or content.get("general", {})
            kickoff = match_header.get("kickoffTime") or match_data.get("kickoffTime")
            if kickoff:
                result["kickoff"] = kickoff

            return result
        except Exception as e:
            return {"home": [], "away": [], "status": f"parse_error:{type(e).__name__}",
                    "error": str(e)[:200],
                    "fetched_at": datetime.now().isoformat()}

    def close(self) -> None:
        if self._context:
            self._context.close()
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()
        self._context = None
        self._browser = None
        self._playwright = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.data.fotmob <fotmob_match_id> [kalshi_ticker]")
        print("Example: python -m src.data.fotmob 4213275 KXWCGAME-26JUN11MEXCUB-MEX")
        sys.exit(1)
    match_id = int(sys.argv[1])
    ticker = sys.argv[2] if len(sys.argv) > 2 else f"fotmob_{match_id}"

    cache = LineupCache()
    print(f"Cache file: {cache.path}")
    print(f"Existing entries: {len(cache.all())}")
    print()

    with FotMobScraper() as scraper:
        print(f"Fetching lineups for FotMob match {match_id}...")
        lineups = scraper.fetch_lineups(match_id)
        print(f"Status: {lineups.get('status')}")
        print(f"Home XI: {len(lineups.get('home', []))} players")
        print(f"Away XI: {len(lineups.get('away', []))} players")
        if lineups.get("home"):
            print("Home:", [p["name"] for p in lineups["home"][:5]], "...")
        if lineups.get("away"):
            print("Away:", [p["name"] for p in lineups["away"][:5]], "...")

        if lineups.get("status") == "ok" and (lineups.get("home") or lineups.get("away")):
            cache.set(ticker, lineups)
            print(f"\nCached → {ticker}")
        else:
            print("\nNo lineups to cache (not published yet or fetch failed)")
