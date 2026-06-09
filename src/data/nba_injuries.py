"""
NBA Injury data fetcher using ESPN's public API.

Caches daily to ``data/nba_cache/injuries.json`` with a 3-hour TTL
during active months (Oct–Jun).  Provides a simple set-based lookup
so the NBA scanner can skip injured players.

Usage::

    from src.data.nba_injuries import get_out_players

    out = get_out_players()
    # out is a set of lowercase full names, e.g. {"miles mcbride", ...}
"""

import json
import time
from datetime import datetime, date
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "nba_cache"
CACHE_PATH = CACHE_DIR / "injuries.json"

ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"

# Consider a fetched report fresh for this many seconds
MAX_AGE_SECONDS = 3 * 3600  # 3 hours

# Statuses that mean the player definitely won't play
OUT_STATUSES = {"Out", "OUT", "Out for Season"}


def _is_cache_fresh() -> bool:
    if not CACHE_PATH.exists():
        return False
    age = time.time() - CACHE_PATH.stat().st_mtime
    return age < MAX_AGE_SECONDS


def _fetch_and_cache() -> dict:
    """Fetch ESPN injury report, cache it, return {status: date, injuries: [...]}."""
    try:
        r = requests.get(ESPN_URL, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        # If ESPN is down, return empty — fail open (don't block scanning)
        print(f"  Warning: ESPN injury API unavailable ({e}) — proceeding without injury filter")
        return {"status": "error", "injuries": []}

    payload = {
        "fetched_at": datetime.utcnow().isoformat(),
        "status": "ok",
        "injuries": data.get("injuries", []),
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(payload, f)
    return payload


def _load_cache() -> dict:
    """Return cached injury payload, fetching if stale or missing."""
    if _is_cache_fresh():
        with open(CACHE_PATH) as f:
            return json.load(f)
    return _fetch_and_cache()


def get_out_players(force_refresh: bool = False) -> set:
    """Return set of **lowercase full names** of players currently OUT.

    Example::

        out = get_out_players()
        # {"miles mcbride", "jayson tatum", "keshon gilbert", ...}

    Parameters
    ----------
    force_refresh : bool
        If True, bypass the cache and fetch fresh data from ESPN.
    """
    if force_refresh:
        payload = _fetch_and_cache()
    else:
        payload = _load_cache()

    out = set()
    for team in payload.get("injuries", []):
        for inj in team.get("injuries", []):
            status = (inj.get("status") or "").strip()
            if status not in OUT_STATUSES:
                continue
            athlete = inj.get("athlete", {})
            name = (athlete.get("displayName") or "").strip()
            if name:
                out.add(name.lower())
    return out


def is_player_out(player_name: str, out_set: set | None = None) -> bool:
    """Check if a player is OUT, with optional pre-fetched set for speed.

    Parameters
    ----------
    player_name : str
        Full player name as it appears in Kalshi titles ("Miles McBride").
    out_set : set | None
        If provided, use this pre-fetched set instead of calling get_out_players().
    """
    if out_set is None:
        out_set = get_out_players()
    return player_name.strip().lower() in out_set
