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
import re
import time
import unicodedata
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


# === Fuzzy name-matching fallback (closes the McBride + suffix gap) ===
# ESPN's displayName and Kalshi's market title can differ in ways that broke
# the previous exact-match filter:
#   - Suffixes: ESPN "Michael Porter Jr." vs Kalshi "Michael Porter"
#   - Roman numerals: ESPN "Marvin Bagley III" vs Kalshi "Marvin Bagley"
#   - Nicknames: Kalshi "Miles 'Deuce' McBride" vs ESPN "Miles McBride"
#   - Accents: ESPN "Luka Dončić" vs Kalshi "Luka Doncic" (diacritics stripped)
#   - Punctuation: "Butler III." vs "Butler III" (trailing period)
#
# `_normalize_name()` strips all of these so "michael porter jr" ==
# "michael porter" == "michael porter jr." in the lookup.
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _strip_accents(s: str) -> str:
    """Strip diacritics via Unicode NFKD decomposition (Dončić -> Doncic)."""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _normalize_name(name: str) -> str:
    """Normalize a name for fuzzy comparison.

    - lowercase
    - strip diacritics (Dončić -> Doncic)
    - strip punctuation (. , ' ")
    - strip generational suffixes (Jr, Sr, II, III, IV, V)
    - collapse whitespace

    Examples
    --------
    >>> _normalize_name("Michael Porter Jr.")
    'michael porter'
    >>> _normalize_name("Marvin Bagley III")
    'marvin bagley'
    >>> _normalize_name("Luka Dončić")
    'luka doncic'
    >>> _normalize_name("Miles 'Deuce' McBride")
    'miles deuce mcbride'
    """
    if not name:
        return ""
    s = name.strip().lower()
    s = _strip_accents(s)
    # Strip all punctuation (anything not letter/whitespace/period/apostrophe)
    s = re.sub(r"[^\w\s]", " ", s)
    # Remove generational suffixes
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", " ", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _names_match(a: str, b: str, threshold: float = 0.85) -> bool:
    """Check if two names refer to the same player (fuzzy match).

    Uses multi-stage heuristics, best match wins:
      1. Exact normalized match → True
      2. Last name match + first-initial match → True
      3. Last name match + shared first part (handles nicknames) → True
      4. rapidfuzz ratio (if installed) >= threshold → True
      5. Character-overlap ratio (fallback) >= threshold → True
    """
    a_n = _normalize_name(a)
    b_n = _normalize_name(b)
    if not a_n or not b_n:
        return False
    if a_n == b_n:
        return True

    a_parts = a_n.split()
    b_parts = b_n.split()
    a_last = a_parts[-1]
    b_last = b_parts[-1]
    if a_last != b_last:
        # Different last names — definitely not the same player
        return False

    a_first = a_parts[0]
    b_first = b_parts[0]

    # 2. First-initial match (handles "Luka Doncic" vs "Luka Dončić",
    # and "M. Porter Jr." vs "Michael Porter")
    if a_first[0] == b_first[0]:
        return True

    # 3. Shared non-suffix token (handles "Miles Deuce McBride" vs
    # "Miles McBride" where 'miles' is shared)
    a_set = set(a_parts)
    b_set = set(b_parts)
    shared = a_set & b_set
    if len(shared) >= 1 and a_first in shared:
        return True

    # 4. rapidfuzz if available
    try:
        from rapidfuzz import fuzz
        if fuzz.ratio(a_n, b_n) / 100.0 >= threshold:
            return True
    except ImportError:
        pass

    # 5. Character-overlap fallback (Jaccard on character bigrams)
    def bigrams(s):
        return {s[i:i+2] for i in range(len(s) - 1)} if len(s) > 1 else {s}
    bg_a, bg_b = bigrams(a_n), bigrams(b_n)
    if bg_a and bg_b:
        overlap = len(bg_a & bg_b) / max(len(bg_a | bg_b), 1)
        if overlap >= 0.7:
            return True

    return False


def is_player_out(player_name: str, out_set: set | None = None) -> bool:
    """Check if a player is OUT, with optional pre-fetched set for speed.

    Uses **fuzzy name matching** so that ESPN-vs-Kalshi name mismatches
    (suffixes, roman numerals, nicknames, accents) still get filtered.

    Parameters
    ----------
    player_name : str
        Full player name as it appears in Kalshi titles ("Miles McBride").
    out_set : set | None
        If provided, use this pre-fetched set instead of calling get_out_players().
    """
    if out_set is None:
        out_set = get_out_players()
    if not player_name or not out_set:
        return False
    # Fast path: exact (case-insensitive) match
    if player_name.strip().lower() in out_set:
        return True
    # Slow path: fuzzy match against every OUT name (≤ a few hundred)
    for out_name in out_set:
        if _names_match(player_name, out_name):
            return True
    return False
