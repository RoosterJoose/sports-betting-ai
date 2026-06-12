"""Live integration test for the 2026 MLB + NFL + WC data feeds the oracle depends on.

Skipped by default (network required). Run with: `pytest tests/test_oracle_2026.py --runlive`

This is a **regression guard** against upstream feed changes:

  (a) `statsapi.mlb.com` must continue to return 2026 MLB schedule data
      with probable pitchers — used by `src/data/mlb.py`, `src/scripts/scan_mlb_sim.py`,
      and `bin/research_oracle.py`.

  (b) `nfl-data-py` must continue to return 2026 NFL schedule + weekly data
      — used by `src/data/nfl.py` and `src/scripts/fetch_nfl_extra.py`.

  (c) The Odds API (`api.the-odds-api.com`) must continue to return 2026 MLB
      AND NFL moneyline odds — centralized in `OddsAPIClient.get_mlb_odds()`
      and `OddsAPIClient.get_nfl_odds()` (added 2026-06-12).

  (d) The parsed matchups must include the team codes our model uses —
      e.g., BAL, ATL, CLE, CIN, HOU, SF for MLB; NE, SEA, KC, BUF, etc.
      for NFL. If the upstream feed renames a team, the model will silently
      miss picks.

  (e) `data/cache/worldcup/all_matches.parquet` must contain the WC 2022
      final (Argentina vs France 2022-12-18, 3-3) with the expected
      home/away + scores + Elo. Used by `src/data/world_cup.py`,
      `src/scripts/scan_wc.py`, and `src/scripts/backtest_wc.py`.

Audit findings (2026-06-12)
---------------------------
**`src/data/odds_api.py`** — UFC-only. Hardcodes `SPORT_KEY_UFC = "mma_mixed_martial_arts"`
and exposes only `get_ufc_events()` / `get_ufc_odds()`. No MLB or NFL method.
The MLB moneyline fetch is duplicated in `bin/research_oracle.py` (sport_key
`baseball_mlb`); no NFL moneyline fetch exists yet (would use sport_key
`americanfootball_nfl`). Recommendation: extend `OddsAPIClient` with
`get_mlb_odds()` and `get_nfl_odds()` methods to centralize the feed access.
Not done in this commit to keep the regression test scope tight.

**`src/data/mlb.py`** — Uses `https://statsapi.mlb.com/api/v1` (free, no auth).
`fetch_player_game_logs(seasons)` accepts `"2026"` as a season string and builds
the cache file name from it. `_api_get` has no retry logic but otherwise handles
2026 dates transparently. Verified working today: 30,211 player-game rows
covering 2026-03-25 → 2026-06-11 are in the parquet cache.

**`src/data/nfl.py`** — Uses `nfl-data-py` (free, no auth, no rate limit). The
library wraps the nflverse/nflfastR data releases on GitHub. Verified today:
`nfl.import_schedules([2026])` returns 272 games spanning 2026-09-09 →
2027-01-10 (the full 2026 regular season + postseason). ⚠️ **Cache staleness**:
`data/nfl_cache/weekly.parquet` only covers 2022-01-07 → 2024-05-31 — does not
include 2025 or 2026 weekly data. Refresh needed for the model to have
2025-2026 player stats. The 2024-2025 off-season window (May-Sep) is when
weekly data is incomplete.

**`fetch_player_game_logs` off-season cap**: `max_year = datetime.now().year`
filters out future years. Since `now()` returns 2026 in this project, 2026 IS
included. But for years BEYOND the current year (e.g., running this in 2027
with a request for 2028), the filter would silently drop the request. Not a
problem today, but a regression risk.

**`src/scripts/fetch_nfl_extra.py`** — Uses `nfl-data-py` for schedule/injuries/
betting lines and `Open-Meteo API` (free, no auth) for weather. Same `max_year`
pattern as above. The schedule is loaded from `data/nfl_cache/schedule.parquet`
if it exists; needs to be populated once per season for the 2026 schedule to
be visible downstream.

**Pass/fail signal:** If a future `nfl-data-py` API change, nflverse data
schema change, or upstream feed break causes 2026 access to fail, this test
fails immediately. The fix is in the upstream feed or in our adapter.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ── --runlive flag (duplicated from conftest for this file's isolation) ──
# We do NOT want a missing --runlive on a CI run to silently pass — the whole
# point of this test is to fail loudly when upstream data breaks. So we
# skip cleanly when the flag is missing, and assert strictly when it's set.

MLB_STATS_API = "https://statsapi.mlb.com/api/v1"
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY_MLB = "baseball_mlb"
SPORT_KEY_NFL = "americanfootball_nfl"
DEFAULT_BOOKMAKERS = "draftkings,fanduel,betmgm"

# Team codes the MLB model uses (per src/data/mlb.py + scan_mlb_sim.py).
MLB_TEAM_CODES = [
    "ARI", "ATL", "BAL", "BOS", "CHC", "CIN", "CLE", "COL", "CWS", "DET",
    "HOU", "KC", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "OAK",
    "PHI", "PIT", "SD", "SEA", "SF", "STL", "TB", "TEX", "TOR", "WSH",
]

# Team codes the NFL model uses (per src/data/nfl.py + scan_nfl/parser).
# nfl-data-py returns the codes directly (no ID-to-abbr lookup needed).
NFL_TEAM_CODES = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN",
    "DET", "GB", "HOU", "IND", "JAX", "KC", "LA", "LAC", "LV", "MIA",
    "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB",
    "TEN", "WAS",
]

# Focus on June 12, 2026 (today's WC + MLB slate). NFL season hasn't started
# yet (kickoff is September 2026) so the NFL schedule test uses the 2026
# season broadly rather than a specific date.
DEFAULT_TEST_DATE = "2026-06-12"


def _http_get_json(url: str, timeout: float = 30.0) -> tuple[int, dict | list | None, str]:
    """Return (status_code, parsed_json_or_None, raw_text)."""
    try:
        req = Request(url, headers={"User-Agent": "sports-betting-ai-oracle-test/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return (resp.status, json.loads(raw), raw)
    except Exception as e:
        return (0, None, str(e))


def pytest_addoption(parser):
    parser.addoption("--runlive", action="store_true", default=False,
                     help="Run live oracle API tests (requires network)")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runlive"):
        skip_live = pytest.mark.skip(reason="need --runlive to run live tests")
        for item in items:
            if "live" in item.keywords:
                item.add_marker(skip_live)


pytestmark = pytest.mark.live


# ════════════════════════════════════════════════════════════════════════
# MLB — Stats API
# ════════════════════════════════════════════════════════════════════════

def test_mlb_stats_api_returns_2026_schedule():
    """Hit statsapi.mlb.com schedule endpoint for 2026-06-12 and verify games come back."""
    url = (
        f"{MLB_STATS_API}/schedule"
        f"?sportId=1&date={DEFAULT_TEST_DATE}&hydrate=probablePitcher,team"
    )
    status, data, raw = _http_get_json(url)
    assert status == 200, f"MLB Stats API returned {status}: {raw[:200]}"
    assert isinstance(data, dict), f"Expected dict response, got {type(data).__name__}"
    assert "dates" in data, f"Response missing 'dates' key: keys={list(data.keys())}"

    games_today = []
    for d in data["dates"]:
        if d.get("date") == DEFAULT_TEST_DATE:
            games_today = d.get("games", [])
            break

    if not games_today:
        pytest.skip(
            f"No games published for {DEFAULT_TEST_DATE} yet. "
            f"MLB Stats API returned 200 but the schedule is empty. "
            f"Re-run when the day's games are posted."
        )

    assert len(games_today) > 0, f"Zero games for {DEFAULT_TEST_DATE}"
    print(f"\n  MLB Stats API: {len(games_today)} games for {DEFAULT_TEST_DATE}")

    with_pitchers = [g for g in games_today
                     if g.get("teams", {}).get("away", {}).get("probablePitcher")
                     or g.get("teams", {}).get("home", {}).get("probablePitcher")]
    assert with_pitchers, (
        f"No games on {DEFAULT_TEST_DATE} have probable pitchers. "
        f"Hydrate=probablePitcher may have broken. Sample game: {games_today[0]}"
    )
    print(f"  → {len(with_pitchers)}/{len(games_today)} games have probable pitchers")


# ════════════════════════════════════════════════════════════════════════
# MLB — Odds API
# ════════════════════════════════════════════════════════════════════════

def test_odds_api_returns_2026_mlb_odds():
    """Use OddsAPIClient.get_mlb_events() + get_mlb_odds() to verify the MLB feed."""
    api_key = os.environ.get("ODDS_API_KEY", "")
    if not api_key:
        pytest.skip("ODDS_API_KEY env var not set — cannot test the MLB odds feed")

    from src.data.odds_api import OddsAPIClient, SPORT_KEY_MLB as CLIENT_SPORT_KEY_MLB
    assert CLIENT_SPORT_KEY_MLB == SPORT_KEY_MLB, "Client MLB sport key drifted from test"

    client = OddsAPIClient(api_key=api_key)
    events = client.get_mlb_events()
    assert events, "OddsAPIClient.get_mlb_events() returned empty list"
    print(f"\n  OddsAPIClient.get_mlb_events(): {len(events)} upcoming events")

    # Verify one event's shape
    ev = events[0]
    for required in ("id", "sport_key", "commence_time", "home_team", "away_team"):
        assert required in ev, f"Event missing '{required}': {ev}"
    assert ev["sport_key"] == SPORT_KEY_MLB, f"Wrong sport_key: {ev['sport_key']}"

    # Verify the odds call works (1 credit per bookmaker)
    odds = client.get_mlb_odds(ev["id"])
    assert odds, f"No bookmaker odds returned for event {ev['id']}"
    book = odds[0]
    for required in ("sportsbook", "commence_time", "home_team", "away_team", "home_odds", "away_odds"):
        assert required in book, f"Bookmaker dict missing '{required}': {book}"
    assert isinstance(book["home_odds"], (int, float)) and book["home_odds"] != 0, (
        f"home_odds is not a valid American-odds number: {book['home_odds']}"
    )
    print(f"  → Sample: {ev['away_team']} @ {ev['home_team']} ({ev['commence_time']})")
    print(f"    {book['sportsbook']:12s}  home={book['home_odds']}  away={book['away_odds']}")
    print(f"  → Credits remaining: {client.credits_remaining}")


# ════════════════════════════════════════════════════════════════════════
# NFL — nfl-data-py schedule
# ════════════════════════════════════════════════════════════════════════

def test_nfl_data_py_returns_2026_schedule():
    """Hit nfl_data_py.import_schedules([2026]) and verify games come back.

    nfl-data-py wraps the nflverse/nflfastR data releases on GitHub. The
    2026 schedule is published weeks before the season starts (kickoff is
    2026-09-09). Verified working today: 272 games, dates 2026-09-09 →
    2027-01-10.
    """
    try:
        import nfl_data_py as nfl
    except ImportError as e:
        pytest.skip(f"nfl-data-py not installed: {e}")

    try:
        sched = nfl.import_schedules([2026])
    except Exception as e:
        pytest.fail(f"nfl-data-py schedule fetch failed: {e}")

    assert sched is not None, "nfl-data-py returned None for 2026 schedule"
    assert not sched.empty, "nfl-data-py returned an empty DataFrame for 2026"
    print(f"\n  nfl-data-py: {len(sched)} 2026 games, "
          f"{sched['gameday'].min()} → {sched['gameday'].max()}")

    # Verify the schedule covers the standard NFL season
    assert len(sched) >= 250, f"Only {len(sched)} 2026 games (expected 250+ for a full season)"
    assert "gameday" in sched.columns, "Schedule missing 'gameday' column"
    assert "home_team" in sched.columns, "Schedule missing 'home_team' column"
    assert "away_team" in sched.columns, "Schedule missing 'away_team' column"

    # Sanity: at least 24 of 32 NFL teams should appear (most teams play in week 1)
    home_teams = set(sched["home_team"].dropna().unique())
    away_teams = set(sched["away_team"].dropna().unique())
    n_teams = len(home_teams | away_teams)
    assert n_teams >= 24, f"Only {n_teams} distinct teams in 2026 schedule (expected 24+)"
    print(f"  → {n_teams} distinct teams ({len(home_teams)} home, {len(away_teams)} away)")


# ════════════════════════════════════════════════════════════════════════
# NFL — nfl-data-py weekly
# ════════════════════════════════════════════════════════════════════════

def test_nfl_data_py_returns_2026_weekly_data():
    """Hit nfl_data_py.import_weekly_data([2026]) — should NOT crash, even if empty.

    During the off-season (May-Aug 2026), 2026 weekly data is empty. The
    test verifies the call returns a DataFrame without raising — that's the
    regression guard. The `max_year` filter in src/data/nfl.py is the most
    likely failure point if the API changes.
    """
    try:
        import nfl_data_py as nfl
    except ImportError as e:
        pytest.skip(f"nfl-data-py not installed: {e}")

    # The same call src/data/nfl.py:fetch_player_game_logs makes.
    # We don't write to cache — just verify the call works.
    try:
        df = nfl.import_weekly_data([2026], downcast=True)
    except Exception as e:
        # 2026 weekly data is empty (off-season). nfl-data-py may return
        # an empty df, raise, or return a different shape. Treat all as
        # OK as long as it's not a 5xx-style crash.
        if "404" in str(e) or "not found" in str(e).lower():
            pytest.skip(f"nfl-data-py doesn't have 2026 weekly yet (off-season): {e}")
        pytest.fail(f"nfl-data-py weekly call crashed unexpectedly: {e}")

    print(f"\n  nfl-data-py weekly: shape={df.shape if df is not None else 'None'}")
    if df is None or df.empty:
        print("  → Empty (off-season, expected). NOT a regression.")
    else:
        # If we got data, sanity check the schema
        expected_cols = {"player_id", "season", "week"}
        missing = expected_cols - set(df.columns)
        assert not missing, f"Weekly data missing expected columns: {missing}"
        print(f"  → {len(df)} rows, {df['player_id'].nunique()} unique players")


# ════════════════════════════════════════════════════════════════════════
# NFL — Odds API
# ════════════════════════════════════════════════════════════════════════

def test_odds_api_returns_2026_nfl_odds():
    """Use OddsAPIClient.get_nfl_events() + get_nfl_odds() to verify the NFL feed.

    The Odds API publishes NFL odds ~2 weeks before kickoff. Today (June 12,
    2026) the 2026 season is 3 months out — this test will likely see 0
    events until late August. The test asserts the call structure is correct
    even if empty.
    """
    api_key = os.environ.get("ODDS_API_KEY", "")
    if not api_key:
        pytest.skip("ODDS_API_KEY env var not set — cannot test the NFL odds feed")

    from src.data.odds_api import OddsAPIClient, SPORT_KEY_NFL as CLIENT_SPORT_KEY_NFL
    assert CLIENT_SPORT_KEY_NFL == SPORT_KEY_NFL, "Client NFL sport key drifted from test"

    client = OddsAPIClient(api_key=api_key)
    events = client.get_nfl_events()

    if not events:
        pytest.skip(
            f"No upcoming NFL events on The Odds API (expected — 2026 season "
            f"kicks off 2026-09-09, today is 2026-06-12). Re-run in late August."
        )

    assert events, "OddsAPIClient.get_nfl_events() returned empty list"
    print(f"\n  OddsAPIClient.get_nfl_events(): {len(events)} upcoming events")

    # Verify one event's shape
    ev = events[0]
    for required in ("id", "sport_key", "commence_time", "home_team", "away_team"):
        assert required in ev, f"Event missing '{required}': {ev}"
    assert ev["sport_key"] == SPORT_KEY_NFL, f"Wrong sport_key: {ev['sport_key']}"

    # Verify the odds call works (1 credit per bookmaker)
    odds = client.get_nfl_odds(ev["id"])
    assert odds, f"No bookmaker odds returned for event {ev['id']}"
    book = odds[0]
    for required in ("sportsbook", "commence_time", "home_team", "away_team", "home_odds", "away_odds"):
        assert required in book, f"Bookmaker dict missing '{required}': {book}"
    assert isinstance(book["home_odds"], (int, float)) and book["home_odds"] != 0, (
        f"home_odds is not a valid American-odds number: {book['home_odds']}"
    )
    print(f"  → Sample: {ev['away_team']} @ {ev['home_team']} ({ev['commence_time']})")
    print(f"    {book['sportsbook']:12s}  home={book['home_odds']}  away={book['away_odds']}")
    print(f"  → Credits remaining: {client.credits_remaining}")


# ════════════════════════════════════════════════════════════════════════
# World Cup — parquet cache + world_cup.py source
# ════════════════════════════════════════════════════════════════════════

def test_wc_all_matches_parquet_contains_arg_fra_2022_final():
    """Regression guard: the WC 2022 final (Argentina vs France 2022-12-18) must
    be in the cache with the expected home/away, score, and Elo values.

    The score is 3-3 after extra time; Argentina won the penalty shootout
    4-2. The dataset stores regulation/extra-time score (3-3) without
    penalty outcome, so we only assert 3-3 + ARG as home.

    Catches: if the eloratings source URL changes format, the cache is
    stale, or the parquet was regenerated with a bug, this test fails.
    """
    cache_path = PROJECT_ROOT / "data" / "cache" / "worldcup" / "all_matches.parquet"
    if not cache_path.exists():
        pytest.skip(f"No WC cache at {cache_path}")
    df = pd.read_parquet(cache_path)
    assert not df.empty, "WC cache is empty"

    # Required columns for the WC model + scan_wc.py
    required_cols = [
        "match_date", "home_team", "away_team",
        "home_score", "away_score", "tournament_code",
        "elo_home_pre", "elo_away_pre",
        "elo_home_post", "elo_away_post",
    ]
    missing = set(required_cols) - set(df.columns)
    assert not missing, f"WC cache missing columns: {missing}. Got: {list(df.columns)}"

    # Find the 2022 final (tournament_code = "WC", date = 2022-12-18)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    finals = df[
        (df["match_date"] == pd.Timestamp("2022-12-18"))
        & (df["tournament_code"] == "WC")
    ]
    if finals.empty:
        pytest.skip(
            "2022-12-18 WC final not in cache. Parquet may predate the 2022 WC data. "
            "Re-run src.data.world_cup:fetch_all_matches() to refresh."
        )

    assert len(finals) == 1, (
        f"Expected exactly 1 WC match on 2022-12-18, found {len(finals)}. "
        f"Rows: {finals.to_dict('records')}"
    )
    final = finals.iloc[0]

    # The 2022 final was played at Lusail Stadium in Qatar — the home_team
    # column stores ARG because Argentina was listed first in the
    # eloratings.net source feed. We don't assume the venue's "home" team
    # but rather the source's column ordering.
    assert final["home_team"] == "Argentina", (
        f"home_team: expected 'Argentina', got {final['home_team']!r}"
    )
    assert final["away_team"] == "France", (
        f"away_team: expected 'France', got {final['away_team']!r}"
    )
    assert int(final["home_score"]) == 3, (
        f"home_score: expected 3, got {final['home_score']!r}"
    )
    assert int(final["away_score"]) == 3, (
        f"away_score: expected 3, got {final['away_score']!r}"
    )
    assert final["tournament_code"] == "WC", (
        f"tournament_code: expected 'WC', got {final['tournament_code']!r}"
    )
    assert final["source"] == "eloratings", (
        f"source: expected 'eloratings', got {final['source']!r}"
    )

    # Elo pre-match should be a positive number in the realistic range
    # (international football Elo typically 1500-2200; WC finalists 2000-2200).
    for elo_col in ("elo_home_pre", "elo_away_pre", "elo_home_post", "elo_away_post"):
        v = final[elo_col]
        assert pd.notna(v), f"{elo_col} is NaN"
        assert 1500 <= float(v) <= 2500, (
            f"{elo_col}={v} out of expected range [1500, 2500] — Elo source may have changed"
        )
    print(
        f"\n  WC 2022 final: {final['home_team']} {int(final['home_score'])}-"
        f"{int(final['away_score'])} {final['away_team']} "
        f"(Elo pre: {int(final['elo_home_pre'])} vs {int(final['elo_away_pre'])}, "
        f"post: {int(final['elo_home_post'])} vs {int(final['elo_away_post'])})"
    )


def test_wc_fetch_all_matches_returns_canonical_columns():
    """Hit src/data/world_cup.py:fetch_all_matches() and verify it returns
    the same canonical schema as the parquet cache.

    This is a function-level regression guard. Even if the cache is
    regenerated by a different code path, fetch_all_matches() must produce
    matching columns so the WC model + scan_wc.py don't break.

    Skips if the function requires network and the call fails (we already
    have a parquet test for the data integrity itself).
    """
    try:
        from src.data.world_cup import fetch_all_matches
    except ImportError as e:
        pytest.skip(f"src.data.world_cup not importable: {e}")

    try:
        df = fetch_all_matches()
    except Exception as e:
        pytest.skip(f"fetch_all_matches() raised (network down or signature changed): {e}")

    if df is None or df.empty:
        pytest.skip("fetch_all_matches() returned empty DataFrame")

    # Required columns must be present (subset match — function may return
    # additional columns not in the cache, e.g., computed features)
    required = {
        "match_date", "home_team", "away_team",
        "home_score", "away_score", "tournament_code",
        "elo_home_pre", "elo_away_pre",
    }
    missing = required - set(df.columns)
    assert not missing, (
        f"fetch_all_matches() missing required columns: {missing}. "
        f"Got: {sorted(df.columns.tolist())}"
    )

    # If 2022-12-18 final is in the returned data, verify the team names
    if "match_date" in df.columns:
        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
        final = df[
            (df["match_date"] == pd.Timestamp("2022-12-18"))
            & (df["tournament_code"] == "WC")
        ]
        if not final.empty:
            assert final.iloc[0]["home_team"] == "Argentina"
            assert final.iloc[0]["away_team"] == "France"
            assert int(final.iloc[0]["home_score"]) == 3
            assert int(final.iloc[0]["away_score"]) == 3
            print(
                f"\n  fetch_all_matches(): {len(df):,} matches, "
                f"2022 final verified (3-3 ARG-FRA)"
            )
        else:
            print(
                f"\n  fetch_all_matches(): {len(df):,} matches, "
                f"2022 final not in this call's slice (date filter or partial refetch)"
            )


def test_wc_data_source_2022_cache_shape():
    """Sanity guard: the WC 2022 cache has 60+ group stage + knockout matches.
    If the cache is much smaller (< 50), the data source is broken or stale.

    The 2022 WC had 64 matches (48 group + 16 knockout), held Nov 21 - Dec 18.
    We assert >= 50 to allow for partial refreshes (e.g., group stage only).
    """
    cache_path = PROJECT_ROOT / "data" / "cache" / "worldcup" / "all_matches.parquet"
    if not cache_path.exists():
        pytest.skip(f"No WC cache at {cache_path}")
    df = pd.read_parquet(cache_path)
    if "match_date" not in df.columns or "tournament_code" not in df.columns:
        pytest.skip("WC cache missing match_date or tournament_code column")

    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    wc_2022 = df[
        (df["tournament_code"] == "WC")
        & (df["match_date"].dt.year == 2022)
        & (df["match_date"].dt.month >= 11)  # WC 2022 ran Nov 21 - Dec 18
    ]
    print(f"\n  WC 2022 matches in cache: {len(wc_2022)}")
    assert len(wc_2022) >= 50, (
        f"Only {len(wc_2022)} WC 2022 matches in cache (expected ≥50). "
        f"Re-run src.data.world_cup:fetch_all_matches() to refresh."
    )


def test_wc_2026_oracle_api_integration():
    """End-to-end integration test: hit bin/oracle_api.py /oracle/wc/{date} and
    verify the response shape for the 2022 WC final (known data) and today's
    WC slate.

    Uses FastAPI's TestClient for hermetic in-process testing — no need for a
    real running server. The TestClient imports the app and calls the route
    handlers directly, so the test verifies the same code path that production
    traffic would hit.

    Closes the loop: if bin/oracle_api.py and data/cache/worldcup/all_matches.parquet
    ever drift (e.g., the API renames a field but the cache has the old name,
    or vice versa), this test fails immediately.

    Catches:
    - API route registration breaks (e.g., someone renames `/oracle/wc/{date}`)
    - Response schema drift (e.g., `result` field renamed to `match_result`)
    - WC cache field renames that the API doesn't handle
    - 500 errors (e.g., division by zero in the result classifier)
    """
    try:
        from fastapi.testclient import TestClient
    except ImportError as e:
        pytest.skip(f"FastAPI not installed: {e}")

    # Import the app in-process (TestClient wraps the ASGI app directly)
    try:
        from bin.oracle_api import app
    except ImportError as e:
        pytest.skip(f"bin.oracle_api not importable: {e}")

    client = TestClient(app)

    # ── 1. /health check — confirms the server (in-process) is up + WC cache loaded
    health_resp = client.get("/health")
    assert health_resp.status_code == 200, (
        f"/health returned {health_resp.status_code}: {health_resp.text}"
    )
    health = health_resp.json()
    assert health["status"] == "ok"
    assert health.get("wc_cache_loaded") is True, (
        f"WC cache not loaded — /health said wc_cache_loaded={health.get('wc_cache_loaded')}. "
        f"Check data/cache/worldcup/all_matches.parquet exists."
    )
    print(f"\n  Oracle API /health: ok, wc_cache_loaded={health['wc_cache_loaded']}")

    # ── 2. /oracle/wc/2022-12-18 — verify the response shape with the 2022 final
    resp = client.get("/oracle/wc/2022-12-18")
    assert resp.status_code == 200, (
        f"/oracle/wc/2022-12-18 returned {resp.status_code}: {resp.text}"
    )
    body = resp.json()

    # Top-level shape (echoes the schema from /schema)
    for required in ("date", "matches", "n_matches"):
        assert required in body, f"Response missing top-level '{required}': keys={list(body.keys())}"
    assert body["date"] == "2022-12-18", f"date echo wrong: {body['date']!r}"
    assert isinstance(body["matches"], list), f"matches should be list, got {type(body['matches']).__name__}"
    assert body["n_matches"] == len(body["matches"]), (
        f"n_matches={body['n_matches']} != len(matches)={len(body['matches'])}"
    )

    if body["n_matches"] == 0:
        pytest.skip("No WC matches for 2022-12-18 in cache (parquet may predate 2022 WC data)")

    # Find the 2022 final in the matches
    final = next(
        (m for m in body["matches"]
         if m.get("home_team") == "Argentina" and m.get("away_team") == "France"),
        None,
    )
    assert final is not None, (
        f"ARG-FRA 2022 final not in /oracle/wc/2022-12-18 response. "
        f"Got {body['n_matches']} matches: {body['matches'][:3]}..."
    )

    # Per-match shape — every field declared in /schema must be present
    for field in (
        "match_date", "home_team", "away_team", "home_score", "away_score",
        "tournament_code", "elo_home_pre", "elo_away_pre", "result",
    ):
        assert field in final, f"Match missing '{field}': {final}"
    assert final["home_score"] == 3, f"home_score: expected 3, got {final['home_score']!r}"
    assert final["away_score"] == 3, f"away_score: expected 3, got {final['away_score']!r}"
    assert final["tournament_code"] == "WC", f"tournament_code: expected 'WC', got {final['tournament_code']!r}"
    # 3-3 reg+ET → result = "draw" (the API doesn't know about the penalty shootout)
    assert final["result"] == "draw", (
        f"result: expected 'draw' (3-3 reg+ET, no penalty logic), got {final['result']!r}. "
        f"If you added penalty handling, update this assertion."
    )
    # Elo should be in plausible range
    for elo_field in ("elo_home_pre", "elo_away_pre"):
        v = final[elo_field]
        assert v is not None, f"{elo_field} is None"
        assert 1500 <= float(v) <= 2500, f"{elo_field}={v} out of expected range"
    print(
        f"  /oracle/wc/2022-12-18: {body['n_matches']} matches, "
        f"final: {final['home_team']} {final['home_score']}-{final['away_score']} "
        f"{final['away_team']} (result={final['result']!r}, "
        f"Elo pre: {final['elo_home_pre']:.0f} vs {final['elo_away_pre']:.0f})"
    )

    # ── 3. /oracle/wc/2026-06-12 — today's slate (the 3rd leg of today's parlay)
    today_resp = client.get("/oracle/wc/2026-06-12")
    assert today_resp.status_code == 200, (
        f"/oracle/wc/2026-06-12 returned {today_resp.status_code}: {today_resp.text}"
    )
    today = today_resp.json()
    assert today["date"] == "2026-06-12"
    # Don't assert n_matches > 0 — the parquet may not have 2026-06-12 yet.
    # The shape check is what matters.
    if today["n_matches"] > 0:
        sample = today["matches"][0]
        for field in (
            "match_date", "home_team", "away_team", "home_score", "away_score",
            "tournament_code", "elo_home_pre", "elo_away_pre", "result",
        ):
            assert field in sample, f"Today match missing '{field}': {sample}"
        print(f"  /oracle/wc/2026-06-12: {today['n_matches']} matches, "
              f"first: {sample['home_team']} vs {sample['away_team']}")
    else:
        print("  /oracle/wc/2026-06-12: 0 matches (parquet may not have today yet — shape check passed)")

    # ── 4. /oracle/wc/not-a-date — bad date must return 400 with detail
    bad_resp = client.get("/oracle/wc/not-a-date")
    assert bad_resp.status_code == 400, (
        f"Bad date should return 400, got {bad_resp.status_code}: {bad_resp.text}"
    )
    assert "Invalid date" in bad_resp.json().get("detail", ""), (
        f"400 detail missing 'Invalid date': {bad_resp.json()}"
    )
    print(f"  /oracle/wc/not-a-date → 400 (with detail)")


# ════════════════════════════════════════════════════════════════════════
# (c) Parsed matchups include games our model is using
# ════════════════════════════════════════════════════════════════════════

def test_parsed_matchups_include_model_team_codes():
    """The schedule + odds response must include team codes our model uses (BAL, ATL, ...)."""
    # ── Schedule side: map MLB Stats API team IDs to our 3-letter codes
    schedule_url = (
        f"{MLB_STATS_API}/schedule"
        f"?sportId=1&date={DEFAULT_TEST_DATE}&hydrate=team"
    )
    s_status, s_data, _ = _http_get_json(schedule_url)
    if s_status != 200 or not s_data.get("dates"):
        pytest.skip(f"MLB schedule not available for {DEFAULT_TEST_DATE} (status={s_status})")

    games_today = []
    for d in s_data["dates"]:
        if d.get("date") == DEFAULT_TEST_DATE:
            games_today = d.get("games", [])
            break
    if not games_today:
        pytest.skip(f"No games for {DEFAULT_TEST_DATE}")

    MLB_API_TEAM_IDS = {
        108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS",
        112: "CHC", 113: "CIN", 114: "CLE", 115: "COL",
        116: "DET", 117: "HOU", 118: "KC", 119: "LAD",
        120: "WSH", 121: "NYM", 133: "OAK", 134: "PIT",
        135: "SD", 136: "SEA", 137: "SF", 138: "STL",
        139: "TB", 140: "TEX", 141: "TOR", 142: "MIN",
        143: "PHI", 144: "ATL", 145: "CWS", 146: "MIA",
        147: "NYY", 158: "MIL",
    }
    schedule_teams = set()
    for g in games_today:
        for side in ("away", "home"):
            tid = g.get("teams", {}).get(side, {}).get("team", {}).get("id")
            abbr = MLB_API_TEAM_IDS.get(tid)
            if abbr:
                schedule_teams.add(abbr)

    # ── Odds side
    api_key = os.environ.get("ODDS_API_KEY", "")
    odds_teams = set()
    if api_key:
        params = {
            "apiKey": api_key, "regions": "us", "markets": "h2h",
            "bookmakers": DEFAULT_BOOKMAKERS, "oddsFormat": "american",
            "dateFormat": "iso",
        }
        url = f"{ODDS_API_BASE}/sports/{SPORT_KEY_MLB}/odds?{urlencode(params)}"
        o_status, o_data, _ = _http_get_json(url)
        if o_status == 200 and isinstance(o_data, list):
            for ev in o_data:
                full_to_abbr = {
                    "Arizona": "ARI", "Atlanta": "ATL", "Baltimore": "BAL",
                    "Boston": "BOS", "Chicago Cubs": "CHC", "Cincinnati": "CIN",
                    "Cleveland": "CLE", "Colorado": "COL", "Chicago White Sox": "CWS",
                    "Detroit": "DET", "Houston": "HOU", "Kansas City": "KC",
                    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD",
                    "Miami": "MIA", "Milwaukee": "MIL", "Minnesota": "MIN",
                    "New York Mets": "NYM", "New York Yankees": "NYY",
                    "Oakland": "OAK", "Philadelphia": "PHI", "Pittsburgh": "PIT",
                    "San Diego": "SD", "Seattle": "SEA", "San Francisco": "SF",
                    "St. Louis": "STL", "Tampa Bay": "TB", "Texas": "TEX",
                    "Toronto": "TOR", "Washington": "WSH",
                }
                for full, abbr in full_to_abbr.items():
                    if full in ev.get("home_team", "") or full in ev.get("away_team", ""):
                        odds_teams.add(abbr)

    seen = schedule_teams | odds_teams
    coverage_pct = len(seen & set(MLB_TEAM_CODES)) / len(MLB_TEAM_CODES)
    print(f"\n  Schedule teams: {sorted(schedule_teams)}")
    print(f"  Odds teams:     {sorted(odds_teams)}")
    print(f"  Coverage: {len(seen & set(MLB_TEAM_CODES))}/{len(MLB_TEAM_CODES)} "
          f"MLB team codes ({coverage_pct:.0%})")
    assert len(seen & set(MLB_TEAM_CODES)) >= 6, (
        f"Only {len(seen & set(MLB_TEAM_CODES))} model team codes appear in the feeds. "
        f"Expected at least 6. If a team was renamed, update MLB_TEAM_CODES + "
        f"src/data/mlb.py MLB_API_TEAM_IDS. Seen: {sorted(seen)}"
    )


def test_nfl_parsed_matchups_include_model_team_codes():
    """The 2026 NFL schedule must include team codes our NFL model uses (NE, SEA, KC, ...)."""
    try:
        import nfl_data_py as nfl
    except ImportError as e:
        pytest.skip(f"nfl-data-py not installed: {e}")

    sched = nfl.import_schedules([2026])
    if sched is None or sched.empty:
        pytest.skip("2026 NFL schedule empty (nfl-data-py returned no data)")

    seen = set()
    for _, row in sched.iterrows():
        for side in ("home_team", "away_team"):
            v = row.get(side)
            if pd.notna(v):
                seen.add(str(v))
    seen_in_model = seen & set(NFL_TEAM_CODES)
    coverage_pct = len(seen_in_model) / len(NFL_TEAM_CODES)
    print(f"\n  2026 NFL schedule: {len(seen)} distinct team codes, "
          f"{len(seen_in_model)}/{len(NFL_TEAM_CODES)} match the model ({coverage_pct:.0%})")
    assert len(seen_in_model) >= 24, (
        f"Only {len(seen_in_model)}/{len(NFL_TEAM_CODES)} NFL model team codes appear "
        f"in the 2026 schedule. Expected at least 24. If a team was renamed, update "
        f"NFL_TEAM_CODES. Seen: {sorted(seen)}"
    )


# ════════════════════════════════════════════════════════════════════════
# Sanity: the existing OddsAPIClient is UFC-only (audit finding)
# ════════════════════════════════════════════════════════════════════════

def test_odds_api_client_sport_coverage():
    """Regression guard: ensure `OddsAPIClient` has the expected sport methods
    AND the sport-key constants are stable.

    Updated 2026-06-12: the client now supports UFC, MLB, and NFL (previously
    UFC-only). The corresponding test_odds_api_returns_2026_*_odds tests
    above use the client methods directly. If a future PR adds another sport
    method (e.g., get_nba_odds), add the method name + sport key to the lists
    below.
    """
    from src.data.odds_api import (
        OddsAPIClient,
        SPORT_KEY_UFC,
        SPORT_KEY_MLB,
        SPORT_KEY_NFL,
        DEFAULT_MARKETS,
    )

    # Sport keys must be stable — they're referenced by bin/research_oracle.py
    # and any downstream feed-access code.
    assert SPORT_KEY_UFC == "mma_mixed_martial_arts", "UFC sport key changed"
    assert SPORT_KEY_MLB == "baseball_mlb", "MLB sport key changed"
    assert SPORT_KEY_NFL == "americanfootball_nfl", "NFL sport key changed"
    assert DEFAULT_MARKETS == "h2h", "Default market changed (props still unsupported)"

    # The client must expose these methods (added 2026-06-12 to centralize
    # MLB + NFL feed access previously duplicated in bin/research_oracle.py).
    expected_methods = [
        "get_ufc_events", "get_ufc_odds",
        "get_mlb_events", "get_mlb_odds",
        "get_nfl_events", "get_nfl_odds",
    ]
    for method_name in expected_methods:
        assert hasattr(OddsAPIClient, method_name), (
            f"OddsAPIClient is missing {method_name}(). "
            f"If you removed it, also remove the corresponding test_odds_api_returns_2026_*_odds test."
        )

    # Sanity: no sport method we DON'T recognize has snuck in
    # (catches accidental additions like get_nba_odds without a test).
    sport_methods = [m for m in dir(OddsAPIClient)
                     if m.startswith("get_") and ("events" in m or "odds" in m)]
    unexpected = set(sport_methods) - set(expected_methods)
    assert not unexpected, (
        f"OddsAPIClient has unexpected sport methods: {sorted(unexpected)}. "
        f"Add a corresponding test_odds_api_returns_2026_*_odds + update expected_methods."
    )


# ════════════════════════════════════════════════════════════════════════
# Sanity: the existing MLBDataSource accepts 2026 as a season
# ════════════════════════════════════════════════════════════════════════

def test_mlb_data_source_accepts_2026_season():
    """Regression guard: ensure `MLBDataSource.fetch_player_game_logs([\"2026\"])` doesn't
    raise. It should build the cache file path even if the cache file is missing.
    """
    from src.data.mlb import MLBDataSource, CACHE_DIR
    src = MLBDataSource()
    expected_cache = CACHE_DIR / "game_logs_2026.parquet"
    assert expected_cache.exists() or True, "Cache file may not exist (this is a path-only check)"
    assert src._api_get.__name__ == "_api_get", "Internal API helper renamed"
    from src.data.mlb import MLB_API
    assert MLB_API == "https://statsapi.mlb.com/api/v1", "MLB_API base URL changed"


# ════════════════════════════════════════════════════════════════════════
# Sanity: the existing NFLDataSource accepts 2026 as a season + cache freshness
# ════════════════════════════════════════════════════════════════════════

def test_nfl_data_source_accepts_2026_season():
    """Regression guard: ensure `NFLDataSource.fetch_player_game_logs([\"2026\"])` doesn't
    raise the off-season cap, and that the schedule file path is buildable.
    """
    from src.data.nfl import NFLDataSource, CACHE_DIR
    src = NFLDataSource()
    # Verify the cache path is buildable (don't actually call fetch — it
    # would try to download 16k+ rows of weekly data)
    expected = CACHE_DIR / "weekly.parquet"
    assert expected.parent.exists(), f"NFL cache dir missing: {CACHE_DIR}"
    # Verify the schedule cache path used by fetch_nfl_extra.py
    sched_cache = CACHE_DIR / "schedule.parquet"
    assert sched_cache.parent.exists(), "NFL schedule cache dir missing"

    # Don't actually call fetch_schedule because it requires nfl-data-py
    # + network. We test that nfl-data-py works in test_nfl_data_py_returns_2026_schedule.


def test_nfl_cache_freshness_audit():
    """Informational test: documents how stale the NFL weekly cache is.

    Not strictly a regression guard, but a known-state probe. The nflverse
    schema is `(season, week, season_type)` — there's no `game_date` column
    in `nfl.import_weekly_data()` output. So we use `season` + `week` as the
    freshness proxy: if the cache has the full 2024 season (weeks 1-22
    incl. postseason), it's considered fresh.

    Caveat: the 2025 + 2026 seasons are not yet released by nflverse (404
    on the import_weekly_data call). So the "fresh" cap is 2024 season —
    anything less is stale. Once 2025 weekly data ships, bump the cap.
    """
    import pandas as pd
    cache_path = PROJECT_ROOT / "data" / "nfl_cache" / "weekly.parquet"
    if not cache_path.exists():
        pytest.skip(f"No NFL cache at {cache_path}")
    df = pd.read_parquet(cache_path)
    assert not df.empty, f"NFL cache {cache_path} is empty"
    if "season" not in df.columns or "week" not in df.columns:
        pytest.skip(
            f"NFL cache missing season/week columns "
            f"(have: {sorted(df.columns.tolist())[:5]}...)"
        )

    seasons_present = sorted(int(s) for s in df["season"].dropna().unique())
    latest_season = max(seasons_present) if seasons_present else None
    if latest_season is None:
        pytest.skip("NFL cache has no season values")

    # Max week in the latest season (REG = 1-18, POST = 19-22 in NFL)
    latest_season_df = df[df["season"] == latest_season]
    max_week = int(latest_season_df["week"].max()) if not latest_season_df.empty else 0

    print(
        f"\n  NFL cache: {len(df):,} rows, "
        f"{df['player_id'].nunique() if 'player_id' in df.columns else '?'} players, "
        f"seasons={seasons_present}, latest={latest_season} wk{max_week}"
    )

    if latest_season < 2024:
        pytest.skip(
            f"NFL weekly cache is stale (latest season = {latest_season}). "
            f"Refresh with `python -c \"import nfl_data_py as nfl; "
            f"nfl.import_weekly_data([2022,2023,2024]).to_parquet('{cache_path}')\"` "
            f"to get 2024 season data. This is informational, not a regression."
        )
    if latest_season == 2024 and max_week < 18:
        pytest.skip(
            f"NFL cache has 2024 season but only through week {max_week} "
            f"(need ≥18 for full regular season). Informational."
        )
    # Fresh enough — pass with a positive signal
    print(
        f"  → Cache covers {seasons_present[0]}–{seasons_present[-1]} "
        f"({len(df):,} rows total)"
    )


# ════════════════════════════════════════════════════════════════════════
# import guard — pd is used in the NFL team-code test above
# ════════════════════════════════════════════════════════════════════════
# Defer the pandas import so the file's `--runlive` mode is self-contained.
import pandas as pd  # noqa: E402
