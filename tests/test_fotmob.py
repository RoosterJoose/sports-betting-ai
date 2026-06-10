"""Unit tests for src/data/fotmob.py.

Tests the pure functions in the FotMob module (no network/browser required).
The Playwright scraper is exercised separately in scripts/test_fotmob_scraper.py.
"""
from datetime import datetime, timedelta

from src.data.fotmob import (
    LineupCache, compute_key_player_out, load_star_players,
)


# --- Synthetic XIs -----------------------------------------------------------

# XI where Messi is missing from Argentina (synthetic test data)
ARGENTINA_XI_NO_MESSI = [
    {"name": "Emiliano Martínez", "position": "GK", "shirt": 23},
    {"name": "Nahuel Molina", "position": "RB", "shirt": 26},
    {"name": "Cristian Romero", "position": "CB", "shirt": 13},
    {"name": "Lisandro Martínez", "position": "CB", "shirt": 25},
    {"name": "Nicolás Tagliafico", "position": "LB", "shirt": 3},
    {"name": "Rodrigo De Paul", "position": "CM", "shirt": 7},
    {"name": "Enzo Fernández", "position": "CM", "shirt": 24},
    {"name": "Alexis Mac Allister", "position": "CM", "shirt": 20},
    {"name": "Ángel Di María", "position": "RW", "shirt": 11},
    {"name": "Julián Álvarez", "position": "ST", "shirt": 9},
    {"name": "Lautaro Martínez", "position": "LW", "shirt": 22},
]

# XI with Messi present (synthetic test data)
ARGENTINA_XI_WITH_MESSI = ARGENTINA_XI_NO_MESSI + [
    {"name": "Lionel Messi", "position": "RW", "shirt": 10},
]
# Move Messi to the start so he's in the XI list
ARGENTINA_XI_WITH_MESSI = [
    {"name": "Lionel Messi", "position": "RW", "shirt": 10},
    {"name": "Emiliano Martínez", "position": "GK", "shirt": 23},
    {"name": "Nahuel Molina", "position": "RB", "shirt": 26},
    {"name": "Cristian Romero", "position": "CB", "shirt": 13},
    {"name": "Lisandro Martínez", "position": "CB", "shirt": 25},
    {"name": "Nicolás Tagliafico", "position": "LB", "shirt": 3},
    {"name": "Rodrigo De Paul", "position": "CM", "shirt": 7},
    {"name": "Enzo Fernández", "position": "CM", "shirt": 24},
    {"name": "Alexis Mac Allister", "position": "CM", "shirt": 20},
    {"name": "Julián Álvarez", "position": "ST", "shirt": 9},
    {"name": "Lautaro Martínez", "position": "LW", "shirt": 22},
]

# Brazil XI with Vinícius Júnior present
BRAZIL_XI_WITH_VINICIUS = [
    {"name": "Alisson", "position": "GK", "shirt": 1},
    {"name": "Danilo", "position": "RB", "shirt": 2},
    {"name": "Marquinhos", "position": "CB", "shirt": 4},
    {"name": "Thiago Silva", "position": "CB", "shirt": 3},
    {"name": "Alex Sandro", "position": "LB", "shirt": 6},
    {"name": "Casemiro", "position": "DM", "shirt": 5},
    {"name": "Bruno Guimarães", "position": "CM", "shirt": 17},
    {"name": "Lucas Paquetá", "position": "AM", "shirt": 8},
    {"name": "Raphinha", "position": "RW", "shirt": 11},
    {"name": "Rodrygo", "position": "LW", "shirt": 10},
    {"name": "Vinícius Júnior", "position": "ST", "shirt": 7},
]

# Brazil XI missing Vinícius Júnior
BRAZIL_XI_NO_VINICIUS = [p for p in BRAZIL_XI_WITH_VINICIUS
                          if "Vinícius" not in p["name"]]


# --- Tests for compute_key_player_out ----------------------------------------

def test_star_missing_from_home_xi_returns_1():
    """ARG XI without Messi (a star per data/wc_star_players.json) → returns 1."""
    lineups = {"home": ARGENTINA_XI_NO_MESSI, "away": BRAZIL_XI_WITH_VINICIUS}
    result = compute_key_player_out("ARG", "BRA", lineups)
    assert result == 1, f"Expected 1 (Messi missing), got {result}"


def test_star_missing_from_away_xi_returns_1():
    """BRA XI without Vinícius Júnior → returns 1."""
    lineups = {"home": ARGENTINA_XI_WITH_MESSI, "away": BRAZIL_XI_NO_VINICIUS}
    result = compute_key_player_out("ARG", "BRA", lineups)
    assert result == 1, f"Expected 1 (Vinícius missing), got {result}"


def test_all_stars_present_returns_0():
    """Both teams with all their stars in the XI → returns 0."""
    lineups = {"home": ARGENTINA_XI_WITH_MESSI, "away": BRAZIL_XI_WITH_VINICIUS}
    result = compute_key_player_out("ARG", "BRA", lineups)
    assert result == 0, f"Expected 0 (all stars present), got {result}"


def test_no_lineups_returns_0():
    """Empty lineups dict → returns 0 (no info, no flag)."""
    result = compute_key_player_out("ARG", "BRA", {})
    assert result == 0, f"Expected 0, got {result}"


def test_team_not_in_star_list_returns_0():
    """Team code with no entry in wc_star_players.json → returns 0."""
    lineups = {"home": [{"name": "Some Player"}], "away": [{"name": "Other"}]}
    # SUR is in the list but use a non-existent code
    result = compute_key_player_out("ZZZ", "QQQ", lineups)
    assert result == 0, f"Expected 0 (unknown teams), got {result}"


def test_case_insensitive_name_match():
    """Name matching should be case-insensitive ('lionel messi' should match 'Lionel Messi')."""
    # Full ARG XI but with Messi's name in lowercase — case-insensitive match should find it
    arg_xi_lowercase_messi = [
        {"name": "lionel messi", "position": "RW", "shirt": 10},
        {"name": "Emiliano Martínez", "position": "GK", "shirt": 23},
        {"name": "Cristian Romero", "position": "CB", "shirt": 13},
        {"name": "Rodrigo De Paul", "position": "CM", "shirt": 7},
        {"name": "Alexis Mac Allister", "position": "CM", "shirt": 20},
        {"name": "Enzo Fernández", "position": "CM", "shirt": 24},
        {"name": "Julián Álvarez", "position": "ST", "shirt": 9},
    ]
    lineups = {"home": arg_xi_lowercase_messi, "away": BRAZIL_XI_WITH_VINICIUS}
    result = compute_key_player_out("ARG", "BRA", lineups)
    assert result == 0, f"Expected 0 (case-insensitive match should find Messi), got {result}"


def test_substring_name_match():
    """Substring match: 'Messi' should match 'LIONEL MESSI' (all caps)."""
    # Full ARG XI but with Messi's name in all caps
    arg_xi_caps_messi = [
        {"name": "LIONEL MESSI", "position": "RW", "shirt": 10},
        {"name": "Emiliano Martínez", "position": "GK", "shirt": 23},
        {"name": "Cristian Romero", "position": "CB", "shirt": 13},
        {"name": "Rodrigo De Paul", "position": "CM", "shirt": 7},
        {"name": "Alexis Mac Allister", "position": "CM", "shirt": 20},
        {"name": "Enzo Fernández", "position": "CM", "shirt": 24},
        {"name": "Julián Álvarez", "position": "ST", "shirt": 9},
    ]
    lineups = {"home": arg_xi_caps_messi, "away": BRAZIL_XI_WITH_VINICIUS}
    result = compute_key_player_out("ARG", "BRA", lineups)
    assert result == 0, f"Expected 0 (substring match), got {result}"


# --- Tests for LineupCache --------------------------------------------------

def test_lineup_cache_set_get_roundtrip(tmp_path):
    """LineupCache.set() then .get() returns the same data with fetched_at timestamp."""
    cache = LineupCache(path=tmp_path / "lineups.json")
    lineups = {
        "home": [{"name": "Player 1"}],
        "away": [{"name": "Player 2"}],
        "status": "ok",
    }
    ticker = "KXWCGAME-TEST-ARGBRA-ARG"
    cache.set(ticker, lineups)

    retrieved = cache.get(ticker)
    assert retrieved is not None
    assert retrieved["home"] == lineups["home"]
    assert retrieved["away"] == lineups["away"]
    assert "fetched_at" in retrieved, "fetched_at timestamp should be auto-added"


def test_lineup_cache_is_fresh(tmp_path):
    """A freshly-set entry should be fresh; a manually-aged one should not."""
    cache = LineupCache(path=tmp_path / "lineups.json")
    ticker = "KXWCGAME-TEST-ARGBRA-ARG"
    cache.set(ticker, {"home": [{"name": "x"}], "away": [{"name": "y"}]})

    # Fresh
    assert cache.is_fresh(ticker, max_age_hours=1)

    # Manually age the entry to 2 hours ago
    entry = cache.get(ticker)
    old_ts = (datetime.now() - timedelta(hours=2)).isoformat()
    entry["fetched_at"] = old_ts
    cache.set(ticker, entry)  # re-save with old timestamp

    # Now stale
    assert not cache.is_fresh(ticker, max_age_hours=1)


def test_lineup_cache_missing_ticker(tmp_path):
    """is_fresh() and get() should return False / None for missing tickers."""
    cache = LineupCache(path=tmp_path / "lineups.json")
    assert cache.get("NONEXISTENT-TICKER") is None
    assert not cache.is_fresh("NONEXISTENT-TICKER")


# --- Test for load_star_players ----------------------------------------------

def test_load_star_players_filters_metadata():
    """load_star_players() should strip _comment/_version/_updated keys."""
    stars = load_star_players()
    assert isinstance(stars, dict)
    assert "_comment" not in stars
    assert "_version" not in stars
    assert "_updated" not in stars
    # Should have entries for major WC teams
    assert "ARG" in stars
    assert "BRA" in stars
    assert "FRA" in stars
    # Each entry is a list of player names
    assert isinstance(stars["ARG"], list)
    assert len(stars["ARG"]) > 0
    assert "Lionel Messi" in stars["ARG"]


# --- CLI runner for ad-hoc testing -------------------------------------------

if __name__ == "__main__":
    # Allow running as `python -m tests.test_fotmob` for quick smoke testing
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
