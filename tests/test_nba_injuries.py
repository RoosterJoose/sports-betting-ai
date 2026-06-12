"""Unit tests for src/data/nba_injuries.py — fuzzy name matching.

Tests the _normalize_name() + _names_match() helpers added to close the
McBride / suffix / accent / nickname gap in NBA injury filtering.
"""
import sys
from pathlib import Path

# Make src/ importable when running pytest from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.nba_injuries import _normalize_name, _names_match  # noqa: E402


# === _normalize_name() ===

def test_normalize_lowercases():
    assert _normalize_name("Miles McBride") == "miles mcbride"


def test_normalize_strips_period():
    assert _normalize_name("Michael Porter Jr.") == "michael porter"


def test_normalize_strips_jr():
    assert _normalize_name("Michael Porter Jr") == "michael porter"


def test_normalize_strips_sr():
    assert _normalize_name("Scotty Pippen Sr") == "scotty pippen"


def test_normalize_strips_iii():
    assert _normalize_name("Marvin Bagley III") == "marvin bagley"


def test_normalize_strips_ii():
    assert _normalize_name("Dereck Lively II") == "dereck lively"


def test_normalize_strips_iv():
    assert _normalize_name("Player Name IV") == "player name"


def test_normalize_strips_accents():
    assert _normalize_name("Luka Dončić") == "luka doncic"


def test_normalize_strips_accents_2():
    assert _normalize_name("Nikola Jokić") == "nikola jokic"


def test_normalize_handles_nickname():
    # Quotes become spaces; nickname remains in name
    assert _normalize_name("Miles 'Deuce' McBride") == "miles deuce mcbride"


def test_normalize_handles_apostrophe():
    assert _normalize_name("De'Aaron Fox") == "de aaron fox"


def test_normalize_empty():
    assert _normalize_name("") == ""


def test_normalize_collapses_whitespace():
    assert _normalize_name("  John   Doe  ") == "john doe"


# === _names_match() ===

def test_names_match_exact():
    assert _names_match("Miles McBride", "Miles McBride") is True


def test_names_match_case_insensitive():
    assert _names_match("MILES MCBRIDE", "miles mcbride") is True


def test_names_match_suffix_differs():
    # The McBride / Porter case — ESPN has suffix, Kalshi doesn't
    assert _names_match("Michael Porter", "Michael Porter Jr.") is True
    assert _names_match("Michael Porter Jr.", "Michael Porter") is True
    assert _names_match("Michael Porter Jr.", "Michael Porter Jr") is True


def test_names_match_roman_numerals():
    assert _names_match("Marvin Bagley", "Marvin Bagley III") is True
    assert _names_match("Marvin Bagley III", "Marvin Bagley") is True
    assert _names_match("Dereck Lively II", "Dereck Lively") is True


def test_names_match_nickname():
    # Kalshi uses "Miles 'Deuce' McBride" with nickname, ESPN uses "Miles McBride"
    # Both share "Miles" and "McBride" → match
    assert _names_match("Miles 'Deuce' McBride", "Miles McBride") is True
    assert _names_match("Miles McBride", "Miles 'Deuce' McBride") is True


def test_names_match_accents():
    assert _names_match("Luka Doncic", "Luka Dončić") is True
    assert _names_match("Luka Dončić", "Luka Doncic") is True
    assert _names_match("Nikola Jokic", "Nikola Jokić") is True


def test_names_match_first_initial():
    # "M. Porter" vs "Michael Porter"
    assert _names_match("M. Porter", "Michael Porter") is True
    assert _names_match("Michael Porter", "M Porter") is True


def test_names_match_different_player():
    # Different last name → False
    assert _names_match("Miles McBride", "John Smith") is False
    assert _names_match("LeBron James", "Kevin James") is False


def test_names_match_first_name_only_differs():
    # Same last name, different first name → False (different player)
    assert _names_match("Gary Trent", "Gary Trent Jr.") is True  # suffix match
    assert _names_match("John Porter", "Michael Porter") is False  # diff first


def test_names_match_empty():
    assert _names_match("", "Miles McBride") is False
    assert _names_match("Miles McBride", "") is False
    assert _names_match("", "") is False


if __name__ == "__main__":
    # Allow running as `python -m tests.test_nba_injuries` for quick smoke testing
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
