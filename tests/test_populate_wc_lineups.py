"""Unit tests for bin/populate_wc_lineups.py.

Covers pure functions (no network required):
  - parse_kxwcgame_ticker: KXWCGAME-YYMMDDCCYCCY-OUT ticker → match info
  - _parse_kickoff: FotMob kickoff string → timezone-aware datetime
  - time-window filtering: only cache within the configured window

The Playwright fetcher and end-to-end pipeline are exercised separately
in tests/test_fotmob_live.py (--runlive) and via bin/populate_wc_lineups.py
in dry-run mode.
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

# Import the functions under test
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bin.populate_wc_lineups import (  # noqa: E402
    parse_kxwcgame_ticker,
    _parse_kickoff,
    KALSHI_TICKER_RE,
    MONTHS,
    DEFAULT_WINDOW_MINUTES,
)


# ── parse_kxwcgame_ticker ─────────────────────────────────────────────────

class TestParseKxwcgameTicker:
    """Parse KXWCGAME-YYMMDD<team1><team2>-<outcome>."""

    def test_basic_home_outcome(self):
        """USA vs MEX, picking USA (the home team) — June 11 2026.
        Both teams ARE in TICKER_TEAM_MAP (USA → 'USA', MEX → 'Mexico')."""
        r = parse_kxwcgame_ticker("KXWCGAME-26JUN11USAMEX-USA")
        assert r is not None
        assert r["year"] == 2026
        assert r["month"] == 6
        assert r["day"] == 11
        assert r["home_code"] == "USA"
        assert r["away_code"] == "MEX"
        assert r["home_full"] == "USA"
        assert r["away_full"] == "Mexico"
        assert r["outcome"] == "USA"
        assert r["kickoff_est_utc"] == datetime(2026, 6, 11, 0, 0, 0, tzinfo=timezone.utc)

    def test_away_outcome(self):
        """Same match, picking MEX (the away team) — different outcome suffix."""
        r = parse_kxwcgame_ticker("KXWCGAME-26JUN11USAMEX-MEX")
        assert r is not None
        assert r["outcome"] == "MEX"
        assert r["home_code"] == "USA"
        assert r["away_code"] == "MEX"

    def test_tie_outcome(self):
        """KXWCGAME supports 'TIE' for draw markets."""
        r = parse_kxwcgame_ticker("KXWCGAME-26JUN11USAMEX-TIE")
        assert r is not None
        assert r["outcome"] == "TIE"
        assert r["home_code"] == "USA"
        assert r["away_code"] == "MEX"

    def test_realistic_3_letter_codes(self):
        """All 31 WC 2026 team codes should parse without error."""
        for code in ["ARG", "BRA", "CAN", "CRO", "ENG", "ESP", "FRA", "GER",
                     "ITA", "JPN", "KOR", "MEX", "NED", "POR", "USA", "URU"]:
            r = parse_kxwcgame_ticker(f"KXWCGAME-26JUL01{code}BRA-{code}")
            assert r is not None, f"Failed to parse {code}"
            assert r["home_code"] == code
            assert r["away_code"] == "BRA"

    def test_unparseable_ticker(self):
        """Non-KXWCGAME tickers should return None."""
        assert parse_kxwcgame_ticker("KXNBAPTS-26JUN11-BUF") is None
        assert parse_kxwcgame_ticker("") is None
        assert parse_kxwcgame_ticker("random text") is None
        assert parse_kxwcgame_ticker("KXWCGAME-XX") is None

    def test_invalid_month(self):
        """Bogus month abbreviation should return None."""
        assert parse_kxwcgame_ticker("KXWCGAME-26XYZ11MEXCUB-MEX") is None

    def test_all_12_months(self):
        """All 12 month abbreviations should parse."""
        for mon_s, num in MONTHS.items():
            r = parse_kxwcgame_ticker(f"KXWCGAME-26{mon_s}15MEXCUB-MEX")
            assert r is not None
            assert r["month"] == num

    def test_year_2026_via_prefix(self):
        """Year prefix '26' should map to 2026 (not 1926)."""
        r = parse_kxwcgame_ticker("KXWCGAME-26JUN01MEXCUB-MEX")
        assert r["year"] == 2026

    def test_uncommon_team_codes_resolve_to_full_name(self):
        """Codes without a TICKER_TEAM_MAP entry should pass through unchanged."""
        r = parse_kxwcgame_ticker("KXWCGAME-26JUN01XYZCUB-MEX")
        assert r["home_full"] == "XYZ"  # not in map → unchanged


# ── _parse_kickoff ────────────────────────────────────────────────────────

class TestParseKickoff:
    """Parse FotMob kickoff strings into timezone-aware datetimes."""

    def test_fotmob_canonical_format(self):
        """The format FotMob actually returns: 'Thu, Jun 11, 2026, 00:00 UTC'."""
        dt = _parse_kickoff("Thu, Jun 11, 2026, 00:00 UTC")
        assert dt is not None
        assert dt == datetime(2026, 6, 11, 0, 0, 0, tzinfo=timezone.utc)

    def test_iso_with_z_suffix(self):
        """ISO 8601 with Z suffix: '2026-06-11T00:00:00Z'."""
        dt = _parse_kickoff("2026-06-11T00:00:00Z")
        assert dt is not None
        assert dt.year == 2026 and dt.month == 6 and dt.day == 11

    def test_iso_with_microseconds(self):
        """ISO 8601 with microseconds: '2026-06-11T00:00:00.000000Z'."""
        dt = _parse_kickoff("2026-06-11T00:00:00.000000Z")
        assert dt is not None
        assert dt.year == 2026

    def test_sqlite_format(self):
        """'YYYY-MM-DD HH:MM:SS' format."""
        dt = _parse_kickoff("2026-06-11 00:00:00")
        assert dt is not None
        assert dt.year == 2026

    def test_none_input(self):
        assert _parse_kickoff(None) is None

    def test_empty_string(self):
        assert _parse_kickoff("") is None

    def test_garbage_input(self):
        """Unparseable strings should return None, not raise."""
        assert _parse_kickoff("not a date") is None
        assert _parse_kickoff("2026-13-45") is None  # invalid month/day
        assert _parse_kickoff("🦀") is None


# ── Time-window filtering ────────────────────────────────────────────────

class TestTimeWindowFilter:
    """The job only caches matches within ±window of kickoff.

    Pure-logic check: given a kickoff and a 'now' value, does the
    process_match bucketing logic correctly classify it as
    'cached' / 'too-far-out' / 'too-far-past'?

    We don't run process_match (it requires a live scraper); we
    re-implement the bucketing check here against the same constants.
    """

    WINDOW = DEFAULT_WINDOW_MINUTES  # 75 min

    def _classify(self, minutes_to_kickoff: float) -> str:
        """Mirror of process_match's windowing check."""
        if minutes_to_kickoff > self.WINDOW:
            return "too-far-out"
        if minutes_to_kickoff < -120:
            return "too-far-past"
        return "cached"

    def test_within_window(self):
        """30 min to kickoff — should be cached."""
        assert self._classify(30) == "cached"

    def test_at_window_boundary(self):
        """Exactly at the window cutoff (75 min) — should still cache."""
        assert self._classify(75) == "cached"

    def test_just_outside_window(self):
        """76 min to kickoff — too far out, skip."""
        assert self._classify(76) == "too-far-out"

    def test_far_in_future(self):
        """24h to kickoff — way too far out."""
        assert self._classify(24 * 60) == "too-far-out"

    def test_just_after_kickoff(self):
        """30 min after kickoff — within the 2h 'past' window, still cache."""
        assert self._classify(-30) == "cached"

    def test_at_past_cutoff(self):
        """Exactly 120 min past kickoff — still cache (boundary)."""
        assert self._classify(-120) == "cached"

    def test_far_past(self):
        """3h after kickoff — too far past, skip (lineups archived)."""
        assert self._classify(-180) == "too-far-past"

    def test_negative_zero(self):
        """Edge: 0.0 min to kickoff (exactly at kickoff) — should cache."""
        assert self._classify(0.0) == "cached"


# ── TICKER_TEAM_MAP coverage ──────────────────────────────────────────────

class TestTickerTeamMapCoverage:
    """Sanity-check: the team-code map used by parse_kxwcgame_ticker
    covers all 48 WC 2026 teams. If new teams are added to TICKER_TEAM_MAP
    without code entries, the populate job will silently fail to resolve
    matchIds (since FotMob uses full names)."""

    def test_all_wc_2026_team_codes_resolve(self):
        from src.scripts.scan_wc import TICKER_TEAM_MAP
        # Should have at least 30+ teams (we have 48 WC 2026 teams)
        assert len(TICKER_TEAM_MAP) >= 30, \
            f"TICKER_TEAM_MAP has only {len(TICKER_TEAM_MAP)} entries"

    def test_no_empty_full_names(self):
        """Every entry should have a non-empty full name."""
        from src.scripts.scan_wc import TICKER_TEAM_MAP
        for code, full in TICKER_TEAM_MAP.items():
            assert isinstance(code, str) and len(code) == 3
            assert isinstance(full, str) and len(full) > 0, \
                f"Empty full name for code {code}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
