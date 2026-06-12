"""Unit tests for check_in_season_stale() in src/utils/preflight.py.

Verifies the UFC special-case (uses models/ufc/fighter_augment.json mtime
as the data source instead of a parquet cache):
  (a) is in failures when mtime is >14d
  (b) is NOT in failures when mtime is <14d
  (c) is silently skipped when the file is missing
  (d) always passes the in-season gate regardless of today's date

Approach: monkeypatch `preflight._days_ago` to return a controlled value
(so the test deterministically tests the threshold logic without touching
the real file's mtime), and use the `patch_path_exists` fixture from
conftest.py to control the file side (only the UFC aug path is intercepted;
parquet files use the real `Path.exists`). For test (d), patch
`datetime.now` to a summer date (July 1, 2026) where NBA, NHL, NFL, and
CFB are all off-season — confirming UFC's in-season check doesn't depend
on the date.
"""
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

# Project root is added to sys.path by tests/conftest.py — no need to
# repeat that here (the previous `sys.path.insert(0, ...)` line was
# redundant once conftest.py handled it).

# SPORT_GLOBS and OFF_SEASON_ON_JUNE_11 are used in @pytest.mark.parametrize
# decorators, which are evaluated at module-load time — BEFORE pytest's
# fixture auto-injection runs. So they MUST be importable as plain Python
# names in this module's namespace. We use `from tests.conftest import`
# (slightly discouraged by pytest's docs in favor of auto-injection, but
# auto-injection only works for FIXTURES, not module-level constants)
# because the alternative — a separate `tests/_preflight_constants.py`
# module — is overkill for 2 shared constants. The factory fixtures
# (sport_failures, mock_data_range, patch_datetime) ARE auto-injected as
# test parameters below.

from tests.conftest import SPORT_GLOBS, OFF_SEASON_ON_JUNE_11

from src.utils import preflight
from src.utils.preflight import check_in_season_stale


# ─────────────────────────────────────────────────────────────────────
# (a) Stale mtime (>14d) → UFC is in failures
# ─────────────────────────────────────────────────────────────────────


def test_ufc_stale_mtime_appears_in_failures(patch_path_exists, ufc_entries):
    """(a) UFC is in failures when mtime is >14d old."""
    with patch_path_exists(ufc_exists=True), \
         patch.object(preflight, "_days_ago", return_value=20):
        failures = check_in_season_stale(threshold_days=14)

    ufc = ufc_entries(failures)
    assert len(ufc) == 1, f"Expected 1 UFC entry, got {len(ufc)}: {failures}"
    entry = ufc[0]
    assert entry["sport"] == "UFC"
    assert entry["age_days"] == 20
    assert entry["flag"] == "STALE 20d"
    assert entry["in_season"] is True
    # UFC uses mtime, not a date column — latest_date is None
    assert entry["latest_date"] is None
    # mtime is a YYYY-MM-DD date string (used by the pre-commit hook's
    # "mtime YYYY-MM-DD" display); mtime_str is set when the file exists.
    assert entry["mtime"] is not None, "UFC failure should include mtime date string"
    # Should look like a date: YYYY-MM-DD
    import re
    assert re.match(r"^\d{4}-\d{2}-\d{2}$", entry["mtime"]), (
        f"mtime should be YYYY-MM-DD, got {entry['mtime']!r}"
    )


# ─────────────────────────────────────────────────────────────────────
# (b) Fresh mtime (<14d) → UFC is NOT in failures
# ─────────────────────────────────────────────────────────────────────


def test_ufc_fresh_mtime_not_in_failures(patch_path_exists, ufc_entries):
    """(b) UFC is NOT in failures when mtime is <14d old."""
    with patch_path_exists(ufc_exists=True), \
         patch.object(preflight, "_days_ago", return_value=5):
        failures = check_in_season_stale(threshold_days=14)

    ufc = ufc_entries(failures)
    assert len(ufc) == 0, f"Expected 0 UFC entries, got {ufc}"


def test_ufc_at_threshold_boundary_not_in_failures(patch_path_exists, ufc_entries):
    """UFC at exactly threshold_days is NOT included (the guard uses strict >)."""
    with patch_path_exists(ufc_exists=True), \
         patch.object(preflight, "_days_ago", return_value=14):
        failures = check_in_season_stale(threshold_days=14)

    ufc = ufc_entries(failures)
    assert len(ufc) == 0, f"14d should NOT be stale (uses strict >), got {ufc}"


def test_ufc_just_above_threshold_in_failures(patch_path_exists, ufc_entries):
    """UFC at threshold_days + 1 IS included (the smallest stale value)."""
    with patch_path_exists(ufc_exists=True), \
         patch.object(preflight, "_days_ago", return_value=15):
        failures = check_in_season_stale(threshold_days=14)

    ufc = ufc_entries(failures)
    assert len(ufc) == 1
    assert ufc[0]["age_days"] == 15


# ─────────────────────────────────────────────────────────────────────
# (c) Missing file → silently skipped (no raise, no entry)
# ─────────────────────────────────────────────────────────────────────


def test_ufc_missing_file_silently_skipped(patch_path_exists, ufc_entries):
    """(c) UFC is silently skipped when fighter_augment.json doesn't exist.

    Function should not raise and should not include UFC in failures.
    """
    with patch_path_exists(ufc_exists=False):
        failures = check_in_season_stale(threshold_days=14)

    ufc = ufc_entries(failures)
    assert len(ufc) == 0, f"Expected 0 UFC entries when file missing, got {ufc}"


# ─────────────────────────────────────────────────────────────────────
# (d) In-season gate always passes for UFC (year-round sport)
# ─────────────────────────────────────────────────────────────────────


def test_ufc_in_season_gate_always_passes_in_summer(patch_path_exists, ufc_entries):
    """(d) UFC always passes the in-season gate regardless of today's date.

    Mock datetime.now() to July 1, 2026 — a date when NBA, NHL, NFL, and
    CFB are all off-season (verified manually against SPORT_CONFIG in
    src/utils/preflight.py). UFC is year-round (season_start=None,
    season_end=None) so its in-season check should still return True
    and the stale mtime should put it in failures.
    """
    summer = datetime(2026, 7, 1, 12, 0, 0)
    # `datetime.datetime` is immutable at the class level, so
    # `patch.object(preflight.datetime, "now", return_value=summer)`
    # raises `TypeError: cannot set 'now' attribute of immutable type
    # 'datetime.datetime'`. We instead patch the module-level binding
    # `preflight.datetime` (the imported class reference, mutable at
    # the binding level) and set `mock_dt.now.return_value` on it.
    # Mock scope: only `datetime.now()` is exercised by
    # `check_in_season_stale()` — `_days_ago` is patched separately so
    # its own `datetime.fromtimestamp()` use never reaches the mock.
    # If a future change to `check_in_season_stale` adds a new
    # `datetime` call (e.g. `datetime.utcnow()`), the mock would need
    # an explicit `mock_dt.<method>.return_value = ...` line.
    with patch("src.utils.preflight.datetime") as mock_dt, \
         patch_path_exists(ufc_exists=True), \
         patch.object(preflight, "_days_ago", return_value=30):
        mock_dt.now.return_value = summer
        failures = check_in_season_stale(threshold_days=14)

    ufc = ufc_entries(failures)
    assert len(ufc) == 1, (
        f"Expected 1 UFC entry on summer date (when NBA/NHL/NFL/CFB are "
        f"off-season), got {len(ufc)}: {failures}"
    )
    entry = ufc[0]
    assert entry["sport"] == "UFC"
    assert entry["in_season"] is True
    assert entry["age_days"] == 30
    assert entry["flag"] == "STALE 30d"
    assert entry["latest_date"] is None


# ─────────────────────────────────────────────────────────────────────
# Parquet-sport coverage for the other 7 sports
# (NBA, MLB, NHL, WNBA, WC, NFL, CFB) — the parquet-based path
# complements the UFC mtime-based path already tested above.
# ─────────────────────────────────────────────────────────────────────

# Per-sport data glob lookup + off-season-on-June-11 set + factory
# fixtures (sport_failures, mock_data_range, patch_datetime) now live in
# tests/conftest.py so future preflight test files can reuse them.
# SPORT_GLOBS and OFF_SEASON_ON_JUNE_11 are imported from conftest
# (see comment block above for why); factory fixtures are injected
# as test parameters.


@pytest.mark.parametrize("sport,glob", SPORT_GLOBS.items())
def test_parquet_sport_in_failures_when_stale_and_in_season(
    sport, glob, sport_failures, mock_data_range, patch_datetime,
):
    """Parquet sport is in failures when stale AND in-season on the test date.

    Today is fixed at 2026-06-11 — in-season for NBA, MLB, NHL, WNBA, WC
    (per SPORT_CONFIG season windows). NFL and CFB are off-season on this
    date and are asserted SKIPPED here. The off-season-skip behavior for
    those sports is also covered exhaustively by
    `test_off_season_sport_skipped_even_when_stale` below.

    latest_date is 30 days old (stale > 14d threshold).
    """
    today = datetime(2026, 6, 11, 12, 0, 0)
    latest_date = today - timedelta(days=30)
    with patch.object(preflight, "_data_range",
                      side_effect=mock_data_range({glob: latest_date})), \
         patch_datetime(today):
        failures = check_in_season_stale(threshold_days=14)

    sport_fails = sport_failures(failures, sport)
    if sport in OFF_SEASON_ON_JUNE_11:
        # Off-season on June 11: in-season gate fires before parquet read.
        assert len(sport_fails) == 0, (
            f"{sport}: should be SKIPPED on {today.date()} (off-season), "
            f"got {len(sport_fails)} failure(s): {sport_fails}"
        )
    else:
        assert len(sport_fails) == 1, (
            f"{sport}: expected 1 failure, got {len(sport_fails)}: {failures}"
        )
        entry = sport_fails[0]
        assert entry["sport"] == sport
        assert entry["age_days"] == 30
        assert entry["flag"] == "STALE 30d"
        assert entry["in_season"] is True
        # Parquet sports: latest_date is the parquet's max date; mtime is None.
        assert entry["latest_date"] == str(latest_date.date())
        assert entry["mtime"] is None


@pytest.mark.parametrize("sport,glob", SPORT_GLOBS.items())
def test_parquet_sport_not_in_failures_when_fresh(
    sport, glob, sport_failures, mock_data_range, patch_datetime,
):
    """Parquet sport is NOT in failures when in-season and data is fresh.

    latest_date is 5 days old (< 14d threshold). The sport should be
    in-season on June 11 (or skipped if off-season) but never in failures.
    """
    today = datetime(2026, 6, 11, 12, 0, 0)
    latest_date = today - timedelta(days=5)
    with patch.object(preflight, "_data_range",
                      side_effect=mock_data_range({glob: latest_date})), \
         patch_datetime(today):
        failures = check_in_season_stale(threshold_days=14)

    sport_fails = sport_failures(failures, sport)
    assert len(sport_fails) == 0, (
        f"{sport}: fresh data (5d) should never be in failures, got {sport_fails}"
    )


@pytest.mark.parametrize("sport,glob", SPORT_GLOBS.items())
def test_parquet_sport_missing_data_silently_skipped(
    sport, glob, sport_failures, mock_data_range, patch_datetime,
):
    """Parquet sport with missing data is silently skipped (no raise, no entry).

    `_data_range` returns latest_date=None when the parquet is missing;
    the function should skip the sport without raising.
    """
    today = datetime(2026, 6, 11, 12, 0, 0)
    with patch.object(preflight, "_data_range",
                      side_effect=mock_data_range({})), \
         patch_datetime(today):
        failures = check_in_season_stale(threshold_days=14)

    sport_fails = sport_failures(failures, sport)
    assert len(sport_fails) == 0, (
        f"{sport}: missing parquet should be silently skipped, got {sport_fails}"
    )


# ─────────────────────────────────────────────────────────────────────
# Off-season skip: a sport whose date window doesn't include today is
# excluded from failures even if its data is wildly stale.
# ─────────────────────────────────────────────────────────────────────

# (sport, glob, off-season today date) — verified manually against
# SPORT_CONFIG season windows in src/utils/preflight.py.
_OFF_SEASON_CASES = [
    # NBA: window (10, 1) - (6, 30) — off-season on July 15
    ("NBA",  "data/nba_cache/game_logs_v14.parquet",   datetime(2026, 7, 15, 12, 0, 0)),
    # NHL: window (10, 1) - (6, 30) — off-season on July 15
    ("NHL",  "data/nhl_cache/*.parquet",              datetime(2026, 7, 15, 12, 0, 0)),
    # MLB: window (3, 20) - (11, 15) — off-season on Feb 1
    ("MLB",  "data/cache/mlb/game_logs_*.parquet",    datetime(2026, 2, 1, 12, 0, 0)),
    # WNBA: window (5, 1) - (10, 31) — off-season on Feb 1
    ("WNBA", "data/wnba_cache/wnba_games.parquet",    datetime(2026, 2, 1, 12, 0, 0)),
    # NFL: window (9, 1) - (2, 28) — off-season on June 11 (real today)
    ("NFL",  "data/nfl_cache/*.parquet",              datetime(2026, 6, 11, 12, 0, 0)),
    # NFL: also off-season on July 1 (mid-summer)
    ("NFL",  "data/nfl_cache/*.parquet",              datetime(2026, 7, 1, 12, 0, 0)),
    # NFL: also off-season in May (between Feb-end and Sep-start)
    ("NFL",  "data/nfl_cache/*.parquet",              datetime(2026, 5, 15, 12, 0, 0)),
    # CFB: window (8, 15) - (1, 31) — off-season on June 11
    ("CFB",  "data/cfb_cache/*.parquet",              datetime(2026, 6, 11, 12, 0, 0)),
    # CFB: also off-season in May (between Jan-end and Aug-start)
    ("CFB",  "data/cfb_cache/*.parquet",              datetime(2026, 5, 15, 12, 0, 0)),
]


@pytest.mark.parametrize("sport,glob,off_today", _OFF_SEASON_CASES)
def test_off_season_sport_skipped_even_when_stale(
    sport, glob, off_today, sport_failures, mock_data_range, patch_datetime,
):
    """Off-season sport is excluded from failures even if data is 100d old.

    The in-season gate fires before the parquet read, so a wildly stale
    parquet on an off-season date produces zero failures. This is the
    documented behavior — pre-commit should NOT block commits for
    off-season sports (refresh can wait until the season resumes).
    """
    # Make data 100 days old (way over any threshold)
    latest_date = off_today - timedelta(days=100)
    with patch.object(preflight, "_data_range",
                      side_effect=mock_data_range({glob: latest_date})), \
         patch_datetime(off_today):
        failures = check_in_season_stale(threshold_days=14)

    sport_fails = sport_failures(failures, sport)
    assert len(sport_fails) == 0, (
        f"{sport}: should be SKIPPED on {off_today.date()} (off-season), "
        f"even with 100d-old data, but got {len(sport_fails)} failure(s): "
        f"{sport_fails}"
    )


# ─────────────────────────────────────────────────────────────────────
# In-season gate (focused pin): a sport with a defined season window
# is excluded from failures on an off-season date, even with very stale
# data. The same sport+data IS in failures on an in-season date — proving
# the in-season gate is the differentiator.
# ─────────────────────────────────────────────────────────────────────


def test_in_season_gate_excludes_nfl_on_off_season_date(
    sport_failures, mock_data_range, patch_datetime,
):
    """In-season gate excludes a sport with a defined season window on an
    off-season date, even if its data is wildly stale.

    Pins down the off-season-skip behavior with a focused, discoverable
    test case. The broader parametrized coverage lives in
    `test_off_season_sport_skipped_even_when_stale` above; this test
    makes the contract explicit for a single representative sport.

    NFL's season window is (9, 1) – (2, 28) per SPORT_CONFIG — the window
    wraps around the year-end (start_ord=901 > end_ord=228), so the
    in-season check is `today_ord >= 901 or today_ord <= 228`. On
    July 1 (today_ord=701), NFL is off-season. On January 15
    (today_ord=115), NFL IS in-season (the wrapped half of the window).

    The two halves of the test use the SAME staleness (200d) but different
    `today` dates — proving the in-season gate is the differentiator, not
    the data-pipeline. Note: the staleness is computed PER BLOCK (relative
    to that block's `today`) so the expected `age_days` is 200 in both.
    """
    nfl_glob = SPORT_GLOBS["NFL"]  # data/nfl_cache/*.parquet

    # ── Off-season date: July 1, 2026 ────────────────────────────────
    summer = datetime(2026, 7, 1, 12, 0, 0)
    summer_latest = summer - timedelta(days=200)  # Dec 13, 2025
    with patch.object(preflight, "_data_range",
                      side_effect=mock_data_range({nfl_glob: summer_latest})), \
         patch_datetime(summer):
        failures = check_in_season_stale(threshold_days=14)

    nfl_fails = sport_failures(failures, "NFL")
    assert len(nfl_fails) == 0, (
        f"NFL should be EXCLUDED on {summer.date()} (off-season, window "
        f"Sep 1 – Feb 28) even with 200d-old data, but got "
        f"{len(nfl_fails)} failure(s): {nfl_fails}"
    )
    # Also assert no other parquet-sport failures leaked in. UFC is
    # filtered out because it has its own real-file mtime (Jun 9, 2026)
    # that is unrelated to this test's NFL data — UFC's age on the test
    # dates varies (22d on Jul 1, -145d on Jan 15) but either way it's
    # not what we're testing here.
    nfl_test_only = [f for f in failures if f["sport"] != "UFC"]
    assert nfl_test_only == [], (
        f"Off-season test should produce no parquet-sport failures, "
        f"got: {nfl_test_only}"
    )

    # ── In-season date: January 15, 2026 ─────────────────────────────
    # NFL's window wraps year-end, so Jan 15 is in-season.
    winter = datetime(2026, 1, 15, 12, 0, 0)
    winter_latest = winter - timedelta(days=200)  # Jun 29, 2025
    with patch.object(preflight, "_data_range",
                      side_effect=mock_data_range({nfl_glob: winter_latest})), \
         patch_datetime(winter):
        failures = check_in_season_stale(threshold_days=14)

    nfl_fails = sport_failures(failures, "NFL")
    assert len(nfl_fails) == 1, (
        f"NFL should be IN FAILURES on {winter.date()} (in-season, window "
        f"Sep 1 – Feb 28) with 200d-old data, but got "
        f"{len(nfl_fails)} failure(s): {nfl_fails}"
    )
    entry = nfl_fails[0]
    assert entry["sport"] == "NFL"
    assert entry["age_days"] == 200
    assert entry["flag"] == "STALE 200d"
    assert entry["in_season"] is True
    assert entry["latest_date"] == str(winter_latest.date())


# ─────────────────────────────────────────────────────────────────────
# Threshold edge cases: exactly at threshold (uses strict >) vs just over
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("sport,glob", SPORT_GLOBS.items())
def test_parquet_sport_at_threshold_boundary_not_in_failures(
    sport, glob, sport_failures, mock_data_range, patch_datetime,
):
    """latest_date exactly at threshold_days is NOT stale (strict >)."""
    today = datetime(2026, 6, 11, 12, 0, 0)
    latest_date = today - timedelta(days=14)
    with patch.object(preflight, "_data_range",
                      side_effect=mock_data_range({glob: latest_date})), \
         patch_datetime(today):
        failures = check_in_season_stale(threshold_days=14)

    sport_fails = sport_failures(failures, sport)
    if sport in OFF_SEASON_ON_JUNE_11:
        # Off-season on June 11: off-season skip fires first
        assert len(sport_fails) == 0
    else:
        # 14 > 14 is False — not in failures
        assert len(sport_fails) == 0, (
            f"{sport}: exactly at threshold (14d) should NOT be in failures "
            f"(uses strict >), got {sport_fails}"
        )


@pytest.mark.parametrize("sport,glob", SPORT_GLOBS.items())
def test_parquet_sport_just_above_threshold_in_failures(
    sport, glob, sport_failures, mock_data_range, patch_datetime,
):
    """latest_date at threshold + 1 IS stale (the smallest stale value)."""
    today = datetime(2026, 6, 11, 12, 0, 0)
    latest_date = today - timedelta(days=15)
    with patch.object(preflight, "_data_range",
                      side_effect=mock_data_range({glob: latest_date})), \
         patch_datetime(today):
        failures = check_in_season_stale(threshold_days=14)

    sport_fails = sport_failures(failures, sport)
    if sport in OFF_SEASON_ON_JUNE_11:
        # Off-season on June 11: off-season skip fires first
        assert len(sport_fails) == 0
    else:
        assert len(sport_fails) == 1, (
            f"{sport}: 15d should be in failures (smallest stale value), "
            f"got {len(sport_fails)}: {failures}"
        )
        assert sport_fails[0]["age_days"] == 15
