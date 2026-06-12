"""Project-root conftest for pytest.

Makes the project root importable as a path so `from src.data.fotmob import …`
works in any test file without each test doing its own `sys.path.insert(0, …)`.

This is the canonical fix for "ModuleNotFoundError: No module named 'src'"
when running `python -m pytest` from the project root. With this conftest
plus the `testpaths = ["tests"]` entry in pyproject.toml, no test file
should ever need to mutate sys.path.

Also defines shared fixtures and constants for preflight tests so future
preflight test files don't have to duplicate them.
"""
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

# Project root = parent of the tests/ directory containing this conftest.py
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add project root to sys.path so `src.*` imports resolve cleanly
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def pytest_addoption(parser):
    """Register project-wide CLI flags here (must live in conftest, not test files,
    because pytest parses args BEFORE collecting test files)."""
    parser.addoption("--runlive", action="store_true", default=False,
                     help="Run live FotMob integration tests (requires network)")


# ─────────────────────────────────────────────────────────────────────
# Shared preflight test fixtures
#
# `check_in_season_stale()` in src/utils/preflight.py has two data-source
# branches: parquet (for NBA/MLB/NHL/WNBA/WC/NFL/CFB) and mtime-of-file
# (for UFC, which has no parquet cache — uses models/ufc/fighter_augment.json).
# The fixtures below isolate the UFC file side so tests can control whether
# the aug file "exists" without touching the real file on disk.
# ─────────────────────────────────────────────────────────────────────


# Substring used to match the UFC aug path inside `Path.exists` checks.
# Single source of truth — both the fixture and any test that needs to
# reference the path directly should import this from conftest.
UFC_AUG_SUFFIX = "models/ufc/fighter_augment.json"


@pytest.fixture
def patch_path_exists():
    """Factory fixture: returns a function that patches `Path.exists` so the
    UFC aug path reports a controlled value, while all other paths delegate
    to the real `Path.exists` (so parquet files for the other 7 sports use
    their real on-disk state).

    Usage in a test:
        def test_xxx(patch_path_exists):
            with patch_path_exists(ufc_exists=True):
                ...  # UFC aug is "present"
            with patch_path_exists(ufc_exists=False):
                ...  # UFC aug is "missing"

    The patch is automatically reverted when the `with` block exits.

    Note: `real_exists` is captured at PATCH time (inside `_patch`),
    not at fixture-resolution time, so a botched teardown in a
    previous test that left `Path.exists` patched cannot leak into
    this fixture's behavior.
    """
    def _patch(ufc_exists: bool):
        real_exists = Path.exists
        def _side_effect(self):
            if UFC_AUG_SUFFIX in str(self):
                return ufc_exists
            return real_exists(self)
        return patch.object(Path, "exists", _side_effect)

    return _patch


@pytest.fixture
def ufc_entries():
    """Filter fixture: returns a function that filters a `check_in_season_stale()`
    failures list to only the UFC entries.

    Usage in a test:
        def test_xxx(ufc_entries):
            failures = check_in_season_stale(threshold_days=14)
            ufc = ufc_entries(failures)
            assert len(ufc) == 1
    """
    def _filter(failures):
        return [f for f in failures if f["sport"] == "UFC"]
    return _filter


# Per-sport data glob lookup. Matches the data_glob column in
# src/utils/preflight.py:SPORT_CONFIG so the mock can match by glob.
# Used by `mock_data_range` (below) to know which glob to return the
# controlled latest_date for, and by tests that need to reference a
# specific sport's glob (e.g., the NFL-focused in-season-gate test).
SPORT_GLOBS = {
    "NBA":  "data/nba_cache/game_logs_v14.parquet",
    "MLB":  "data/cache/mlb/game_logs_*.parquet",
    "NHL":  "data/nhl_cache/*.parquet",
    "WNBA": "data/wnba_cache/wnba_games.parquet",
    "WC":   "data/cache/worldcup/all_matches.parquet",
    "NFL":  "data/nfl_cache/*.parquet",
    "CFB":  "data/cfb_cache/*.parquet",
}


# Off-season on June 11 (the "today" used by the per-sport threshold
# tests). Single source of truth — the per-sport threshold tests use
# this set to branch in-season vs off-season on the hardcoded date.
OFF_SEASON_ON_JUNE_11 = {"NFL", "CFB"}


@pytest.fixture
def sport_failures():
    """Filter fixture: returns a function that filters a
    `check_in_season_stale()` failures list to entries for one sport.

    Usage in a test:
        def test_xxx(sport_failures):
            failures = check_in_season_stale(threshold_days=14)
            nfl = sport_failures(failures, "NFL")
            assert len(nfl) == 1
    """
    def _filter(failures, sport):
        return [f for f in failures if f["sport"] == sport]
    return _filter


@pytest.fixture
def mock_data_range():
    """Factory fixture: returns a function that produces a `side_effect` for
    patching `preflight._data_range` with a controlled `latest_date` per glob.

    Globs not in the supplied map return `(missing, None)` so the test
    isolates the sport under test from all others.

    Usage in a test:
        def test_xxx(mock_data_range):
            latest = datetime(2026, 5, 12, 0, 0, 0)
            with patch.object(preflight, "_data_range",
                              side_effect=mock_data_range({glob: latest})):
                failures = check_in_season_stale(threshold_days=14)
    """
    def _factory(glob_to_latest_date):
        def _side_effect(glob, date_col):
            if glob in glob_to_latest_date:
                latest = glob_to_latest_date[glob]
                return ("file.parquet", f"-> {latest.date()}", 100, latest, 0.0)
            return ("(missing)", "—", 0, None, None)
        return _side_effect
    return _factory


@pytest.fixture
def patch_datetime():
    """Factory fixture: returns a context manager that patches
    `src.utils.preflight.datetime` so `now()` returns a controlled `today`,
    while keeping `fromtimestamp` working (UFC's `_ufc_data()` → `_days_ago()`
    path uses it when iterating all 8 sports in `check_in_season_stale()`).

    Usage in a test:
        def test_xxx(patch_datetime):
            today = datetime(2026, 6, 11, 12, 0, 0)
            with patch_datetime(today):
                failures = check_in_season_stale(threshold_days=14)

    Why `fromtimestamp` must stay real: without the override,
    `MagicMock.fromtimestamp()` returns a Mock, and
    `datetime.now() - datetime.fromtimestamp(ts)` (inside `_days_ago`)
    returns a Mock via MagicMock's `__sub__` instead of raising TypeError
    — which then cascades into `age_days` being a Mock and triggering
    `'<=' not supported between instances of 'MagicMock' and 'int'`
    at the threshold check.
    """
    @contextmanager
    def _patch(today):
        with patch("src.utils.preflight.datetime") as mock_dt:
            mock_dt.now.return_value = today
            mock_dt.fromtimestamp = datetime.fromtimestamp
            yield mock_dt
    return _patch

