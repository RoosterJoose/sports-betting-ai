#!/usr/bin/env python3
"""Pre-flight freshness report: per-sport data + model + cal status.

Designed to be called from `bin/morning_scan.sh` BEFORE the live scan
runs, so the user can see at-a-glance whether every sport's data and
models are 2026-current. Non-fatal — always exits 0 (the existing
`src.utils.staleness` module is the hard gate; this is the soft, visual
audit).

For each sport, prints:
  - Data cache file (mtime + date range)
  - Model files directory (mtime of newest file)
  - Cal files (mtime of newest file)
  - Staleness flag if data is > 7 days old during in-season

Sports covered: NBA, MLB, NHL, WNBA, UFC, WC (plus NFL, CFB if present).
"""
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# (sport, data_glob, data_date_col, model_dir, model_glob, cal_dir, cal_glob,
#  season_start_md, season_end_md, in_season_stale_days)
#
# `in_season_stale_days` is the per-sport staleness threshold (in days) used
# by _flag_for() to decide when an in-season sport should be downgraded to
# "OFF" because the data is too old (regular season effectively ended,
# e.g. NHL playoffs running through mid-June, NBA season ending in June).
#   - NBA 30: no playoffs gap; season ends and off-season starts quickly
#   - MLB 60: 162-game season, daily games; off-season starts in November
#   - NHL 90: regular season ends mid-April, playoffs run through mid-June
#             (long gap means the 60-day default would falsely flag NHL as
#             OFF in late April / early May)
#   - WNBA 60: regular season May-Sep, playoffs through October
#   - UFC 60: year-round events
#   - WC 60: single tournament, year-round coverage
#   - NFL 90: off-season from Feb to Sep is a 7-month gap
#   - CFB 90: off-season from Jan to Aug is a 7-month gap
SPORT_CONFIG = [
    ("NBA",  "data/nba_cache/game_logs_v14.parquet", "game_date",
     "models/nba", "*.json", "models/nba", "*_beta_cal.json", (10, 1), (6, 30), 30),
    ("MLB",  "data/cache/mlb/game_logs_*.parquet",   "game_date",
     "models/mlb", "lgb_*.txt", "models/mlb", "*_beta_cal.json", (3, 20), (11, 15), 60),
    ("NHL",  "data/nhl_cache/*.parquet",            "game_date",
     "models/nhl", "*.json", "models/nhl", "*_beta_cal.json", (10, 1), (6, 30), 90),
    ("WNBA", "data/wnba_cache/wnba_games.parquet",  "game_date",
     "models/wnba", "*.json", "models/wnba", "*_beta_cal.json", (5, 1), (10, 31), 60),
    ("UFC",  None, None,  # UFC has no parquet cache — uses fighter_augment.json
     "models/ufc", "*.json", "models/ufc", "*_beta_cal.json", (1, 1), (12, 31), 60),
    ("WC",   "data/cache/worldcup/all_matches.parquet", "match_date",
     "models/worldcup", "*.json", "models/worldcup", "offset_*.json", None, None, 60),
    ("NFL",  "data/nfl_cache/*.parquet",            "game_date",
     "models/nfl", "*.json", "models/nfl", "*_beta_cal.json", (9, 1), (2, 28), 90),
    ("CFB",  "data/cfb_cache/*.parquet",            "game_date",
     "models/cfb", "*.json", "models/cfb", "*_beta_cal.json", (8, 15), (1, 31), 90),
]


def _latest_mtime(paths: list[Path]) -> datetime | None:
    """Return the most recent mtime across a list of files, or None."""
    if not paths:
        return None
    return max(p.stat().st_mtime for p in paths)


def _fmt_mtime(ts: float | None) -> str:
    if ts is None:
        return "—"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def _days_ago(ts: float | None) -> int | None:
    if ts is None:
        return None
    return (datetime.now() - datetime.fromtimestamp(ts)).days


def _data_range(glob: str, date_col: str | None) -> tuple[str, str, int, datetime | None, float | None]:
    """Return (path, range_str, n_rows, latest_date, mtime) for a data glob."""
    if not glob:
        return ("(no cache)", "—", 0, None, None)
    try:
        candidates = sorted(PROJECT_ROOT.glob(glob))
    except (ValueError, OSError):
        return ("(bad glob)", "—", 0, None, None)
    if not candidates:
        return ("(missing)", "—", 0, None, None)
    # Pick the candidate with the LATEST data (max date_col value), not the
    # most recently modified file. Falls back to largest file size, then
    # to most recent mtime. This avoids picking stale `_pre_*.parquet`
    # archives when a current file exists with the same name pattern.
    best_p = None
    best_date = None
    for p in candidates:
        try:
            import pandas as pd
            df = pd.read_parquet(p, columns=[date_col] if date_col else None)
        except Exception:
            continue
        if df.empty or not date_col or date_col not in df.columns:
            continue
        s = pd.to_datetime(df[date_col], errors="coerce").dropna()
        if s.empty:
            continue
        latest = s.max()
        if best_date is None or latest > best_date:
            best_date = latest
            best_p = p
    if best_p is None:
        # Fallback: most recently modified, then largest
        best_p = max(candidates, key=lambda x: (x.stat().st_mtime, x.stat().st_size))
    try:
        import pandas as pd
        df = pd.read_parquet(best_p)
    except Exception as e:
        return (best_p.name, f"read-error: {e}", 0, None, best_p.stat().st_mtime)
    if df.empty:
        return (best_p.name, "empty", 0, None, best_p.stat().st_mtime)
    if date_col and date_col in df.columns:
        s = pd.to_datetime(df[date_col], errors="coerce").dropna()
        if s.empty:
            return (best_p.name, f"{len(df):,} rows, no dates", len(df), None, best_p.stat().st_mtime)
        return (best_p.name, f"{s.min().date()} -> {s.max().date()}", len(df),
                s.max().to_pydatetime(), best_p.stat().st_mtime)
    return (best_p.name, f"{len(df):,} rows", len(df), None, best_p.stat().st_mtime)


# Single source of truth for the UFC data file path. Both _ufc_age_days()
# and _ufc_data() reference this constant, and the two helpers are the
# only places in the module that need to know the path — report() and
# check_in_season_stale() never construct it themselves.
_UFC_AUG_PATH = PROJECT_ROOT / "models/ufc/fighter_augment.json"


def _ufc_age_days() -> int | None:
    """Return the mtime-based age of models/ufc/fighter_augment.json in days.

    UFC has no parquet cache, so the staleness metric is the file's mtime
    (refreshed after each UFC event). Returns None if the file is missing.

    Mirrors the parquet-based staleness check that _data_range() enables
    for the other 7 sports — same caller pattern in both report() and
    check_in_season_stale(), so the path is constructed in exactly one
    place (the _UFC_AUG_PATH constant above).
    """
    if not _UFC_AUG_PATH.exists():
        return None
    return _days_ago(_UFC_AUG_PATH.stat().st_mtime)


def _ufc_data() -> tuple[str, str, float | None, int | None]:
    """Return (name, range_str, mtime, age_days) for fighter_augment.json.

    Mirrors _data_range()'s shape for parquet-based sports but uses
    mtime as the staleness metric instead of latest_date. UFC has no
    parquet cache; the data file is models/ufc/fighter_augment.json
    (refreshed after each UFC event).

    Returns (file_name, f"mtime {_fmt_mtime(mtime)}", mtime, age_days)
    if the file exists, or ("(no data file)", "—", None, None) if missing.

    Note: computes `age_days` directly from the mtime it already read
    (instead of delegating to `_ufc_age_days()`) so the file is stat'd
    exactly once per call. `_ufc_age_days()` remains an independent
    helper for callers (like `check_in_season_stale()`) that only need
    the age.
    """
    if not _UFC_AUG_PATH.exists():
        return ("(no data file)", "—", None, None)
    mtime = _UFC_AUG_PATH.stat().st_mtime
    return (_UFC_AUG_PATH.name, f"mtime {_fmt_mtime(mtime)}", mtime, _days_ago(mtime))


def _in_season(today: datetime, start_md: tuple | None, end_md: tuple | None) -> bool:
    if start_md is None or end_md is None:
        return True  # year-round sport (UFC, WC)
    sm, sd = start_md
    em, ed = end_md
    today_ord = today.month * 100 + today.day
    start_ord = sm * 100 + sd
    end_ord = em * 100 + ed
    if start_ord <= end_ord:
        return start_ord <= today_ord <= end_ord
    return today_ord >= start_ord or today_ord <= end_ord


def _flag_for(sport: str, in_season: bool, age_days: int | None,
              latest_date: datetime | None, in_season_stale_days: int = 60) -> str:
    """Return a staleness flag string.

    Treats end-of-season + playoffs/out-of-window as off-season: if the
    data's latest date is more than `in_season_stale_days` old AND we are
    within the nominal season window, downgrade to OFF (the regular season
    has effectively ended, e.g. NHL regular season ends in mid-April but
    the window extends to June 30 for the playoffs — 90d threshold is
    needed so we don't falsely flag NHL as OFF in late April).

    All output strings are kept <= 20 chars to fit the Flag column.
    """
    if age_days is None:
        return "-"
    if in_season and age_days > in_season_stale_days:
        return f"OFF {age_days}d"  # season effectively ended
    if not in_season:
        return f"OFF {age_days}d" if age_days <= 180 else f"!! OFF {age_days}d"
    if age_days <= 1:
        return "OK"
    if age_days <= 7:
        return f"OK {age_days}d"
    if age_days <= 14:
        return f"WARN {age_days}d"
    return f"STALE {age_days}d"


def _bets_today() -> dict:
    """Return summary of trades logged to trade_tracker.db since 00:00 local time.

    Gives the morning scan a one-glance 'what happened yesterday/since midnight'
    alongside the data freshness table. Returns dict with keys:
      - count: total trades logged today
      - by_sport: {sport_name: count} breakdown
      - by_status: {status_name: count} breakdown (pending/won/lost)
      - pnl_today: sum of pnl for trades resolved today
      - error: str if DB missing or query failed, else None
    """
    db_path = PROJECT_ROOT / "data" / "trade_tracker.db"
    if not db_path.exists():
        return {"count": 0, "by_sport": {}, "by_status": {},
                "pnl_today": 0.0, "error": "no DB"}
    try:
        import sqlite3
        midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        with sqlite3.connect(str(db_path)) as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM trades WHERE timestamp >= ?", (midnight,))
            count = int(cur.fetchone()[0] or 0)
            cur.execute(
                "SELECT sport, COUNT(*) FROM trades WHERE timestamp >= ? GROUP BY sport",
                (midnight,),
            )
            by_sport = {s: int(n) for s, n in cur.fetchall()}
            cur.execute(
                "SELECT status, COUNT(*) FROM trades WHERE timestamp >= ? GROUP BY status",
                (midnight,),
            )
            by_status = {s: int(n) for s, n in cur.fetchall()}
            cur.execute(
                "SELECT COALESCE(SUM(pnl), 0) FROM trades "
                "WHERE timestamp >= ? AND status IN ('won','lost')",
                (midnight,),
            )
            pnl_today = float(cur.fetchone()[0] or 0.0)
        return {
            "count": count, "by_sport": by_sport, "by_status": by_status,
            "pnl_today": pnl_today, "error": None,
        }
    except Exception as e:
        return {"count": 0, "by_sport": {}, "by_status": {},
                "pnl_today": 0.0, "error": str(e)}


def check_in_season_stale(threshold_days: int = 14) -> list:
    """Return sports that should fail the pre-commit hook (in-season + STALE > threshold).

    Used by scripts/hooks/pre-commit as a second guard (the existing
    src.utils.staleness gate has a 7d threshold; this one is a wider
    14d net that uses the preflight per-sport threshold for the
    "in-season" determination but the caller-supplied threshold for
    the actual "too stale to commit" decision).

    Returns a list of dicts: {sport, age_days, flag, in_season, latest_date}.
    Empty list means all clear. Always returns; never raises.

    Data sources per sport:
      - NBA/MLB/NHL/WNBA/NFL/CFB/WC: parquet cache via _data_range() (latest
        game date is the staleness metric).
      - UFC: models/ufc/fighter_augment.json mtime. UFC has no parquet and
        is year-round (season_start=None, season_end=None), so the in-window
        gate always passes; the threshold is applied to the fighter-augment
        mtime (refreshed after each UFC event).
    """
    today = datetime.now()
    failures = []
    for (sport, data_glob, data_date_col, model_dir, model_glob,
         cal_dir, cal_glob, season_start, season_end, in_season_stale_days) in SPORT_CONFIG:
        # Determine in-window (year-round sports like UFC always pass)
        in_window = _in_season(today, season_start, season_end)
        if not in_window:
            continue  # Off-season — pre-commit should not block commits for off-season sports
        # Compute age_days from the data source for this sport.
        latest_date_str = None
        mtime_str = None
        if data_glob is None:
            # UFC special-case: no parquet; use fighter_augment.json mtime.
            # _ufc_data() returns (name, range_str, mtime, age_days) so we
            # can also surface the mtime date in the failure dict (for the
            # pre-commit hook's "mtime YYYY-MM-DD" display).
            _, _, mtime, age_days = _ufc_data()
            if mtime is not None:
                mtime_str = str(datetime.fromtimestamp(mtime).date())
        else:
            # Reuse _data_range() — it already does the same candidate-selection
            # (max date_col value across glob candidates) and returns latest_date
            # as a Python datetime. Saves ~20 lines of duplicated parquet logic
            # and ensures the failure check uses the exact same selection as
            # the report() output.
            _, _, _, latest_date, _ = _data_range(data_glob, data_date_col)
            if latest_date is None:
                continue
            age_days = (today.date() - latest_date.date()).days
            latest_date_str = str(latest_date.date())
        if age_days is None or age_days <= threshold_days:
            continue
        failures.append({
            "sport": sport,
            "age_days": age_days,
            "flag": f"STALE {age_days}d",
            "in_season": in_window,
            "latest_date": latest_date_str,  # Date str for parquet; None for UFC
            "mtime": mtime_str,  # Date str for UFC; None for parquet
        })
    return failures


def report() -> int:
    today = datetime.now()
    print()
    print("  +-- PRE-FLIGHT: per-sport data + model + cal freshness -------------")
    print(f"  |  Today: {today:%Y-%m-%d %H:%M}")
    print("  |")
    header = (f"  |  {'Sport':<5} {'Data range':<25} {'Data mtime':<17} "
              f"{'Model mtime':<17} {'Cal mtime':<17} {'Flag':<15}")
    print(header)
    print("  |  " + "-" * (len(header) - 6))

    for (sport, data_glob, data_date_col, model_dir, model_glob,
         cal_dir, cal_glob, season_start, season_end, in_season_stale_days) in SPORT_CONFIG:
        # Data
        if data_glob is None:
            # UFC special-case: no parquet; use fighter_augment.json mtime.
            # Same helper as check_in_season_stale() — the path is
            # constructed in exactly one place.
            data_name, data_range_str, data_mtime_ts, data_age = _ufc_data()
            latest_date = None  # UFC has no date column
        else:
            data_name, data_range_str, n_rows, latest_date, data_mtime_ts = _data_range(
                data_glob, data_date_col)
            data_age = (today.date() - latest_date.date()).days if latest_date else None

        # Models
        model_files = list((PROJECT_ROOT / model_dir).glob(model_glob)) if (PROJECT_ROOT / model_dir).exists() else []
        # Exclude meta/importance/cal files
        model_files = [p for p in model_files if "_calibration_diag" not in p.name
                        and "_importance" not in p.name
                        and "metrics.json" not in p.name
                        and "feature_meta" not in p.name]
        model_mtime = _latest_mtime(model_files) if model_files else None

        # Calibrations
        cal_files = list((PROJECT_ROOT / cal_dir).glob(cal_glob)) if (PROJECT_ROOT / cal_dir).exists() else []
        cal_mtime = _latest_mtime(cal_files) if cal_files else None

        in_window = _in_season(today, season_start, season_end)
        flag = _flag_for(sport, in_window, data_age, latest_date, in_season_stale_days)

        # Truncate data_range_str if too long for the column
        if len(data_range_str) > 25:
            data_range_str = data_range_str[:22] + "..."

        print(f"  |  {sport:<5} {data_range_str:<25} "
              f"{_fmt_mtime(data_mtime_ts):<17} "
              f"{_fmt_mtime(model_mtime):<17} "
              f"{_fmt_mtime(cal_mtime):<17} "
              f"{flag:<15}")

    print("  |")
    bets = _bets_today()
    if bets.get("error") == "no DB":
        print("  |  Bets placed today: (no trade_tracker.db yet)")
    elif bets.get("error"):
        print(f"  |  Bets placed today: (query failed: {bets['error']})")
    else:
        sport_parts = [f"{s} {n}" for s, n in sorted(bets["by_sport"].items())]
        sport_breakdown = " / ".join(sport_parts) if sport_parts else "(no sports)"
        won = bets["by_status"].get("won", 0)
        lost = bets["by_status"].get("lost", 0)
        pnl = bets["pnl_today"]
        print(f"  |  Bets placed today: {bets['count']:>4d}  ({sport_breakdown})  |  "
              f"Resolved: W{won} / L{lost}  PNL: ${pnl:+.2f}")
    print("  |")
    print("  |  Legend: OK=fresh, OK Nd=<7d, WARN Nd=7-14d, STALE Nd=>14d, OFF=off-season")
    print("  +---------------------------------------------------------------------")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(report())
