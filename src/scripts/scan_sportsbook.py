#!/usr/bin/env python3
"""SharpAPI Sportsbook Odds Scanner — MLB game lines.

SharpAPI returns ONE entry per selection per sportsbook.  Each entry has
home_team / away_team, market_type, team_side, odds_probability, and line.
We re-group by game × sportsbook to present familiar tables.

Market-type normalization:
  moneyline  → moneyline
  run_line   → spread
  *total*    → totals (inning-specific like "1st_5_innings_total_runs")

Usage:
    python -m src.scripts.scan_sportsbook                    # all books
    python -m src.scripts.scan_sportsbook --book Pinnacle     # one book
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import re

warnings.filterwarnings("ignore")

# ── .env bootstrap ────────────────────────────────────────────────
_dotenv = Path(__file__).resolve().parents[2] / ".env"
if _dotenv.exists():
    for _line in _dotenv.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SHARPAPI_BASE = "https://api.sharpapi.io/api/v1"
SHARPAPI_KEY = os.environ.get("SHARPAPI_KEY")

# ── Helpers ────────────────────────────────────────────────────────

TEAM_ABBR = {
    "ari":        "AZ",   "diamondbacks": "AZ",
    "atl":        "ATL",  "braves":       "ATL",
    "bal":        "BAL",  "orioles":      "BAL",
    "bos":        "BOS",  "red sox":      "BOS",
    "chc":        "CHC",  "cubs":         "CHC",
    "chw":        "CWS",  "white sox":    "CWS",
    "cin":        "CIN",  "reds":         "CIN",
    "cle":        "CLE",  "guardians":    "CLE",
    "col":        "COL",  "rockies":      "COL",
    "det":        "DET",  "tigers":       "DET",
    "hou":        "HOU",  "astros":       "HOU",
    "kc":         "KC",   "royals":       "KC",
    "laa":        "LAA",  "angels":       "LAA",
    "lad":        "LAD",  "dodgers":      "LAD",
    "mia":        "MIA",  "marlins":      "MIA",
    "mil":        "MIL",  "brewers":      "MIL",
    "min":        "MIN",  "twins":        "MIN",
    "nym":        "NYM",  "mets":         "NYM",
    "nyy":        "NYY",  "yankees":      "NYY",
    "oak":        "OAK",  "athletics":    "OAK",
    "phi":        "PHI",  "phillies":     "PHI",
    "pit":        "PIT",  "pirates":      "PIT",
    "sd":         "SD",   "padres":       "SD",
    "sea":        "SEA",  "mariners":     "SEA",
    "sf":         "SF",   "giants":       "SF",
    "stl":        "STL",  "cardinals":    "STL",
    "tb":         "TB",   "rays":         "TB",
    "tex":        "TEX",  "rangers":      "TEX",
    "tor":        "TOR",  "blue jays":    "TOR",
    "wsh":        "WSH",  "nationals":    "WSH",
    "wsn":        "WSH",
}


def _team_abbr(team_name: str) -> str:
    """Map a full team name to 2-3 letter code."""
    t = team_name.strip().lower()
    # Try exact match on the full name
    for keyword, code in TEAM_ABBR.items():
        if keyword in t:
            return code
    # Fallback: first letters of city + mascot
    parts = t.replace("  ", " ").split(" ")
    if len(parts) >= 2:
        return (parts[0][:2] + parts[1][:1]).upper()
    return parts[0][:3].upper()


def _norm_market(market_type: str) -> str | None:
    """Normalize SharpAPI market_type into our internal key."""
    mt = (market_type or "").lower()
    if mt == "moneyline":
        return "moneyline"
    if mt == "run_line":
        return "spread"
    if "total" in mt:
        return "totals"
    return None


def _innings_label(market_type: str) -> str:
    """Extract inning label from a totals market type like '1st_5_innings_total_runs'."""
    m = re.search(r"(\d+)(?:st|nd|rd|th)_(\d+)_innings", market_type)
    if m:
        return f"({m.group(2)} Inn)"
    return ""


# ── Data fetching ─────────────────────────────────────────────────

def fetch_raw(market: str) -> list[dict]:
    """Fetch one market from SharpAPI, return list of entries."""
    if not SHARPAPI_KEY:
        print("  SHARPAPI_KEY not found in .env")
        return []
    import urllib.request
    url = f"{SHARPAPI_BASE}/odds?league=MLB&market={market}"
    req = urllib.request.Request(url, headers={"x-api-key": SHARPAPI_KEY})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.load(resp)
    except Exception as e:
        print(f"  {market}: fetch error — {e}")
        return []
    items = body if isinstance(body, list) else body.get("data", [])
    print(f"  {market}: {len(items)} selections")
    return items


def fetch_all_markets() -> list[dict]:
    """Fetch all markets."""
    all_entries: list[dict] = []
    for market in ("moneyline", "spread", "total"):
        all_entries.extend(fetch_raw(market=market))
    return all_entries


# ── Grouping ───────────────────────────────────────────────────────

def group_entries(entries: list[dict]) -> dict:
    """Group raw entries → game_key → sportsbook → market → side → info."""
    grouped: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for e in entries:
        home = e.get("home_team", "")
        away = e.get("away_team", "")
        if not home or not away:
            continue
        game_key = f"{away} @ {home}"

        sb = (e.get("sportsbook") or "").lower()
        raw_market = e.get("market_type", "")
        norm = _norm_market(raw_market)
        if norm is None:
            continue

        side = (e.get("team_side") or e.get("selection_type") or "").lower()
        base = {
            "selection": e.get("selection", ""),
            "odds_american": e.get("odds_american"),
            "odds_decimal": e.get("odds_decimal"),
            "probability": e.get("odds_probability"),
            "line": e.get("line"),
            "side": side,
            "innings": _innings_label(raw_market),
            "market_raw": raw_market,
        }
        grouped[game_key][sb][norm][side] = base

    return dict(grouped)


# ── Display ────────────────────────────────────────────────────────

def _fmt_odds(entry: dict | None, *, show_line: bool = False, always_line: bool = False) -> str:
    """Format odds + probability, optionally prepending the line value.

    If *show_line* is True and the entry has a non-null line, prepend it.
    If *always_line* is True, show the line even when null (for totals display).
    """
    if not entry:
        return "—"
    am = entry.get("odds_american")
    prob = entry.get("probability")
    line = entry.get("line")
    parts = []
    if show_line and line is not None:
        parts.append(str(line))
    elif always_line:
        parts.append("?")
    if am:
        parts.append(f"{am:+d}")
    if prob:
        parts.append(f"({prob:.0%})")
    return " ".join(parts) if parts else "—"


def display_games(grouped: dict, filter_book: str | None = None):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n{'='*105}")
    print(f"  MLB GAME LINES — {now}")
    print(f"  Source: SharpAPI{' → ' + filter_book if filter_book else ' (all books)'}")
    print(f"{'='*105}")

    for gk in sorted(grouped.keys()):
        books = grouped[gk]
        away_abbr = _team_abbr(gk.split(" @ ")[0])
        home_abbr = _team_abbr(gk.split(" @ ")[-1])
        teams_short = f"{away_abbr} @ {home_abbr}"

        sb_list = [filter_book.lower()] if filter_book else sorted(books.keys())
        has_pin = "pinnacle" in books
        header_shown = False

        for sb in sb_list:
            if sb not in books:
                continue
            mkt = books[sb]

            ml = mkt.get("moneyline", {})
            sp = mkt.get("spread", {})
            tt = mkt.get("totals", {})

            away_ml = ml.get("away", {})
            home_ml = ml.get("home", {})
            away_rl = sp.get("away", {})
            home_rl = sp.get("home", {})

            # Totals: check over/under
            over = tt.get("over", {})
            under = tt.get("under", {})
            inn_label = over.get("innings", "") or under.get("innings", "")

            over_str = _fmt_odds(over, show_line=True, always_line=True) if over else "—"
            under_str = _fmt_odds(under, show_line=True, always_line=True) if under else "—"

            if not header_shown:
                print(f"\n  {teams_short}")
                header_shown = True

            print(f"  {sb:12s} | ML: {away_abbr} {_fmt_odds(away_ml):>12s} | "
                  f"{home_abbr} {_fmt_odds(home_ml):>12s} | "
                  f"RL: {_fmt_odds(away_rl, show_line=True):>18s} {_fmt_odds(home_rl, show_line=True):>18s} | "
                  f"O/U {inn_label:>9s}: {over_str:>14s} {under_str:>14s}")

            # Show Pinnacle comparison for first non-Pinnacle book
            if not filter_book and sb != "pinnacle" and has_pin and sb == sb_list[0]:
                pm = books.get("pinnacle", {}).get("moneyline", {})
                pa = pm.get("away", {})
                ph = pm.get("home", {})
                if pa or ph:
                    print(f"  {'Pinnacle':12s} | ML: {away_abbr} {_fmt_odds(pa):>12s} | "
                          f"{home_abbr} {_fmt_odds(ph):>12s}")


# ── CLI ────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--book", type=str, default=None, help="Filter to one book")
    parser.add_argument("--raw", action="store_true", help="Dump raw JSON")
    args = parser.parse_args()

    if not SHARPAPI_KEY:
        print("Error: SHARPAPI_KEY not found in environment or .env")
        sys.exit(1)

    print("Fetching MLB odds from SharpAPI...")
    entries = fetch_all_markets()
    if not entries:
        print("No data received.")
        return

    groups = group_entries(entries)
    n_games = len(groups)
    n_books = len({sb for v in groups.values() for sb in v})
    print(f"\n  → {n_games} unique games across {n_books} sportsbooks")

    if args.raw:
        for gk in sorted(groups.keys())[:3]:
            sb = next(iter(groups[gk]))
            print(f"\n{gk} [{sb}]:")
            print(json.dumps(groups[gk][sb], indent=2, default=str))
    else:
        display_games(groups, filter_book=args.book)

    print(f"\n  Done at {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
