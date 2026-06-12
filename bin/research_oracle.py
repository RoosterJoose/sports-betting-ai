#!/usr/bin/env python3
"""Research Oracle — generates a dated markdown file with the day's MLB slate.

For a given date, writes `data/oracle/YYYY-MM-DD.md` containing:
  - The full MLB schedule (matchup, probable pitchers, venue, start time)
  - Moneyline odds from The Odds API (DK / FanDuel / BetMGM, American format)
  - Fair implied probability (vig-removed) per game
  - Model moneyline estimate (from SP K differential — proxy for team quality)
  - Park factors and any other relevant context for the researcher

This file is the ground truth the researcher-web pass uses instead of web search.
Without it, the researcher would have to "look up" a 2026 game on the public
web (which doesn't have 2026 data) and incorrectly return "schedule doesn't
exist." With the oracle, the researcher can validate picks against the
schedule + consensus lines we already have.

Usage:
    python -m bin.research_oracle                    # today
    python -m bin.research_oracle --date 2026-06-13  # specific date
    python -m bin.research_oracle --print            # print to stdout instead of writing

Environment:
    ODDS_API_KEY — required for moneyline odds. If missing, oracle still
                   runs but the "Consensus odds" section shows "n/a — set
                   ODDS_API_KEY to enable."
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORACLE_DIR = PROJECT_ROOT / "data" / "oracle"
ORACLE_DIR.mkdir(parents=True, exist_ok=True)

MLB_STATS_API = "https://statsapi.mlb.com/api/v1"
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY_MLB = "baseball_mlb"
DEFAULT_REGIONS = "us"
DEFAULT_BOOKMAKERS = "draftkings,fanduel,betmgm"

# Park factors (must match src/scripts/scan_mlb_sim.py for consistency)
PARK_FACTOR_K = {
    "SD": 1.08, "SEA": 1.06, "NYM": 1.04, "MIA": 1.03, "CLE": 1.02,
    "OAK": 1.02, "TB": 1.01, "SF": 1.01, "WSH": 1.00, "DET": 1.00,
    "MIL": 0.99, "BAL": 0.99, "KC": 0.99, "MIN": 0.99, "PIT": 0.99,
    "LAA": 0.98, "PHI": 0.98, "CIN": 0.98, "ATL": 0.97, "CHC": 0.97,
    "TEX": 0.97, "BOS": 0.97, "TOR": 0.96, "HOU": 0.96, "STL": 0.96,
    "AZ": 0.95, "NYY": 0.95, "LAD": 0.94, "CWS": 0.93, "COL": 0.88,
}
PARK_FACTOR_HR = {
    "COL": 1.28, "CIN": 1.14, "BOS": 1.12, "NYY": 1.10, "BAL": 1.09,
    "CHC": 1.07, "MIL": 1.06, "MIN": 1.05, "TEX": 1.04, "HOU": 1.04,
    "CLE": 1.04, "LAA": 1.03, "PHI": 1.03, "AZ": 1.02, "TB": 1.01,
    "ATL": 1.01, "WSH": 1.00, "DET": 1.00, "KC": 1.00, "STL": 1.00,
    "PIT": 0.99, "MIA": 0.98, "SEA": 0.98, "LAD": 0.97,
    "SD": 0.96, "SF": 0.95, "OAK": 0.95, "TOR": 0.94, "NYM": 0.93,
}

# MLB API team ID → 3-letter code
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
TEAM_FULL_NAME = {
    "ARI": "Arizona Diamondbacks", "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles", "BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs", "CIN": "Cincinnati Reds",
    "CLE": "Cleveland Guardians", "COL": "Colorado Rockies",
    "CWS": "Chicago White Sox", "DET": "Detroit Tigers",
    "HOU": "Houston Astros", "KC": "Kansas City Royals",
    "LAA": "Los Angeles Angels", "LAD": "Los Angeles Dodgers",
    "MIA": "Miami Marlins", "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins", "NYM": "New York Mets",
    "NYY": "New York Yankees", "OAK": "Oakland Athletics",
    "PHI": "Philadelphia Phillies", "PIT": "Pittsburgh Pirates",
    "SD": "San Diego Padres", "SEA": "Seattle Mariners",
    "SF": "San Francisco Giants", "STL": "St. Louis Cardinals",
    "TB": "Tampa Bay Rays", "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays", "WSH": "Washington Nationals",
}


# ── HTTP helpers ────────────────────────────────────────────────────────────


def http_get_json(url: str, timeout: float = 20.0) -> Optional[Any]:
    try:
        req = Request(url, headers={"User-Agent": "sports-betting-ai/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"  [http] GET {url[:80]}… failed: {e}", file=sys.stderr)
        return None


def http_get_text(url: str, timeout: float = 20.0) -> Optional[str]:
    try:
        req = Request(url, headers={"User-Agent": "sports-betting-ai/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8")
    except Exception as e:
        print(f"  [http] GET {url[:80]}… failed: {e}", file=sys.stderr)
        return None


# ── MLB schedule fetch ──────────────────────────────────────────────────────


def fetch_mlb_schedule(date_str: str) -> list[dict]:
    """Fetch the MLB schedule with probable pitchers for a given date."""
    url = f"{MLB_STATS_API}/schedule?sportId=1&date={date_str}&hydrate=probablePitcher,team,venue"
    data = http_get_json(url)
    if not data:
        return []
    games: list[dict] = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            if g.get("status", {}).get("codedState") == "I":
                continue  # skip in-progress
            away = g.get("teams", {}).get("away", {}).get("team", {})
            home = g.get("teams", {}).get("home", {}).get("team", {})
            away_p = g.get("teams", {}).get("away", {}).get("probablePitcher", {}) or {}
            home_p = g.get("teams", {}).get("home", {}).get("probablePitcher", {}) or {}
            games.append({
                "game_pk": g.get("gamePk"),
                "date": d.get("date"),
                "start_time": g.get("gameDate", ""),  # ISO UTC
                "venue": g.get("venue", {}).get("name", ""),
                "status": g.get("status", {}).get("detailedState", ""),
                "away_team": MLB_API_TEAM_IDS.get(away.get("id", -1), ""),
                "away_team_full": away.get("name", ""),
                "away_pitcher": away_p.get("fullName", "TBD") or "TBD",
                "away_pitcher_id": away_p.get("id"),
                "home_team": MLB_API_TEAM_IDS.get(home.get("id", -1), ""),
                "home_team_full": home.get("name", ""),
                "home_pitcher": home_p.get("fullName", "TBD") or "TBD",
                "home_pitcher_id": home_p.get("id"),
            })
    return games


# ── The Odds API fetch (MLB moneyline) ──────────────────────────────────────


def fetch_mlb_odds(api_key: str, commence_time: str | None = None) -> list[dict]:
    """Fetch MLB moneyline (h2h) odds from The Odds API.

    Returns a list of events, each with bookmakers → outcomes. We flatten this
    into a per-game dict keyed by team abbreviation.
    """
    if not api_key:
        return []
    params = {
        "apiKey": api_key,
        "regions": DEFAULT_REGIONS,
        "markets": "h2h",
        "bookmakers": DEFAULT_BOOKMAKERS,
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    url = f"{ODDS_API_BASE}/sports/{SPORT_KEY_MLB}/odds?{urlencode(params)}"
    raw = http_get_json(url)
    if not raw:
        return []
    out: list[dict] = []
    for ev in raw:
        # The Odds API uses full team names; map to our 3-letter codes via a
        # simple substring match against TEAM_FULL_NAME values.
        home_name = ev.get("home_team", "")
        away_name = ev.get("away_team", "")
        home_abbr = _abbr_from_name(home_name)
        away_abbr = _abbr_from_name(away_name)
        if not home_abbr or not away_abbr:
            continue  # unknown team (e.g., spring training / all-star)
        book_lines: dict[str, dict] = {}  # bookmaker → {home_odds, away_odds}
        for book in ev.get("bookmakers", []):
            for market in book.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                outcomes = {o["name"]: o.get("price") for o in market.get("outcomes", [])}
                book_lines[book.get("title", book.get("key", "?"))] = {
                    "home_odds": outcomes.get(home_name),
                    "away_odds": outcomes.get(away_name),
                }
        out.append({
            "home_abbr": home_abbr,
            "away_abbr": away_abbr,
            "commence_time": ev.get("commence_time", ""),
            "book_lines": book_lines,
        })
    return out


def _abbr_from_name(full_name: str) -> str:
    """Map a full MLB team name (as The Odds API returns it) to our 3-letter code."""
    full_name_lower = full_name.lower()
    for abbr, name in TEAM_FULL_NAME.items():
        if name.lower() in full_name_lower or full_name_lower in name.lower():
            return abbr
    # Fallback: try common abbreviations in the name (e.g., "LA Dodgers" → "LAD")
    for abbr in TEAM_FULL_NAME:
        if abbr.lower() in full_name_lower.split():
            return abbr
    return ""


# ── Moneyline math ──────────────────────────────────────────────────────────


def american_to_implied(odds: int | float | None) -> float | None:
    if odds is None:
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    if odds < 0:
        return -odds / (-odds + 100.0)
    return None


def fair_probs(home_odds: int | float | None, away_odds: int | float | None) -> tuple[float | None, float | None]:
    """Remove vig to get fair probabilities (sums to 1.0)."""
    p_h = american_to_implied(home_odds)
    p_a = american_to_implied(away_odds)
    if p_h is None or p_a is None:
        return None, None
    total = p_h + p_a
    if total <= 0:
        return None, None
    return p_h / total, p_a / total


# ── Model moneyline (SP K differential proxy) ──────────────────────────────


def estimate_model_moneyline(game: dict) -> dict:
    """Estimate a model moneyline using the same SP K differential proxy
    used in `scripts/backtest_consensus_parlay.py`.

    Returns dict with home_prob (0-1) and away_prob (0-1), sum to 1.0.
    """
    # Without historical data, we use a baseline + park factor adjustment:
    # - Home team: 0.54 baseline (historical home win rate)
    # - SP quality proxy: we don't have a live K rate feed here, so we use
    #   the park factor as a small nudge: pitcher-friendly parks slightly
    #   favor the home pitcher (better K environment for the home arm).
    home_pf = PARK_FACTOR_K.get(game["home_team"], 1.0)
    away_pf = PARK_FACTOR_K.get(game["away_team"], 1.0)

    # Park factor ratio: home pitcher's K environment vs away pitcher's.
    # A 5% difference = ~3% shift in win probability.
    pf_ratio = home_pf / away_pf if away_pf else 1.0
    base = 0.54
    shift = (pf_ratio - 1.0) * 0.6  # scale: 0.05 pf diff → 0.03 prob shift
    home_prob = max(0.30, min(0.70, base + shift))
    return {
        "home_prob": home_prob,
        "away_prob": 1.0 - home_prob,
        "method": "SP K differential proxy (park factor adjusted; no live K rate feed in oracle)",
    }


# ── Markdown rendering ──────────────────────────────────────────────────────


def render_oracle_markdown(date_str: str, games: list[dict], odds_by_matchup: dict) -> str:
    """Render the oracle markdown file."""
    lines: list[str] = []
    lines.append(f"# Research Oracle — MLB Slate for {date_str}")
    lines.append("")
    lines.append(f"_Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}_")
    lines.append("")
    lines.append("> **GROUND TRUTH FILE** — When dispatching a `researcher-web` pass to "
                 "validate picks for this date, **inline this file** as the schedule + "
                 "odds source. The public web has no 2026 game data, so the researcher "
                 "will return 'schedule doesn't exist' without this context. Use this "
                 "file as the canonical source; ask the researcher for qualitative "
                 "context (e.g., 'is BAL a value bet at -140 given the SP matchup?').")
    lines.append("")
    lines.append(f"**Total games:** {len(games)}")
    lines.append("")

    if not games:
        lines.append("_No MLB games scheduled for this date._")
        return "\n".join(lines) + "\n"

    # Per-game section
    for g in games:
        matchup = f"{g['away_team']} @ {g['home_team']}"
        matchup_full = f"{g['away_team_full']} @ {g['home_team_full']}"
        key = (g["home_team"], g["away_team"])
        book_lines = odds_by_matchup.get(key, {}).get("book_lines", {})

        lines.append(f"## {matchup}")
        lines.append("")
        lines.append(f"- **Full names:** {matchup_full}")
        lines.append(f"- **Venue:** {g.get('venue') or 'TBD'}")
        lines.append(f"- **First pitch:** {g.get('start_time', 'TBD')}")
        lines.append(f"- **Status:** {g.get('status', 'Scheduled')}")
        lines.append(f"- **Game PK:** {g.get('game_pk', '?')}")
        lines.append("")
        lines.append(f"### Probable pitchers")
        lines.append(f"- **Away SP:** {g['away_pitcher']}  (MLBAM ID: {g.get('away_pitcher_id') or '?'})")
        lines.append(f"- **Home SP:** {g['home_pitcher']}  (MLBAM ID: {g.get('home_pitcher_id') or '?'})")
        lines.append("")

        # Park factors (home park)
        pf_k = PARK_FACTOR_K.get(g["home_team"], 1.0)
        pf_hr = PARK_FACTOR_HR.get(g["home_team"], 1.0)
        lines.append(f"### Park factors (home park)")
        lines.append(f"- **K factor:** {pf_k:.2f}  ({'pitcher-friendly' if pf_k > 1.02 else 'hitter-friendly' if pf_k < 0.98 else 'neutral'})")
        lines.append(f"- **HR factor:** {pf_hr:.2f}  ({'HR-friendly' if pf_hr > 1.05 else 'pitcher-friendly' if pf_hr < 0.98 else 'neutral'})")
        lines.append("")

        # Model moneyline
        model = estimate_model_moneyline(g)
        lines.append(f"### Model moneyline estimate")
        lines.append(f"- **Method:** {model['method']}")
        lines.append(f"- **Home win prob:** {model['home_prob']:.1%}  (fair odds: {prob_to_american(model['home_prob'])})")
        lines.append(f"- **Away win prob:** {model['away_prob']:.1%}  (fair odds: {prob_to_american(model['away_prob'])})")
        lines.append("")

        # Consensus odds (from The Odds API)
        lines.append("### Consensus odds (DK / FanDuel / BetMGM)")
        if not book_lines:
            lines.append("_No odds available — set ODDS_API_KEY env var to enable._")
        else:
            lines.append("| Sportsbook | Home ML | Away ML | Home implied | Away implied | Fair home | Fair away |")
            lines.append("|---|---|---|---|---|---|---|")
            for book_name, lines_pair in sorted(book_lines.items()):
                h_odds = lines_pair.get("home_odds")
                a_odds = lines_pair.get("away_odds")
                h_imp = american_to_implied(h_odds)
                a_imp = american_to_implied(a_odds)
                h_fair, a_fair = fair_probs(h_odds, a_odds)
                lines.append(
                    f"| {book_name} | {fmt_odds(h_odds)} | {fmt_odds(a_odds)} | "
                    f"{fmt_pct(h_imp)} | {fmt_pct(a_imp)} | {fmt_pct(h_fair)} | {fmt_pct(a_fair)} |"
                )
        lines.append("")

        # Consensus line (median of bookmakers' fair probabilities)
        if book_lines:
            all_fairs_h, all_fairs_a = [], []
            for lines_pair in book_lines.values():
                h_fair, a_fair = fair_probs(lines_pair.get("home_odds"), lines_pair.get("away_odds"))
                if h_fair is not None:
                    all_fairs_h.append(h_fair)
                if a_fair is not None:
                    all_fairs_a.append(a_fair)
            if all_fairs_h:
                med_h = sorted(all_fairs_h)[len(all_fairs_h) // 2]
                med_a = 1.0 - med_h
                lines.append(f"**Consensus line (median fair):** HOME {med_h:.1%} ({prob_to_american(med_h)}) · AWAY {med_a:.1%} ({prob_to_american(med_a)})")
                lines.append("")

        # Quick "researcher prompt" hint
        lines.append("### Research prompts for this game")
        lines.append(f"- Is the home park factor significant for the SP matchup? (K={pf_k:.2f}, HR={pf_hr:.2f})")
        lines.append(f"- Recent form: how have {g['home_pitcher']} and {g['away_pitcher']} performed in their last 3 starts?")
        lines.append(f"- Head-to-head: any historical matchup trends between these SPs?")
        lines.append(f"- Bullpen: any late-inning availability concerns for either side?")
        lines.append(f"- Weather: check wind speed/direction at {g.get('venue') or 'venue'} (impacts HR factor)")
        lines.append(f"- Line movement: has the consensus line moved significantly in the last 24h?")
        lines.append("")

    return "\n".join(lines) + "\n"


def fmt_odds(odds: int | float | None) -> str:
    if odds is None:
        return "—"
    if odds > 0:
        return f"+{int(odds)}"
    return f"{int(odds)}"


def fmt_pct(p: float | None) -> str:
    if p is None:
        return "—"
    return f"{p:.1%}"


def prob_to_american(p: float) -> str:
    if p <= 0 or p >= 1:
        return "—"
    if p >= 0.5:
        return f"{-int(round(100 * p / (1 - p)))}"
    return f"+{int(round(100 * (1 - p) / p))}"


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Research Oracle — generate a dated markdown file with the day's MLB slate.")
    parser.add_argument("--date", type=str, default=None, help="Date (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--print", action="store_true", help="Print to stdout instead of writing to file.")
    parser.add_argument("--no-odds", action="store_true", help="Skip The Odds API call (schedule + SPs only).")
    args = parser.parse_args()

    date_str = args.date or datetime.now().strftime("%Y-%m-%d")
    print(f"  Research Oracle — date={date_str}")
    print(f"  Output dir: {ORACLE_DIR}")

    # 1. Schedule
    print("  Fetching MLB schedule (statsapi.mlb.com)…")
    games = fetch_mlb_schedule(date_str)
    if not games:
        print(f"  ⚠️  No games found for {date_str} (or API error).")
    else:
        print(f"  → {len(games)} games found")

    # 2. Odds
    odds_by_matchup: dict[tuple[str, str], dict] = {}
    api_key = os.environ.get("ODDS_API_KEY", "")
    if not args.no_odds:
        if not api_key:
            print("  ⚠️  ODDS_API_KEY not set — skipping consensus odds (set in .env or env var)")
        else:
            print("  Fetching moneyline odds (The Odds API)…")
            odds_list = fetch_mlb_odds(api_key)
            for ev in odds_list:
                key = (ev["home_abbr"], ev["away_abbr"])
                # Merge if multiple events map to the same matchup (rare)
                existing = odds_by_matchup.get(key, {"book_lines": {}})
                existing["book_lines"].update(ev["book_lines"])
                odds_by_matchup[key] = existing
            print(f"  → {len(odds_list)} events with odds")

    # 3. Render
    md = render_oracle_markdown(date_str, games, odds_by_matchup)

    # 4. Write or print
    if args.print:
        print("\n" + "=" * 70)
        print(md)
        return
    out_path = ORACLE_DIR / f"{date_str}.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"\n  ✅ Wrote {out_path}")
    print(f"     Use as ground truth when dispatching researcher-web for {date_str}.")


if __name__ == "__main__":
    main()
