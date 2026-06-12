#!/usr/bin/env python3
"""Pre-populate the FotMob lineup cache for upcoming World Cup matches.

For each open KXWCGAME market on Kalshi, this job:
  1. Parses the ticker → match date + 3-letter team codes
  2. Looks up (or resolves) the FotMob matchId by date + team names
  3. Checks if the match is within the scrape window (60-90 min pre-kickoff)
  4. Scrapes the lineup via the fixed FotMobScraper + writes to cache

Idempotent: re-runs are safe; cached lineups within TTL are skipped.
Dry-run: pass --dry-run to skip all writes (logs only).
Cron: install via scripts/install-cron.sh (hourly during WC season).

Usage:
    python bin/populate_wc_lineups.py                # standard run
    python bin/populate_wc_lineups.py --dry-run      # log only, no writes
    python bin/populate_wc_lineups.py --verbose      # extra debug logging
    python bin/populate_wc_lineups.py --window 90    # 90 min pre-kickoff cutoff
"""
import argparse
import asyncio
import json
import logging
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the team-code map from scan_wc (3-letter → full country name)
from src.scripts.scan_wc import TICKER_TEAM_MAP  # noqa: E402
from src.data.kalshi import KalshiClient  # noqa: E402
from src.data.fotmob import (  # noqa: E402
    LineupCache, FOTMOB_ID_MAP_FILE, FotMobScraper, load_star_players,
)

# KXWCGAME-26JUN11MEXCUB-MEX → year=26, month=JUN, day=11, t1=MEX, t2=CUB, outcome=MEX
KALSHI_TICKER_RE = re.compile(
    r"KXWCGAME-(\d{2})([A-Z]{3})(\d{2})([A-Z]{3})([A-Z]{3})-([A-Z]+)"
)
MONTHS = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
          "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}

# Default: only scrape matches whose kickoff is 60-90 min away.
# (FotMob publishes lineups ~60 min before kickoff; 90 min gives us
# slack to retry on transient errors before the market locks.)
DEFAULT_WINDOW_MINUTES = 75  # midpoint of 60-90 window
DEFAULT_LOOKAHEAD_HOURS = 24  # only consider markets in next 24h

logger = logging.getLogger("populate_wc_lineups")


# ── Ticker parsing ────────────────────────────────────────────────────────

def parse_kxwcgame_ticker(ticker: str) -> Optional[dict]:
    """Parse KXWCGAME ticker into match info. Returns None if unparseable.

    Returns dict with: ticker, year, month, day, t1, t2, outcome,
                       home_code, away_code, home_full, away_full,
                       kickoff_utc (midnight UTC as initial estimate —
                       actual kickoff time comes from FotMob scrape).
    """
    m = KALSHI_TICKER_RE.match(ticker)
    if not m:
        return None
    yr_s, mon_s, day_s, t1, t2, outcome = m.groups()
    year = 2000 + int(yr_s)
    month = MONTHS.get(mon_s)
    if not month:
        return None
    try:
        day = int(day_s)
    except ValueError:
        return None

    # TICKER_TEAM_MAP is code -> full name; both teams are "home" / "away"
    # in the ticker (we just have code1 vs code2). The KXWCGAME convention
    # is t1=home, t2=away for the regular season.
    home_full = TICKER_TEAM_MAP.get(t1, t1)
    away_full = TICKER_TEAM_MAP.get(t2, t2)

    # Initial kickoff estimate = midnight UTC of match day.
    # Real kickoff comes from the FotMob scrape.
    kickoff_est = datetime(year, month, day, 0, 0, 0, tzinfo=timezone.utc)

    return {
        "ticker": ticker,
        "year": year, "month": month, "day": day,
        "home_code": t1, "away_code": t2,
        "home_full": home_full, "away_full": away_full,
        "outcome": outcome,
        "kickoff_est_utc": kickoff_est,
    }


# ── FotMob matchId resolution ─────────────────────────────────────────────

def load_fotmob_id_map() -> dict:
    if FOTMOB_ID_MAP_FILE.exists():
        try:
            return json.loads(FOTMOB_ID_MAP_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_fotmob_id_map(id_map: dict) -> None:
    FOTMOB_ID_MAP_FILE.parent.mkdir(parents=True, exist_ok=True)
    FOTMOB_ID_MAP_FILE.write_text(json.dumps(id_map, indent=2))


async def resolve_fotmob_match_id(date_str_yyyymmdd: str,
                                  home_full: str, away_full: str) -> Optional[int]:
    """Find a FotMob matchId by date + home/away full names.

    Returns the matchId, or None if not found / can't determine.
    Strategy: hit /api/data/matches?date=YYYYMMDD&... in a real browser
    (Cloudflare blocks curl), parse the JSON, match by team name.
    """
    from playwright.async_api import async_playwright
    url = (
        f"https://www.fotmob.com/api/data/matches?date={date_str_yyyymmdd}"
        f"&timezone=America%2FLos_Angeles&ccode3=USA"
    )
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
            ctx = await browser.new_context(user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0.0.0 Safari/537.36"
            ))
            page = await ctx.new_page()
            await page.goto("https://www.fotmob.com/", wait_until="domcontentloaded", timeout=20000)
            r = await page.goto(url, wait_until="domcontentloaded", timeout=20000)
            body = await page.content()
            await browser.close()
    except Exception as e:
        logger.debug(f"  Playwright fetch failed: {e}")
        return None

    m = re.search(r"<pre[^>]*>(.*?)</pre>", body, re.S)
    txt = m.group(1) if m else body
    if not txt.lstrip().startswith("{"):
        return None
    try:
        payload = json.loads(txt)
    except json.JSONDecodeError:
        return None

    home_lower = home_full.lower()
    away_lower = away_full.lower()
    for lg in payload.get("leagues", []):
        for match in lg.get("matches", []):
            h = match.get("home", {}).get("name", "")
            a = match.get("away", {}).get("name", "")
            # Match: both home and away names are contained (case-insensitive)
            if home_lower in h.lower() and away_lower in a.lower():
                return match.get("id")
            # Also try the reverse (in case FotMob swapped home/away)
            if away_lower in h.lower() and home_lower in a.lower():
                return match.get("id")
    return None


# ── Main pipeline ─────────────────────────────────────────────────────────

def process_match(parsed: dict, scraper: FotMobScraper, cache: LineupCache,
                   id_map: dict, now: datetime, window_minutes: int,
                   dry_run: bool, verbose: bool) -> tuple[str, Optional[int]]:
    """Process a single parsed KXWCGAME match. Returns (status, fotmob_id).

    This is a SYNC function because the underlying FotMobScraper uses the
    SYNC Playwright API. Sync Playwright uses greenlets internally and
    cannot switch threads (raises "Cannot switch to a different thread"
    on close()). So we keep the entire lifecycle — open + fetch + close
    — on the same thread by calling process_match via asyncio.to_thread
    from main_async.

    The id_map is mutated in place; we save it on every successful resolve
    so the next cron iteration doesn't re-query FotMob.
    """
    ticker = parsed["ticker"]
    home_full = parsed["home_full"]
    away_full = parsed["away_full"]
    match_date = parsed["kickoff_est_utc"].date()

    # 1. Skip if already cached + fresh (TTL is 6h by default)
    if cache.is_fresh(ticker):
        return "cached-fresh", None

    # 2. Resolve or look up FotMob matchId
    fotmob_id = id_map.get(ticker)
    if fotmob_id is None:
        date_str = parsed["kickoff_est_utc"].strftime("%Y%m%d")
        if verbose:
            logger.info(f"  {ticker}: resolving FotMob matchId for {date_str} {home_full} vs {away_full}")
        # NOTE: this is async (uses async_playwright) but we're in a sync
        # function. The HTTP fetch is independent of the FotMobScraper
        # lifecycle, so we run it via asyncio.run_coroutine_threadsafe or
        # simply do the resolve as part of a separate async step.
        # Simpler: leave this branch to the caller (main_async) and pass
        # in the resolved id. For now, return early.
        return "needs-resolve", None

    # 3. Scrape lineups + check kickoff time
    try:
        lineups = scraper.fetch_lineups(fotmob_id)
    except Exception as e:
        logger.debug(f"  {ticker}: scrape failed: {e}")
        return f"scrape-error", fotmob_id

    status = lineups.get("status", "")
    if status != "ok" or not (lineups.get("home") or lineups.get("away")):
        return f"no-lineup-published ({status})", fotmob_id

    # 4. Parse real kickoff time
    kickoff_str = lineups.get("kickoff")
    kickoff_dt = _parse_kickoff(kickoff_str)
    if kickoff_dt is None:
        return "unparseable-kickoff", fotmob_id

    # 5. Check time window: only cache if kickoff is within the scrape window
    minutes_to_kickoff = (kickoff_dt - now).total_seconds() / 60.0
    if minutes_to_kickoff > window_minutes:
        return f"too-far-out ({minutes_to_kickoff:.0f} min)", fotmob_id
    if minutes_to_kickoff < -120:
        return f"too-far-past ({-minutes_to_kickoff:.0f} min ago)", fotmob_id

    # 6. Cache it
    if not dry_run:
        cache.set(ticker, lineups)
    return f"cached ({minutes_to_kickoff:.0f} min to kickoff, {len(lineups.get('home',[]))}H + {len(lineups.get('away',[]))}A)", fotmob_id


def _parse_kickoff(s: Optional[str]) -> Optional[datetime]:
    """Parse FotMob kickoff string like 'Thu, Jun 11, 2026, 00:00 UTC'."""
    if not s:
        return None
    s = s.strip()
    # Common FotMob format: "Thu, Jun 11, 2026, 00:00 UTC"
    for fmt in ("%a, %b %d, %Y, %H:%M UTC", "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


async def main_async(args) -> int:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("=" * 70)
    logger.info(f"  WC lineup auto-population  |  window={args.window}min  |  "
                f"dry-run={args.dry_run}")
    logger.info("=" * 70)

    # 1. Fetch KXWCGAME markets
    kc = KalshiClient()
    try:
        wc_mkts = kc.list_markets(series_ticker="KXWCGAME", limit=500)
    except Exception as e:
        logger.error(f"Failed to fetch KXWCGAME markets: {e}")
        return 1
    if wc_mkts is None or wc_mkts.empty:
        logger.info("No KXWCGAME markets found. Exiting.")
        return 0
    logger.info(f"Fetched {len(wc_mkts)} KXWCGAME market rows")

    # 2. Group 3 outcomes → 1 match
    matches = {}  # match_key → {parsed, tickers: [home, tie, away]}
    for _, m in wc_mkts.iterrows():
        ticker = m.get("ticker", "")
        parsed = parse_kxwcgame_ticker(ticker)
        if not parsed:
            continue
        match_key = f"{parsed['home_code']}_{parsed['away_code']}_{parsed['kickoff_est_utc'].date()}"
        if match_key not in matches:
            matches[match_key] = {"parsed": parsed, "tickers": []}
        matches[match_key]["tickers"].append(ticker)

    logger.info(f"Unique matches: {len(matches)}")

    # 3. Filter to next 24h only
    now = datetime.now(timezone.utc)
    horizon = now + timedelta(hours=args.lookahead)
    eligible = []
    for mk, info in matches.items():
        kickoff_est = info["parsed"]["kickoff_est_utc"]
        if kickoff_est < now - timedelta(hours=3):
            continue  # already finished
        if kickoff_est > horizon:
            continue  # too far out
        eligible.append(info)

    logger.info(f"Within {args.lookahead}h horizon: {len(eligible)} matches")

    # 4. Process each
    cache = LineupCache()
    id_map = load_fotmob_id_map()
    logger.info(f"Cache: {len(cache.all())} existing entries  |  "
                f"ID map: {len(id_map)} entries")

    summary = {"cached-fresh": 0, "cached": 0, "no-fotmob-id": 0,
               "no-lineup-published": 0, "too-far-out": 0, "too-far-past": 0,
               "unparseable-kickoff": 0, "scrape-error": 0, "other": 0}

    # Two-phase pipeline:
    #   Phase 1 (async): resolve any missing FotMob matchIds via the live
    #                    /api/data/matches endpoint (uses async_playwright).
    #   Phase 2 (sync, in worker thread): for each match with a known id,
    #                    call process_match via asyncio.to_thread so the
    #                    entire sync-Playwright lifecycle (open + fetch +
    #                    close) stays on one thread. (Async Playwright
    #                    can't switch threads either, so this avoids both
    #                    the "Sync API inside asyncio loop" AND the
    #                    "Cannot switch to a different thread" greenlet
    #                    errors.)

    # ── Phase 1: async-resolve any missing FotMob matchIds ────────────────
    needs_resolve = []
    for info in eligible:
        parsed = info["parsed"]
        if id_map.get(parsed["ticker"]) is None:
            needs_resolve.append(parsed)
    if needs_resolve:
        logger.info(f"Resolving {len(needs_resolve)} missing FotMob matchIds (async)...")
        for parsed in needs_resolve:
            date_str = parsed["kickoff_est_utc"].strftime("%Y%m%d")
            if verbose := args.verbose:
                logger.info(f"  {parsed['ticker']}: resolving for {date_str} {parsed['home_full']} vs {parsed['away_full']}")
            mid = await resolve_fotmob_match_id(date_str, parsed["home_full"], parsed["away_full"])
            if mid is not None:
                id_map[parsed["ticker"]] = mid
                if not args.dry_run:
                    save_fotmob_id_map(id_map)
                logger.info(f"  {parsed['ticker']}: resolved → {mid}")
            else:
                logger.info(f"  {parsed['ticker']}: no FotMob matchId found for {date_str}")

    # ── Phase 2: sync-scrape (one thread for the whole FotMobScraper) ──
    # The scraper is created + used + closed all on a single thread via
    # asyncio.to_thread, which avoids both the sync-in-asyncio error AND
    # the greenlet cross-thread error.
    def _run_all_matches() -> dict:
        """Sync runner: opens scraper, calls process_match for each match,
        closes scraper. All on one thread."""
        results = {}
        with FotMobScraper() as scraper:
            for info in eligible:
                parsed = info["parsed"]
                try:
                    status, _ = process_match(
                        parsed, scraper, cache, id_map, now,
                        window_minutes=args.window,
                        dry_run=args.dry_run, verbose=args.verbose,
                    )
                except Exception as e:
                    logger.debug(f"  {parsed['ticker']}: process_match error: {e}")
                    status = "other"
                results[parsed["ticker"]] = (parsed, status)
        return results

    logger.info("Scraping + caching (sync, in worker thread)...")
    results = await asyncio.to_thread(_run_all_matches)

    for ticker, (parsed, status) in results.items():
        for key in summary:
            if status.startswith(key):
                summary[key] += 1
                break
        else:
            summary["other"] += 1
        logger.info(f"  {parsed['home_full']:>20s} vs {parsed['away_full']:<20s}  "
                    f"→ {status}")

    logger.info("")
    logger.info("=" * 70)
    logger.info("  SUMMARY")
    logger.info("=" * 70)
    for k, v in summary.items():
        logger.info(f"  {k:>22s}: {v}")
    logger.info(f"  Total: {sum(summary.values())} matches processed")
    if not args.dry_run:
        logger.info(f"  Cache: {len(cache.all())} entries  |  "
                    f"ID map: {len(id_map)} entries")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Log only, don't write to cache")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Extra debug logging")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW_MINUTES,
                        help=f"Max minutes-before-kickoff to cache (default: {DEFAULT_WINDOW_MINUTES})")
    parser.add_argument("--lookahead", type=int, default=DEFAULT_LOOKAHEAD_HOURS,
                        help=f"Lookahead window in hours (default: {DEFAULT_LOOKAHEAD_HOURS})")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
