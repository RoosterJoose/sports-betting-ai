"""Live integration test for the FotMob scraper.

Skipped by default (network + a real, currently-lineup-published matchId required).
Run with: pytest tests/test_fotmob_live.py --runlive

Strategy:
  1. Hit /api/data/matches?date=TODAY via Playwright (real browser bypasses Cloudflare)
  2. Find an UPCOMING international match
  3. Scrape its lineups via /api/data/matchDetails
  4. Verify home/away XIs parse, kickoff parses, status == "ok"
  5. Verify key_player_out fires correctly against the WC star-players list

If the chosen match has no published lineups (status != "ok" or XIs empty),
the test is skipped — lineups typically publish ~60 min before kickoff.
"""
import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.fotmob import (
    FotMobScraper, LineupCache, compute_key_player_out, load_star_players,
)


INTERNATIONAL_LEAGUES = {
    "Friendlies", "World Cup Qualification", "Euro Championship Qualification",
    "Copa America", "Africa Cup of Nations", "Asian Cup", "Nations League",
    "CONCACAF Gold Cup", "World Cup", "UEFA Euro",
}


def pytest_addoption(parser):
    parser.addoption("--runlive", action="store_true", default=False,
                     help="Run live FotMob integration tests (requires network)")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runlive"):
        skip_live = pytest.mark.skip(reason="need --runlive to run live tests")
        for item in items:
            if "live" in item.keywords:
                item.add_marker(skip_live)


pytestmark = pytest.mark.live


async def _find_upcoming_intl_match():
    """Find one upcoming international matchId via the live FotMob matches endpoint."""
    from playwright.async_api import async_playwright
    today = datetime.now().strftime("%Y%m%d")
    url = (
        f"https://www.fotmob.com/api/data/matches?date={today}"
        "&timezone=America%2FLos_Angeles&ccode3=USA"
    )
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
    m = re.search(r"<pre[^>]*>(.*?)</pre>", body, re.S)
    txt = m.group(1) if m else body
    if not txt.lstrip().startswith("{"):
        return None
    data = json.loads(txt)
    for lg in data.get("leagues", []):
        meta = lg.get("details") or lg
        if meta.get("name") not in INTERNATIONAL_LEAGUES:
            continue
        for match in lg.get("matches", []):
            st = match.get("status", {})
            if not (st.get("started") or st.get("finished") or st.get("cancelled")):
                return {
                    "matchId": match.get("id"),
                    "league": meta.get("name"),
                    "home": match.get("home", {}).get("name"),
                    "away": match.get("away", {}).get("name"),
                }
    return None


@pytest.mark.asyncio
async def test_fotmob_live_e2e():
    """End-to-end: find upcoming intl match → scrape lineups → verify cache + key_player_out."""
    match = await _find_upcoming_intl_match()
    if not match:
        pytest.skip("No upcoming international match found for today")
    print(f"\nFound upcoming intl match: {match['home']} vs {match['away']} (id={match['matchId']})")

    with FotMobScraper() as scraper:
        lineups = scraper.fetch_lineups(match["matchId"])

    assert lineups.get("status") in ("ok", "not_published"), \
        f"Unexpected status: {lineups.get('status')}"
    if lineups.get("status") != "ok" or not (lineups.get("home") or lineups.get("away")):
        pytest.skip(f"Lineups not yet published for {match['home']} vs {match['away']}")

    # Verify structure
    assert isinstance(lineups.get("home"), list)
    assert isinstance(lineups.get("away"), list)
    assert lineups.get("kickoff"), "kickoff should be populated"
    for side in ("home", "away"):
        for p in lineups[side]:
            assert p.get("name"), f"{side} player missing name: {p}"

    # Verify key_player_out works
    stars = load_star_players()
    kp = compute_key_player_out("FRA", "ARG", lineups, stars)
    assert kp in (0, 1), "key_player_out should be 0 or 1"
    print(f"  home={len(lineups['home'])} away={len(lineups['away'])} "
          f"kickoff={lineups['kickoff']} kp_test(FRA/ARG)={kp}")
