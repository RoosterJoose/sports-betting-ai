#!/usr/bin/env python3
"""Find real FotMob matchIds by hitting /api/matches?date=YYYY-MM-DD in a real browser.

curl returns Cloudflare-challenge HTML. Playwright (already in the project) runs a
real Chromium, so it can fetch the JSON endpoint successfully.

Usage:
    python bin/find_fotmob_matchid.py                    # today
    python bin/find_fotmob_matchid.py 2026-06-10         # specific date
    python bin/find_fotmob_matchid.py 2026-06-10 international  # filter intl only
"""
import sys, json, asyncio
from datetime import datetime, timedelta
from pathlib import Path

from playwright.async_api import async_playwright


def parse_date(s):
    if s:
        for fmt in ("%Y-%m-%d", "%Y%m%d"):
            try:
                return datetime.strptime(s, fmt).strftime("%Y%m%d")
            except ValueError:
                continue
        raise ValueError(f"bad date {s!r}")
    return (datetime.now() - timedelta(days=2)).strftime("%Y%m%d")


def extract_international_matches(payload, intl_only=True):
    """Walk the matches payload and return list of {matchId, home, away, league, has_lineup}."""
    out = []
    # The response is a dict keyed by league id; each value is a list of matches
    for league_id, league_block in payload.items():
        if not isinstance(league_block, dict):
            continue
        # League metadata in 'details' or top-level
        meta = league_block.get("details") or {}
        league_name = meta.get("name", league_id) if isinstance(meta, dict) else league_id
        country = meta.get("country", "") if isinstance(meta, dict) else ""
        matches = league_block.get("matches") or []
        for m in matches:
            if not isinstance(m, dict):
                continue
            mid = m.get("id") or m.get("matchId")
            if mid is None:
                continue
            home = m.get("home") or {}
            away = m.get("away") or {}
            home_name = home.get("name") or home.get("shortName") or "?"
            away_name = away.get("name") or away.get("shortName") or "?"
            # Status
            status = m.get("status") or {}
            finished = status.get("finished") if isinstance(status, dict) else None
            cancelled = status.get("cancelled") if isinstance(status, dict) else None
            started = status.get("started") if isinstance(status, dict) else None
            # Lineup availability: any of the lineup fields
            has_lineup = any(
                m.get(k) for k in ("lineup", "lineupData", "confirmedLineup",
                                    "homeLineup", "awayLineup")
            )
            # International = no club, or country != major domestic league
            is_intl = "int" in str(league_name).lower() or country in (
                "International", "World", "Europe", "South America", "Africa",
                "Asia", "Oceania", "North America",
            )
            if intl_only and not is_intl:
                continue
            out.append({
                "matchId": mid,
                "league": league_name,
                "country": country,
                "home": home_name,
                "away": away_name,
                "finished": finished,
                "started": started,
                "cancelled": cancelled,
                "has_lineup_hint": has_lineup,
            })
    return out


async def main():
    args = sys.argv[1:]
    date_str = parse_date(args[0] if args else None)
    intl_only = (args[1] if len(args) > 1 else "all").lower() != "all"
    # Default: intl-only for relevance to WC
    intl_only = (args[1] if len(args) > 1 else "international").lower() != "all"

    url = f"https://www.fotmob.com/api/matches?date={date_str}"
    print(f"[fetch] {url}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0.0.0 Safari/537.36"
            ),
            locale="en-US",
        )
        page = await context.new_page()
        # Navigate to the homepage first to seed cookies (avoids the bare-request 403)
        await page.goto("https://www.fotmob.com/", wait_until="domcontentloaded", timeout=20000)
        # Now the JSON endpoint
        resp = await page.goto(url, wait_until="domcontentloaded", timeout=20000)
        print(f"[http] status={resp.status if resp else 'no resp'}")
        body = await page.content()
        await browser.close()

    # Parse
    try:
        # body is HTML wrapper; extract <pre> JSON if Cloudflare returned a page
        import re
        m = re.search(r"<pre[^>]*>(.*?)</pre>", body, re.S)
        json_text = m.group(1) if m else body
        if json_text.lstrip().startswith("<!"):
            print(f"[parse] body is HTML, length={len(body)}; first 300 chars:")
            print(json_text[:300])
            return 2
        payload = json.loads(json_text)
    except json.JSONDecodeError as e:
        print(f"[parse] JSON decode failed: {e}")
        print(f"body first 500: {body[:500]}")
        return 3

    matches = extract_international_matches(payload, intl_only=intl_only)
    print(f"[extract] {len(matches)} {'intl' if intl_only else 'all'} matches on {date_str}")

    # Sort: started & not finished first (most likely to have lineups)
    matches.sort(key=lambda r: (
        not (r["started"] and not r["finished"] and not r["cancelled"]),
        r["league"], r["home"],
    ))

    out_path = Path(f"/tmp/fotmob_matches_{date_str}.json")
    out_path.write_text(json.dumps({
        "date": date_str,
        "url": url,
        "intl_only": intl_only,
        "n_matches": len(matches),
        "matches": matches,
    }, indent=2))
    print(f"[save] {out_path}")

    # Print top 30
    print(f"\nTop {min(30, len(matches))} matches:")
    print(f"{'matchId':>10}  {'league':<35} {'home':<25} vs {'away':<25}  status")
    print("-" * 120)
    for m in matches[:30]:
        status = "✓started" if m["started"] and not m["finished"] else \
                 "✓finished" if m["finished"] else \
                 "✗cancelled" if m["cancelled"] else "·not started"
        if m["has_lineup_hint"]:
            status += " [LINEUP]"
        print(f"{m['matchId']:>10}  {m['league'][:35]:<35} {m['home'][:25]:<25} vs "
              f"{m['away'][:25]:<25}  {status}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
