"""DraftKings public event page scraper for UFC prop markets.

DK's public event pages (e.g. https://sportsbook.draftkings.com/event/u-fight-XXXXX)
embed the full event data — including all prop markets (method of victory,
round of finish, fight goes to distance) — in a `<script id="__NEXT_DATA__">`
JSON blob. This module parses that blob and returns structured prop odds.

**No API key required** — DK's event pages are public.

**Brittleness warning**: DK's page structure (script tag id, JSON shape)
can change without notice. The parser is defensive: if the expected
structure isn't found, it returns an empty list rather than raising.
Add a `?wait_for=...` parameter or switch to Playwright if DK changes
their frontend.

**ToS note**: scraping public pages is generally OK, but be polite:
- Set a User-Agent
- Respect rate limits (default 1 request per second)
- Don't hammer the site

Usage:
    from src.data.dk_props_scraper import DKPropsScraper

    scraper = DKPropsScraper()
    props = scraper.get_event_props("https://sportsbook.draftkings.com/event/u-fight-12345")
    # props = [
    #     {
    #         "prop_type": "method_of_victory",
    #         "fighter": "Ilia Topuria",
    #         "outcome": "KO/TKO",
    #         "odds": +250,
    #         "sportsbook": "draftkings",
    #     },
    #     ...
    # ]
"""

import json
import re
import time
import warnings
from typing import Any
from urllib.request import Request, urlopen

warnings.filterwarnings("ignore")

# Public-facing DK event URL pattern. We don't try to enumerate event IDs —
# the caller passes the full URL (from search, from UFC schedule page, etc.).
DEFAULT_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) " \
                     "AppleWebKit/537.36 (KHTML, like Gecko) " \
                     "Chrome/120.0.0.0 Safari/537.36"

# Map DK market "label" strings to our canonical prop_type names.
# DK's labels vary by event; this is a best-effort mapping. Order matters:
# more specific phrases first so "method of victory" doesn't get caught by
# a generic "method" keyword.
_METHOD_OF_VICTORY_KEYWORDS = {
    "method of victory", "victory method", "how will the fight end",
    "decision - unanimous", "decision - split", "decision - majority",
    "ko/tko/dq", "ko/tko", "knockout",
    # Compound labels (DK often phrases as "Fighter to win by KO/TKO")
    "to win by ko/tko", "to win by submission", "to win by decision",
    "to win inside the distance", "to win by knockout",
    "win by ko/tko", "win by submission", "win by decision",
}
# Bare outcome names that should classify as method_of_victory via
# EXACT match (lowercased + stripped), not substring match. DK's event
# page layout often lists each MoV outcome (KO/TKO, Submission, Decision)
# as a separate market under a parent "Method of Victory" header, but
# the per-outcome label can be the bare name. The substring sets above
# intentionally omit "ko/tko", "submission", and "decision" alone
# because those words are too broad (e.g. "Decision" in a non-MoV
# context); exact-match is safer.
_METHOD_OF_VICTORY_EXACT_LABELS = {
    # KO/TKO variants
    "ko/tko", "ko/tko/dq", "knockout", "tko", "ko",
    # Submission variants
    "submission", "sub",
    # Decision variants
    "decision",
    "unanimous decision", "split decision", "majority decision",
    "decision - unanimous", "decision - split", "decision - majority",
    # Other fight outcomes (MoV-adjacent)
    "draw", "no contest", "nc",
}
_ROUND_OF_FINISH_KEYWORDS = {
    "round of finish", "round betting", "fight ends in round",
    "round 1", "round 2", "round 3", "round 4", "round 5",
    "goes the distance", "fight goes to distance", "distance",
}
_TOTAL_ROUNDS_KEYWORDS = {
    "total rounds", "over/under", "rounds over", "rounds under",
    "o/u", "total rounds o/u",
}


class DKScraperError(Exception):
    """Raised when the DK page can't be fetched or parsed."""


class DKPropsScraper:
    """Scraper for DK public event pages.

    No API key required. Uses urllib (no external deps). The scraper is
    stateless — each call fetches a fresh page. Cache externally if you
    need to poll frequently.
    """

    def __init__(
        self,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = 15.0,
        min_request_interval: float = 1.0,
    ):
        self.user_agent = user_agent
        self.timeout = timeout
        self.min_request_interval = min_request_interval
        self._last_request_time: float = 0.0

    def _fetch(self, url: str) -> str:
        """Fetch the URL with a polite delay between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        req = Request(url, headers={"User-Agent": self.user_agent})
        with urlopen(req, timeout=self.timeout) as resp:
            self._last_request_time = time.time()
            return resp.read().decode("utf-8", errors="replace")

    def get_event_props(self, event_url: str) -> list[dict]:
        """Fetch one DK event page and return all available prop markets.

        Returns a flat list of dicts, one per outcome:
            {
                "prop_type": "method_of_victory" | "round_of_finish" | "total_rounds" | "other",
                "fighter": "Ilia Topuria",    # fighter the prop is about (if applicable)
                "outcome": "KO/TKO",          # the specific outcome being priced
                "odds": 250,                  # American odds
                "sportsbook": "draftkings",
                "market_label": "Method of Victory",  # raw DK label
            }
        Returns [] on parse failure (defensive — DK's frontend can change).
        """
        try:
            html = self._fetch(event_url)
        except Exception as e:
            warnings.warn(f"DK fetch failed for {event_url}: {e}")
            return []

        blob = self._extract_next_data(html)
        if not blob:
            return []

        return self._parse_prop_markets(blob, source_url=event_url)

    def _extract_next_data(self, html: str) -> dict | None:
        """Extract the __NEXT_DATA__ JSON blob from a DK event page.

        DK embeds the full event data in a `<script id="__NEXT_DATA__">` tag.
        If the script tag isn't found, return None (defensive — DK's
        frontend can change).
        """
        # Pattern 1: standard __NEXT_DATA__ script tag
        m = re.search(
            r'<script\s+id="__NEXT_DATA__"\s+type="application/json">(.+?)</script>',
            html,
            re.DOTALL,
        )
        if not m:
            # Pattern 2: some DK pages use a different attribute order
            m = re.search(
                r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.+?)</script>',
                html,
                re.DOTALL,
            )
        if not m:
            return None
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            return None

    def _parse_prop_markets(self, blob: dict, source_url: str) -> list[dict]:
        """Walk the __NEXT_DATA__ blob and extract all UFC prop markets.

        The blob shape is deep and version-dependent. We use a defensive
        recursive search for "markets" / "outcomes" patterns rather than
        hardcoding paths.
        """
        results: list[dict] = []
        # The blob is {"props": {"pageProps": {...}}}-shaped on most DK pages.
        # Walk all "outcomes" arrays and pair with the enclosing market label.
        self._walk_for_markets(blob, results, current_label="", source_url=source_url)
        return results

    def _walk_for_markets(
        self,
        node: Any,
        results: list[dict],
        current_label: str,
        source_url: str,
    ) -> None:
        """Recursively walk the blob, tracking the current market label.

        A "market" is a dict that has a "label" (or "name" or "title") and
        an "outcomes" array. Each outcome in the array is a separate prop
        line we want to surface.
        """
        if isinstance(node, dict):
            label = (
                node.get("label")
                or node.get("name")
                or node.get("title")
                or node.get("market")
                or current_label
            )
            outcomes = node.get("outcomes") or node.get("selections")
            if isinstance(outcomes, list) and outcomes and label:
                for outcome in outcomes:
                    if not isinstance(outcome, dict):
                        continue
                    parsed = self._parse_outcome(outcome, label, source_url)
                    if parsed:
                        results.append(parsed)
            for v in node.values():
                self._walk_for_markets(v, results, label, source_url)
        elif isinstance(node, list):
            for item in node:
                self._walk_for_markets(item, results, current_label, source_url)

    def _parse_outcome(
        self, outcome: dict, market_label: str, source_url: str
    ) -> dict | None:
        """Convert one DK outcome dict to our flat prop schema.

        Returns None if the outcome doesn't have odds or a label (not
        a real prop line, e.g. a section header).
        """
        # Outcomes have either "oddsAmerican" / "oddsDecimal" / "price" /
        # "odds" — try each.
        odds = (
            outcome.get("oddsAmerican")
            or outcome.get("price")
            or outcome.get("odds")
        )
        if odds is None and outcome.get("oddsDecimal") is not None:
            # Convert decimal to American
            dec = float(outcome["oddsDecimal"])
            odds = int((dec - 1) * 100) if dec >= 2 else int(-100 / (dec - 1))
        if odds is None:
            return None

        # Outcome label
        outcome_label = (
            outcome.get("label")
            or outcome.get("name")
            or outcome.get("title")
            or outcome.get("description")
        )
        if not outcome_label:
            return None

        # Fighter (if the prop is fighter-specific — most MoV/round props are)
        fighter = (
            outcome.get("participant")
            or outcome.get("fighter")
            or outcome.get("team")
        )

        return {
            "prop_type": _classify_prop(market_label),
            "fighter": fighter or "",
            "outcome": outcome_label,
            "odds": int(odds) if isinstance(odds, (int, float)) else 0,
            "sportsbook": "draftkings",
            "market_label": market_label,
            "source_url": source_url,
        }


def _classify_prop(market_label: str) -> str:
    """Map a raw DK market label to one of our canonical prop types.

    Order of checks:
    1. Exact-label match for bare MoV outcome names (e.g. "KO/TKO",
       "Submission", "Decision") — these are the per-outcome labels DK
       uses when MoV is split into separate sub-markets. Exact-match is
       safer than substring because "Decision" alone is too broad.
    2. Substring match for richer MoV labels (e.g. "Method of Victory",
       "How will the fight end", "Method of Victory - KO/TKO").
    3. Substring match for round-of-finish / distance labels.
    4. Substring match for total-rounds labels.
    5. Fallback: "other".
    """
    label_lower = market_label.lower().strip()
    # 1. Exact-label match FIRST (bare outcome names win over partial match)
    if label_lower in _METHOD_OF_VICTORY_EXACT_LABELS:
        return "method_of_victory"
    # 2. Substring matches for richer labels
    if any(kw in label_lower for kw in _METHOD_OF_VICTORY_KEYWORDS):
        return "method_of_victory"
    if any(kw in label_lower for kw in _ROUND_OF_FINISH_KEYWORDS):
        return "round_of_finish"
    if any(kw in label_lower for kw in _TOTAL_ROUNDS_KEYWORDS):
        return "total_rounds"
    return "other"


if __name__ == "__main__":
    # Smoke test: try a known DK UFC event URL pattern
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else ""
    if not url:
        print("Usage: python -m src.data.dk_props_scraper <dk_event_url>")
        print("Example: python -m src.data.dk_props_scraper \\")
        print('  "https://sportsbook.draftkings.com/event/u-fight-12345"')
        sys.exit(1)
    scraper = DKPropsScraper()
    props = scraper.get_event_props(url)
    print(f"Found {len(props)} prop outcomes for {url}")
    for p in props[:20]:
        print(f"  [{p['prop_type']:20s}] {p['fighter']:25s} {p['outcome']:30s} {p['odds']:+d}")
