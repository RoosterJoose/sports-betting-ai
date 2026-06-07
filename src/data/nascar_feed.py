import re
import time
import urllib.parse
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache" / "nascar_feed"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

WIKI_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
}

TRACK_TYPES = {
    "DAY": "superspeedway", "ATL": "intermediate", "COA": "road",
    "PHO": "short", "LVS": "intermediate", "HOM": "intermediate",
    "MAR": "short", "BRI": "short", "TAL": "superspeedway",
    "RCH": "short", "CLT": "intermediate", "KAN": "intermediate",
    "SON": "road", "NHA": "short", "POC": "triangle",
    "IND": "speedway", "MCH": "intermediate", "GLN": "road",
    "DAR": "intermediate", "CAL": "intermediate", "TEX": "intermediate",
    "DOV": "intermediate", "CHI": "road", "NWS": "short",
    "IRP": "short", "ROA": "road", "LRC": "short",
    "CSC": "road", "NSH": "short", "GWY": "intermediate",
    "KEN": "intermediate", "WAT": "road", "IOW": "short",
    "MXC": "road", "ROV": "road", "NSS": "short",
    "GTW": "intermediate",
}


def fetch_live_feed():
    """Fetch current live feed data from NASCAR's public API."""
    try:
        resp = requests.get(
            "https://cf.nascar.com/live/feeds/live-feed.json",
            headers=WIKI_HEADERS,
            timeout=10
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def extract_vehicles(live_feed):
    """Extract driver vehicle data from live feed (practice/qualifying/race)."""
    return live_feed.get("vehicles", []) if live_feed else []


def fetch_wikipedia_starting_grid(race_wiki_title):
    """Scrape starting grid from a Wikipedia race article.
    Returns {driver_name_lower: starting_position}."""
    url = f"https://en.wikipedia.org/wiki/{race_wiki_title.replace(' ', '_')}"
    try:
        resp = requests.get(url, headers=WIKI_HEADERS, timeout=10)
        if resp.status_code != 200:
            return {}
    except Exception:
        return {}

    soup = BeautifulSoup(resp.text, "html.parser")
    for table in soup.find_all("table", class_="wikitable"):
        rows = table.find_all("tr")
        if len(rows) < 3:
            continue
        hdrs = [h.get_text(strip=True).lower()[:10] for h in rows[0].find_all(["th", "td"])]
        if "grid" not in hdrs or "driver" not in hdrs:
            continue
        grid_idx = hdrs.index("grid")
        drv_idx = hdrs.index("driver")
        result = {}
        for row in rows[1:]:
            cells = row.find_all(["td", "th"])
            texts = [c.get_text(strip=True) for c in cells]
            if len(texts) <= max(grid_idx, drv_idx):
                continue
            try:
                pos = int(re.match(r"(\d+)", texts[grid_idx])[1])
            except (ValueError, TypeError, AttributeError):
                continue
            driver = re.sub(r"\s*\([^)]*\)", "", texts[drv_idx]).strip()
            if pos >= 1 and pos <= 45:
                driver_lower = re.sub(r"\s*\([^)]*\)", "", driver).strip().lower()
                result[driver_lower] = pos
        if len(result) >= 10:
            return result
    return {}


def get_race_schedule(year: int = 2025) -> pd.DataFrame:
    """Get NASCAR Cup Series schedule with race names, dates, and wiki titles."""
    cache_file = CACHE_DIR / f"schedule_{year}.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    url = f"https://en.wikipedia.org/wiki/{year}_NASCAR_Cup_Series"
    try:
        resp = requests.get(url, headers=WIKI_HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception:
        return pd.DataFrame()

    soup = BeautifulSoup(resp.text, "html.parser")
    records = []

    # First pass: extract confirmed race links from race results table
    seen_numbers = set()
    for table in soup.find_all("table", class_="wikitable"):
        text = table.get_text().lower()
        if "pole position" not in text or "report" not in text:
            continue
        rows = table.find_all("tr")
        for row in rows[1:]:
            cells = row.find_all(["td", "th"])
            if len(cells) < 3:
                continue
            try:
                no = int(cells[0].get_text(strip=True))
            except (ValueError, TypeError):
                continue
            if no < 1 or no > 36:
                continue
            report_cell = cells[-1]
            link = report_cell.find("a")
            wiki_title = ""
            race_name = cells[1].get_text(strip=True) if len(cells) > 1 else ""
            if link and link.get("href"):
                wiki_title = link["href"].replace("/wiki/", "").replace("_", " ")
            records.append({
                "race_number": no,
                "race_name": race_name,
                "wiki_title": wiki_title,
            })
            seen_numbers.add(no)
        break

    # Second pass: extract from schedule table for races without report links
    for table in soup.find_all("table", class_="wikitable"):
        if isinstance(table.find_all("tr")[0].find_all(["th", "td"])[0].get_text(strip=True), str):
            pass
        hdrs = [h.get_text(strip=True) for h in table.find_all("tr")[0].find_all(["th", "td"])]
        hdrs_lower = [h.lower().strip()[:10] for h in hdrs]
        if "race name" not in hdrs_lower and "race name" not in str(hdrs).lower():
            continue
        rows = table.find_all("tr")
        # Try to find No. and Race name columns (usually indices 0 and 1)
        no_col, name_col = 0, 1
        for ri, h in enumerate(hdrs):
            hl = h.lower().strip()
            if hl in ("no", "no.", "#"):
                no_col = ri
            if hl == "race name" or hl == "race":
                name_col = ri
        for row in rows[1:]:
            cells = row.find_all(["td", "th"])
            if len(cells) <= max(no_col, name_col):
                continue
            try:
                no = int(cells[no_col].get_text(strip=True))
            except (ValueError, TypeError):
                continue
            if no < 1 or no > 36:
                continue
            if no in seen_numbers:
                continue
            race_name = cells[name_col].get_text(strip=True)
            # Construct wiki title from race name
            import urllib.parse
            safe_name = race_name.replace(" ", "_")
            safe_name = re.sub(r"\[.*?\]", "", safe_name)
            safe_name = re.sub(r"[#]", "", safe_name)
            wiki_title = f"{year}_{safe_name}"
            records.append({
                "race_number": no,
                "race_name": race_name,
                "wiki_title": wiki_title,
            })
            seen_numbers.add(no)
        break

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records)
    result.to_parquet(cache_file, index=False)
    return result


def find_next_race(schedule: pd.DataFrame) -> dict:
    """Find the next upcoming race that hasn't happened yet.
    Returns dict with race_number, race_name, wiki_title."""
    today = date.today()

    # For each race in schedule, check if the Wikipedia article exists
    # (future races might not have articles yet)
    for _, r in schedule.iterrows():
        if not r.get("wiki_title"):
            continue
        # Check if the race article has race results (indicating it happened)
        wiki = r["wiki_title"].replace(" ", "_")
        url = f"https://en.wikipedia.org/wiki/{wiki}"
        try:
            resp = requests.head(url, headers=WIKI_HEADERS, timeout=5)
            if resp.status_code != 200:
                return r.to_dict()
        except Exception:
            return r.to_dict()

        # Check if race results table exists (race hasn't happened)
        try:
            resp2 = requests.get(url, headers=WIKI_HEADERS, timeout=10)
            soup = BeautifulSoup(resp2.text, "html.parser")
            has_grid = False
            for table in soup.find_all("table", class_="wikitable"):
                hdrs = [h.get_text(strip=True).lower()[:10] for h in table.find_all(["th", "td"])]
                if "grid" in hdrs and "driver" in hdrs:
                    has_grid = True
                    break
            if not has_grid:
                return r.to_dict()
        except Exception:
            return r.to_dict()

    # All races have articles with grids — return last race (season over)
    return schedule.iloc[-1].to_dict()


def get_driver_loop_stats() -> dict:
    """Fetch current driver stats from FRCS.pro.
    Returns {driver_name_lower: {metric: value}} or empty dict."""
    try:
        resp = requests.get(
            "https://frcs.pro/nascar/cup/drivers/current",
            headers=WIKI_HEADERS,
            timeout=15
        )
        if resp.status_code != 200:
            return {}
    except Exception:
        return {}

    soup = BeautifulSoup(resp.text, "html.parser")
    tables = soup.find_all("table")
    stats = {}
    for table in tables:
        rows = table.find_all("tr")
        if len(rows) < 3:
            continue
        hdrs = [h.get_text(strip=True).lower()[:10] for h in rows[0].find_all(["th", "td"])]
        has_driver = "driver" in hdrs or "name" in hdrs
        has_points = "point" in hdrs or "pts" in hdrs
        if not has_driver or not has_points:
            continue

        drv_idx = hdrs.index("driver") if "driver" in hdrs else hdrs.index("name")
        for row in rows[1:]:
            cells = row.find_all(["td", "th"])
            texts = [c.get_text(strip=True) for c in cells]
            if len(texts) <= drv_idx:
                continue
            driver = re.sub(r"\s*\([^)]*\)", "", texts[drv_idx]).strip().lower()
            row_stats = {}
            for ci, h in enumerate(hdrs):
                if ci < len(texts) and h not in ("driver", "name", "no", "car", "rank"):
                    try:
                        row_stats[h] = float(re.sub(r"[^0-9.]", "", texts[ci]))
                    except (ValueError, TypeError, AttributeError):
                        row_stats[h] = texts[ci]
            if driver:
                stats[driver] = row_stats
        if stats:
            break
    return stats
