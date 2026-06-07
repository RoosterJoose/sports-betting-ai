import re
import time
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.data.base import DataSource

# Map config stat types to DataFrame column names
NASCAR_STAT_MAP = {
    "finishing_position": "finish_position",
    "laps_led": "laps_led_most",
    "top_5": None,  # derived from finish_position
    "top_10": None,  # derived from finish_position
    "top_20": None,  # derived from finish_position
    "stage_points": None,  # not available in data
    "avg_finish": None,  # derived from finish_position
    "laps_completed": None,  # not available in data
}


CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache" / "nascar"

WIKI_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
}

def _parse_finish(val):
    """Extract leading finish number from annotated NASCAR value like '1*12' or '27F'."""
    if isinstance(val, (int, float)):
        if np.isnan(val):
            return None
        return int(val)
    if not isinstance(val, str):
        return None
    val = val.replace("\u2020", "").replace("\u2021", "").replace("\u2022", "").strip()
    m = re.match(r"^(\d+)", val)
    if m:
        return int(m.group(1))
    return None


def _scrape_race_grid(wiki_title):
    """Scrape starting grid (Grid column) from a Wikipedia race article.
    
    Returns dict mapping normalized driver name -> starting position (1-40+).
    Returns empty dict on failure.
    """
    import time
    from bs4 import BeautifulSoup
    url = f"https://en.wikipedia.org/wiki/{wiki_title.replace(' ', '_')}"
    try:
        resp = requests.get(url, headers=WIKI_HEADERS, timeout=10)
        if resp.status_code != 200:
            return {}
    except requests.RequestException:
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
                result[_normalize_driver_name(driver)] = pos
        if len(result) >= 10:
            return result
    return {}


def _normalize_driver_name(name):
    n = re.sub(r"\s*\([^)]*\)\s*", "", name)
    n = re.sub(r"\s*†", "", n)
    n = re.sub(r"\s+\d+$", "", n)
    return n.strip().lower()

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

TRACK_NAMES = {
    "DAY": "Daytona", "ATL": "Atlanta", "COA": "Circuit of the Americas",
    "PHO": "Phoenix", "LVS": "Las Vegas", "HOM": "Homestead-Miami",
    "MAR": "Martinsville", "BRI": "Bristol", "TAL": "Talladega",
    "RCH": "Richmond", "CLT": "Charlotte", "KAN": "Kansas",
    "SON": "Sonoma", "NHA": "New Hampshire", "POC": "Pocono",
    "IND": "Indianapolis", "MCH": "Michigan", "GLN": "Watkins Glen",
    "DAR": "Darlington", "CAL": "Auto Club", "TEX": "Texas",
    "DOV": "Dover", "CHI": "Chicago", "NWS": "North Wilkesboro",
    "IRP": "Indianapolis Road", "ROA": "Road America", "LRC": "Lucas Oil",
    "CSC": "COTA", "NSH": "Nashville", "GWY": "Gateway",
    "KEN": "Kentucky", "WAT": "Watkins Glen", "IOW": "Iowa",
    "MXC": "Mexico City", "ROV": "Charlotte Roval", "NSS": "Nashville SS",
    "GTW": "Gateway",
}


class NASCARDataSource(DataSource):
    def __init__(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def fetch_player_game_logs(self, seasons: list[str]) -> pd.DataFrame:
        cache_file = CACHE_DIR / f"standings_{'_'.join(sorted(seasons))}.parquet"
        if cache_file.exists():
            return pd.read_parquet(cache_file)

        frames = []
        for season in seasons:
            try:
                year = int(str(season)[:4])
            except ValueError:
                continue
            df = self._fetch_year_standings(year)
            if not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True)
        result.to_parquet(cache_file, index=False)
        return result

    @staticmethod
    def _normalize_driver(name):
        n = re.sub(r"\s*\([^)]*\)\s*", "", name)
        n = re.sub(r"\s*†", "", n)
        n = re.sub(r"\s+\d+$", "", n)
        return n.strip().lower()

    def _scrape_season_grids(self, year: int) -> dict:
        """Scrape starting grids for all races in a season.
        Returns {race_number: {driver_norm: starting_position}}."""
        cache_file = CACHE_DIR / f"grids_{year}.parquet"
        if cache_file.exists():
            df = pd.read_parquet(cache_file)
            result = {}
            for rn, grp in df.groupby("race_number"):
                result[rn] = dict(zip(grp["driver_norm"], grp["starting_position"]))
            return result

        url = f"https://en.wikipedia.org/wiki/{year}_NASCAR_Cup_Series"
        try:
            resp = requests.get(url, headers=WIKI_HEADERS, timeout=15)
            resp.raise_for_status()
        except Exception:
            return {}

        soup = BeautifulSoup(resp.text, "html.parser")
        race_links = {}
        for table in soup.find_all("table", class_="wikitable"):
            text = table.get_text().lower()
            if "pole position" not in text or "report" not in text:
                continue
            rows = table.find_all("tr")
            for row in rows[1:]:
                cells = row.find_all(["td", "th"])
                n_cells = len(cells)
                if n_cells < 3:
                    continue
                try:
                    no = int(cells[0].get_text(strip=True))
                except (ValueError, TypeError):
                    continue
                if no < 1 or no > 36:
                    continue
                # Report column is the last cell (index varies due to rowspan)
                report_cell = cells[-1]
                link = report_cell.find("a")
                if link and link.get("href"):
                    wiki_title = link["href"].replace("/wiki/", "").replace("_", " ")
                    race_links[no] = wiki_title
            break

        if not race_links:
            print(f"    No race report links found for {year}")
            return {}

        all_records = []
        for race_no in sorted(race_links.keys())[:36]:
            wiki_title = race_links[race_no]
            grid = _scrape_race_grid(wiki_title)
            time.sleep(0.3)
            if not grid:
                continue
            for driver_norm, start_pos in grid.items():
                all_records.append({
                    "race_number": race_no,
                    "driver_norm": driver_norm,
                    "starting_position": start_pos,
                })

        if not all_records:
            return {}

        result_df = pd.DataFrame(all_records)
        result_df.to_parquet(cache_file, index=False)
        result = {}
        for rn, grp in result_df.groupby("race_number"):
            result[rn] = dict(zip(grp["driver_norm"], grp["starting_position"]))
        return result

    def _fetch_year_standings(self, year: int) -> pd.DataFrame:
        url = f"https://en.wikipedia.org/wiki/{year}_NASCAR_Cup_Series"
        resp = requests.get(url, headers=WIKI_HEADERS, timeout=15)
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))

        standings_table = None
        for t in tables:
            cols_lower = [str(c).lower().strip() for c in t.columns]
            if "pos." in cols_lower and "driver" in cols_lower:
                standings_table = t
                break

        if standings_table is None:
            return pd.DataFrame()

        df = standings_table.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)
        df.columns = [str(c).strip() for c in df.columns]

        # Determine correct race abbreviation sequence (positional, ignores unnamed column)
        pts_idx = list(df.columns).index("Pts.") if "Pts." in list(df.columns) else None
        if pts_idx is None:
            pts_idx = list(df.columns).index("Points") if "Points" in list(df.columns) else 39
        all_header_labels = list(df.columns[2:pts_idx])
        correct_seq = [c for c in all_header_labels if "unnamed" not in str(c).lower()][:36]

        if len(correct_seq) < 30:
            return pd.DataFrame()

        # Scrape starting grids for this season
        season_grids = self._scrape_season_grids(year)

        roster = {}
        for t in tables:
            cols_set = set(str(c).lower().strip() for c in t.columns)
            if not ("driver" in cols_set and ("team" in cols_set or "manufacturer" in cols_set)):
                continue
            rt = t.copy()
            if isinstance(rt.columns, pd.MultiIndex):
                rt.columns = rt.columns.get_level_values(-1)
            rt.columns = [str(c).strip() for c in rt.columns]
            drv_col = "Driver" if "Driver" in rt.columns else None
            if drv_col:
                for _, row in rt.iterrows():
                    d = str(row.get(drv_col, "")).strip()
                    if not d or d.lower() == "nan":
                        continue
                    norm = self._normalize_driver(d)
                    team = str(row.get("Team", "")).strip() if "Team" in rt.columns else ""
                    roster[norm] = {
                        "team": team,
                        "manufacturer": str(row.get("Manufacturer", "")).strip() if "Manufacturer" in rt.columns else "",
                        "car_number": str(row.get("No.", row.get("No", ""))).strip(),
                    }

        pole_drivers = {}
        laps_led_drivers = {}
        winning_drivers = {}
        for t in tables:
            cols_lower = [str(c).lower().strip() for c in t.columns]
            if "pole position" in cols_lower and "winning driver" in cols_lower:
                rt = t.copy()
                if isinstance(rt.columns, pd.MultiIndex):
                    rt.columns = rt.columns.get_level_values(-1)
                rt.columns = [str(c).strip() for c in rt.columns]
                no_col = "No." if "No." in rt.columns else "No" if "No" in rt.columns else None
                if no_col:
                    for _, row in rt.iterrows():
                        try:
                            race_num = int(row.get(no_col, 0))
                        except (ValueError, TypeError):
                            continue
                        if race_num < 1 or race_num > 36:
                            continue
                        race_abbr = correct_seq[race_num - 1]
                        pole_drivers[race_abbr] = str(row.get("Pole position", "")).strip()
                        laps_led_drivers[race_abbr] = str(row.get("Most laps led", "")).strip()
                        winning_drivers[race_abbr] = str(row.get("Winning driver", "")).strip()
                break

        records = []
        for _, row in df.iterrows():
            driver = str(row.get("Driver", "")).strip()
            if not driver:
                continue
            driver_lower = self._normalize_driver(driver)

            pos_text = str(row.get("Pos.", "0")).strip()
            try:
                standings_pos = int(pos_text.replace("\u2020", "").strip())
            except ValueError:
                standings_pos = 0

            driver_info = roster.get(driver_lower, {})

            pts = 0
            if pts_idx is not None and pts_idx < len(row):
                try:
                    pts = float(str(row.iloc[pts_idx]).replace(",", ""))
                except (ValueError, TypeError):
                    pts = 0

            for race_idx, race_abbr in enumerate(correct_seq):
                col_pos = 2 + race_idx
                if col_pos >= len(row):
                    continue
                finish_raw = row.iloc[col_pos]
                parsed = _parse_finish(finish_raw)
                if parsed is None or parsed < 1 or parsed > 45:
                    continue

                is_pole = 1 if (self._normalize_driver(pole_drivers.get(race_abbr, "")) == driver_lower) else 0
                is_laps_leader = 1 if (self._normalize_driver(laps_led_drivers.get(race_abbr, "")) == driver_lower) else 0
                is_winner = 1 if (self._normalize_driver(winning_drivers.get(race_abbr, "")) == driver_lower) else 0

                race_grid = season_grids.get(race_idx + 1, {})
                starting_position = race_grid.get(driver_lower, 0)

                records.append({
                    "driver_name": driver,
                    "team": driver_info.get("team", ""),
                    "manufacturer": driver_info.get("manufacturer", ""),
                    "car_number": driver_info.get("car_number", ""),
                    "season": str(year),
                    "race_abbr": race_abbr,
                    "race_number": race_idx + 1,
                    "race_order": race_idx + 1,
                    "finish_position": parsed,
                    "starting_position": starting_position,
                    "standings_position": standings_pos,
                    "total_points": pts,
                    "pole_position": is_pole,
                    "laps_led_most": is_laps_leader,
                    "is_winner": is_winner,
                    "track_type": TRACK_TYPES.get(race_abbr, "other"),
                    "track_name": TRACK_NAMES.get(race_abbr, race_abbr),
                    "game_date": f"{year}-01-01",
                })

        result = pd.DataFrame(records)
        
        # Add player_id for pipeline merge compatibility
        result["player_id"] = result["driver_name"]
        
        # Derive PrizePicks-compatible columns
        result["finishing_position"] = result["finish_position"]
        result["top_5"] = (result["finish_position"] <= 5).astype(int)
        result["top_10"] = (result["finish_position"] <= 10).astype(int)
        result["top_20"] = (result["finish_position"] <= 20).astype(int)
        result["laps_led"] = result["laps_led_most"] * 1
        
        # Add alias for pipeline target resolution
        for alias, col in NASCAR_STAT_MAP.items():
            if col is not None and col in result.columns and alias not in result.columns:
                result[alias] = result[col]
        
        print(f"  NASCAR {year}: {len(result)} driver-race rows, {result['driver_name'].nunique()} drivers")
        return result

    def fetch_schedule(self, season: str) -> pd.DataFrame:
        return self.fetch_player_game_logs([season])

    def fetch_player_stats(self, driver_id: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        return pd.DataFrame()

    def fetch_team_stats(self, team_id: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        return pd.DataFrame()
