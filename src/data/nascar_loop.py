"""
NASCAR Loop Data Scraper — Racing-Reference + cf.nascar.com Swagger API.

Provides:
- Historical loop data per race (driver rating, avg running position, laps led)
- Live qualifying/practice data from cf.nascar.com
- Merged driver-season statistics

Based on research: Driver Rating (r≈0.614) is the single most predictive pre-race feature.
"""
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache" / "nascar_loop"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
}

# Racing-Reference track codes -> short codes for merge compatibility
RR_TRACK_MAP = {
    "daytona": "DAY", "atlanta": "ATL", "las vegas": "LVS", "phoenix": "PHO",
    "homestead": "HOM", "martinsville": "MAR", "bristol": "BRI", "talladega": "TAL",
    "richmond": "RCH", "charlotte": "CLT", "kansas": "KAN", "sonoma": "SON",
    "new hampshire": "NHA", "pocono": "POC", "indianapolis": "IND", "michigan": "MCH",
    "watkins glen": "GLN", "darlington": "DAR", "auto club": "CAL", "texas": "TEX",
    "dover": "DOV", "chicago": "CHI", "nashville": "NSH", "gateway": "GWY",
    "kentucky": "KEN", "iowa": "IOW", "north wilkesboro": "NWS", "road america": "ROA",
    "cota": "COA", "lucas oil": "LRC", "daytona road": "DAY", "charlotte roval": "ROV",
}


def _normalize_driver(name: str) -> str:
    """Normalize driver name to lowercase, stripped of parenthetical annotations."""
    n = re.sub(r"\s*\([^)]*\)\s*", "", name)
    n = re.sub(r"\s*\*+\s*$", "", n)  # trailing asterisks (rookie markers)
    return n.strip().lower()


def _race_name_to_code(race_name: str) -> str:
    """Extract track code from a race name like 'Pennzoil 400 presented by Jiffy Lube'."""
    name_lower = race_name.lower()
    for key, code in RR_TRACK_MAP.items():
        if key in name_lower:
            return code
    return "OTHER"


def fetch_loop_data_season(year: int) -> pd.DataFrame:
    """Fetch loop data for all races in a season from Racing-Reference.
    
    Returns DataFrame with columns:
        race_number, driver_norm, driver_name, track_code, track_name,
        driver_rating, avg_running_pos, laps_led, fast_laps,
        start_position, finish_position, laps_completed, points_earned
    """
    cache_file = CACHE_DIR / f"loop_data_{year}.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    print(f"  Fetching Racing-Reference loop data for {year}...", flush=True)

    # Step 1: Get the season schedule to find race URLs
    schedule_url = f"https://www.racing-reference.info/season-stats/{year}/W/"
    try:
        resp = requests.get(schedule_url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"    Schedule error: {e}", flush=True)
        return pd.DataFrame()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Find race links from the schedule table
    race_entries = []
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if len(rows) < 3:
            continue
        # Look for a table with race results links
        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 3:
                continue
            # Check for numeric first cell (race number)
            first_text = cells[0].get_text(strip=True)
            try:
                race_no = int(first_text)
            except (ValueError, TypeError):
                continue
            if race_no < 1 or race_no > 40:
                continue
            # Look for link in the second cell (race name)
            link = cells[1].find("a") if len(cells) > 1 else None
            if link and link.get("href"):
                race_entries.append({
                    "race_number": race_no,
                    "url": link["href"] if link["href"].startswith("http") else f"https://www.racing-reference.info{link['href']}",
                    "race_name": link.get_text(strip=True),
                })

    if not race_entries:
        # Fallback: try the loop data index
        print(f"    No schedule entries found via HTML, trying loop index...", flush=True)
        return pd.DataFrame()

    # Step 2: For each race, fetch the loop data page
    all_races = []
    for entry in race_entries:
        race_no = entry["race_number"]
        race_url = entry["url"]
        race_name = entry["race_name"]

        # Construct loop data URL
        # Racing-Reference loop data pattern: https://www.racing-reference.info/loopdata/{year}-{race_no}/W/
        loop_url = f"https://www.racing-reference.info/loopdata/{year}-{race_no:02d}/W/"
        
        try:
            resp = requests.get(loop_url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                time.sleep(0.5)
                continue
        except Exception:
            time.sleep(0.5)
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        loop_table = None
        for table in soup.find_all("table", class_="tb"):
            hdrs = [h.get_text(strip=True).lower()[:15] for h in table.find_all("th")]
            hdrs_str = " ".join(hdrs)
            if "driver rating" in hdrs_str and "avg running" in hdrs_str:
                loop_table = table
                break

        if loop_table is None:
            time.sleep(0.5)
            continue

        # Parse the loop data table
        rows = loop_table.find_all("tr")
        if len(rows) < 3:
            time.sleep(0.5)
            continue

        # Get header indices
        hdrs = [h.get_text(strip=True).lower() for h in rows[0].find_all("th")]
        
        def find_col(keywords, hdrs_list):
            for i, h in enumerate(hdrs_list):
                for kw in keywords:
                    if kw in h:
                        return i
            return -1

        finish_idx = find_col(["fin", "pos"], hdrs)
        start_idx = find_col(["start"], hdrs)
        driver_idx = find_col(["driver"], hdrs)
        rating_idx = find_col(["driver rating", "rating"], hdrs)
        avg_run_idx = find_col(["avg run", "average running"], hdrs)
        laps_led_idx = find_col(["laps led"], hdrs)
        fast_laps_idx = find_col(["fast laps", "fastest laps"], hdrs)
        laps_comp_idx = find_col(["laps comp", "laps completed"], hdrs)

        if driver_idx < 0 or rating_idx < 0:
            time.sleep(0.5)
            continue

        track_code = _race_name_to_code(race_name)

        race_drivers = []
        for row in rows[1:]:
            cells = row.find_all("td")
            if len(cells) <= max(driver_idx, rating_idx):
                continue
            texts = [c.get_text(strip=True) for c in cells]

            driver_name = texts[driver_idx] if driver_idx < len(texts) else ""
            if not driver_name:
                continue

            def safe_float(val, default=0.0):
                try:
                    return float(re.sub(r"[^\d.]", "", val))
                except (ValueError, TypeError):
                    return default

            def safe_int(val, default=0):
                try:
                    return int(re.sub(r"[^\d]", "", val))
                except (ValueError, TypeError):
                    return default

            driver_rating = safe_float(texts[rating_idx] if rating_idx >= 0 and rating_idx < len(texts) else "0")
            avg_running_pos = safe_float(texts[avg_run_idx] if avg_run_idx >= 0 and avg_run_idx < len(texts) else "20")
            laps_led = safe_int(texts[laps_led_idx] if laps_led_idx >= 0 and laps_led_idx < len(texts) else "0")
            fast_laps = safe_int(texts[fast_laps_idx] if fast_laps_idx >= 0 and fast_laps_idx < len(texts) else "0")
            finish = safe_int(texts[finish_idx] if finish_idx >= 0 and finish_idx < len(texts) else "0")
            start = safe_int(texts[start_idx] if start_idx >= 0 and start_idx < len(texts) else "0")
            laps_completed = safe_float(texts[laps_comp_idx] if laps_comp_idx >= 0 and laps_comp_idx < len(texts) else "0")

            race_drivers.append({
                "race_number": race_no,
                "driver_name": driver_name,
                "driver_norm": _normalize_driver(driver_name),
                "track_code": track_code,
                "track_name": race_name,
                "driver_rating": driver_rating,
                "avg_running_position": avg_running_pos,
                "laps_led": laps_led,
                "fastest_laps": fast_laps,
                "finish_position": finish,
                "start_position": start,
                "laps_completed": laps_completed,
                "season": str(year),
            })

        if race_drivers:
            df_race = pd.DataFrame(race_drivers)
            all_races.append(df_race)
            if len(race_entries) <= 5 or race_no % 5 == 0:
                print(f"    Race {race_no:2d}: {len(race_drivers)} drivers", flush=True)

        time.sleep(0.5)  # Be respectful to the server

    if not all_races:
        print(f"    No loop data found for {year}", flush=True)
        return pd.DataFrame()

    result = pd.concat(all_races, ignore_index=True)
    result.to_parquet(cache_file, index=False)
    print(f"  Saved {len(result)} driver-race rows for {year}", flush=True)
    return result


def fetch_multiyear_loop_data(years: list[int]) -> pd.DataFrame:
    """Fetch loop data for multiple years, combining with caching."""
    frames = []
    for year in years:
        df = fetch_loop_data_season(year)
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_live_qualifying() -> dict:
    """Fetch live qualifying data from cf.nascar.com Swagger API.
    
    Returns {driver_norm: {'position': int, 'speed': float, 'best_time': float}}.
    """
    try:
        # Qualifying results endpoint
        resp = requests.get(
            "https://cf.nascar.com/cacher/2026/qualifying/qualifying_results.json",
            headers=HEADERS,
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            # Parse qualifying results
            results = {}
            sessions = data.get("sessions", [data] if not isinstance(data, list) else data)
            for session in (sessions if isinstance(sessions, list) else [sessions]):
                vehicles = session.get("vehicles", []) if isinstance(session, dict) else []
                for v in vehicles:
                    driver = v.get("driver", {})
                    name = driver.get("fullName", driver.get("name", ""))
                    if not name:
                        continue
                    norm_name = _normalize_driver(name)
                    results[norm_name] = {
                        "position": int(v.get("qualifyingPosition", v.get("position", 0))),
                        "speed": float(v.get("speed", v.get("qualifyingSpeed", 0))),
                        "best_time": float(v.get("bestTime", v.get("qualifyingTime", 0))),
                    }
            if results:
                return results
    except Exception:
        pass

    # Fallback: try live-feed endpoint
    try:
        resp = requests.get(
            "https://cf.nascar.com/live/feeds/live-feed.json",
            headers=HEADERS,
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            results = {}
            for v in data.get("vehicles", []):
                driver = v.get("driver", {})
                name = driver.get("fullName", driver.get("name", ""))
                if not name:
                    continue
                norm_name = _normalize_driver(name)
                results[norm_name] = {
                    "position": int(v.get("qualifiedPosition", 0)),
                    "speed": float(v.get("speed", 0)),
                    "best_time": float(v.get("bestLapTime", 0)),
                }
            return results
    except Exception:
        pass

    return {}


def merge_loop_data_with_standings(loop_df: pd.DataFrame, standings_df: pd.DataFrame) -> pd.DataFrame:
    """Merge Racing-Reference loop data into the existing standings DataFrame.
    
    Adds columns: driver_rating, avg_running_position, laps_led_loop, fastest_laps
    """
    if loop_df.empty:
        return standings_df

    result = standings_df.copy()

    # Create merge keys: normalized driver name + race number
    standings_key = result["driver_name"].str.lower().str.strip()
    loop_key = loop_df["driver_norm"]

    # Merge on driver name + race number
    result = result.reset_index(drop=True)
    
    # Add loop data columns with NaN defaults
    for col in ["driver_rating", "avg_running_position", "laps_led_loop", "fastest_laps"]:
        if col not in result.columns:
            result[col] = np.nan

    # Manual merge: for each driver-race in standings, find matching loop data
    for idx, row in result.iterrows():
        driver_norm = _normalize_driver(row.get("driver_name", ""))
        race_no = row.get("race_number", 0)
        if not driver_norm or not race_no:
            continue

        match = loop_df[(loop_df["driver_norm"] == driver_norm) & (loop_df["race_number"] == race_no)]
        if match.empty:
            continue

        result.at[idx, "driver_rating"] = match.iloc[0].get("driver_rating", np.nan)
        result.at[idx, "avg_running_position"] = match.iloc[0].get("avg_running_position", np.nan)
        result.at[idx, "laps_led_loop"] = match.iloc[0].get("laps_led", 0)
        result.at[idx, "fastest_laps"] = match.iloc[0].get("fastest_laps", 0)

    matched = result["driver_rating"].notna().sum()
    print(f"  Merged loop data: {matched}/{len(result)} rows matched", flush=True)
    return result


if __name__ == "__main__":
    # Test: fetch loop data for 2025
    df = fetch_loop_data_season(2025)
    if not df.empty:
        print(f"\n2025 loop data: {len(df)} rows, {df['driver_norm'].nunique()} drivers")
        print(f"Columns: {list(df.columns)}")
        print(f"\nDriver Rating stats:")
        print(f"  Mean: {df['driver_rating'].mean():.1f}")
        print(f"  Std:  {df['driver_rating'].std():.1f}")
        print(f"  Min:  {df['driver_rating'].min():.1f}")
        print(f"  Max:  {df['driver_rating'].max():.1f}")
        print(f"\nTop drivers by avg rating:")
        top = df.groupby("driver_name")["driver_rating"].mean().sort_values(ascending=False).head(10)
        for name, rating in top.items():
            print(f"  {name:30s} {rating:.1f}")
