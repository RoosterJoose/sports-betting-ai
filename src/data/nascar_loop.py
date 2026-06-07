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
    "cota": "COA", "lucas oil": "LRC", "daytona road course": "DAY", "charlotte roval": "ROV",
    "bristol dirt": "BRI", "nashville": "NSH",
}


def _normalize_driver(name: str) -> str:
    """Normalize driver name for cross-source matching.
    
    Handles:
    - Parenthetical annotations: "Chase Elliott (i)" -> "chase elliott"
    - Trailing asterisks (rookie markers): "William Byron*" -> "william byron"
    - Car number prefixes: "#24 William Byron" -> "william byron"
    - Periods in initials: "A. J. Allmendinger" -> "aj allmendinger"
    - Diacritics: "Daniel Suárez" -> "daniel suarez"
    - Suffixes with periods: "Martin Truex Jr." -> "martin truex jr"
    - Extra whitespace
    """
    n = re.sub(r"\s*\([^)]*\)\s*", " ", name)
    n = re.sub(r"\s*\*+\s*$", "", n)  # trailing asterisks
    n = re.sub(r"\s*\#.*$", "", n)    # trailing car number like #24
    n = re.sub(r"\.", " ", n)          # periods -> spaces (for initials)
    n = re.sub(r"[áàâãäå]", "a", n, flags=re.I)
    n = re.sub(r"[éèêë]", "e", n, flags=re.I)
    n = re.sub(r"[íìîï]", "i", n, flags=re.I)
    n = re.sub(r"[óòôõö]", "o", n, flags=re.I)
    n = re.sub(r"[úùûü]", "u", n, flags=re.I)
    n = re.sub(r"[ñ]", "n", n, flags=re.I)
    n = re.sub(r"[ç]", "c", n, flags=re.I)
    n = re.sub(r"[^a-z0-9\s]", "", n, flags=re.I)  # remove remaining non-alpha
    n = re.sub(r"\s+", " ", n)  # collapse whitespace
    return n.strip().lower()


def _race_name_to_code(race_name: str) -> str:
    """Extract track code from a race name."""
    name_lower = race_name.lower()
    for key, code in RR_TRACK_MAP.items():
        if key in name_lower:
            return code
    return "OTHER"


def fetch_loop_data_season(year: int) -> pd.DataFrame:
    """Fetch loop data for all races in a season from Racing-Reference.
    
    Iterates race numbers 1-38 directly without needing the schedule page.
    
    Returns DataFrame with columns:
        race_number, driver_norm, driver_name, track_code, track_name,
        driver_rating, avg_running_pos, laps_led, quality_passes,
        fast_laps, top15_laps, start_position, finish_position
    """
    cache_file = CACHE_DIR / f"loop_data_{year}.parquet"
    if cache_file.exists():
        df = pd.read_parquet(cache_file)
        print(f"  Loaded cached loop data for {year} ({len(df)} rows)", flush=True)
        return df

    print(f"  Fetching Racing-Reference loop data for {year}...", flush=True)
    print(f"    Iterating race numbers 1-38...", flush=True)

    all_races = []
    # NASCAR Cup Series has ~36-38 races per season
    for race_no in range(1, 39):
        loop_url = f"https://www.racing-reference.info/loopdata/{year}-{race_no:02d}/W/"
        
        try:
            resp = requests.get(loop_url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                continue
        except Exception:
            time.sleep(1.0)
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Find the loop data table.
        # Page has multiple tables with class "tb loopData".
        # Table index 1 is the data table (40+ rows, 19 columns including DRIVER RATING).
        loop_tables = soup.find_all("table", class_=["tb", "loopData"])
        loop_table = None
        for t in loop_tables:
            rows = t.find_all("tr")
            if len(rows) >= 30:  # Data table has 40+ rows
                # Check that it has the right headers (Row 2 has the column headers)
                if len(rows) > 2:
                    hdrs = [h.get_text(strip=True).lower()[:15] for h in rows[2].find_all(["td", "th"])]
                    hdrs_str = " ".join(hdrs)
                    if "driver rating" in hdrs_str and "laps led" in hdrs_str:
                        loop_table = t
                        break

        if loop_table is None:
            time.sleep(0.3)
            continue

        rows = loop_table.find_all("tr")
        if len(rows) < 4:
            time.sleep(0.3)
            continue

        # Structure of the loop data table:
        #   Row 0: merged header with ~800 cells (hidden TH elements) - skip
        #   Row 1: label row ("Loop data for this race:") - skip
        #   Row 2: header row (19 cells: Driver, Start, Mid Race, ..., DRIVER RATING)
        #   Row 3+: data rows (19 cells each)
        header_row = rows[2]
        header_cells = header_row.find_all(["td", "th"])
        hdrs = [c.get_text(strip=True) for c in header_cells]
        hdrs_lower = [h.lower() for h in hdrs]
        
        def find_col(keywords, hdrs_list):
            for i, h in enumerate(hdrs_list):
                for kw in keywords:
                    if kw in h:
                        return i
            return -1
        
        def safe_float(val, default=0.0):
            try:
                return float(re.sub(r"[^\d.\-]", "", val))
            except (ValueError, TypeError):
                return default
        
        def safe_int(val, default=0):
            try:
                return int(re.sub(r"[^\d\-]", "", val))
            except (ValueError, TypeError):
                return default

        driver_idx = find_col(["driver"], hdrs_lower)
        rating_idx = find_col(["driver rating"], hdrs_lower)
        avg_run_idx = find_col(["avg. pos", "avg. pos.", "avg pos"], hdrs_lower)
        laps_led_idx = find_col(["laps led"], hdrs_lower)
        fast_laps_idx = find_col(["fastest lap"], hdrs_lower)
        top15_idx = find_col(["top 15 laps"], hdrs_lower)
        qual_idx = find_col(["quality passes"], hdrs_lower)
        finish_idx = find_col(["finish"], hdrs_lower)
        start_idx = find_col(["start"], hdrs_lower)
        total_laps_idx = find_col(["total laps"], hdrs_lower)
        mid_race_idx = find_col(["mid race"], hdrs_lower)
        pct_qual_idx = find_col(["pct. quality"], hdrs_lower)
        pct_top15_idx = find_col(["pct. top 15"], hdrs_lower)
        green_pass_idx = find_col(["green flag passes"], hdrs_lower)
        pass_diff_idx = find_col(["pass diff"], hdrs_lower)
        high_pos_idx = find_col(["high pos"], hdrs_lower)
        low_pos_idx = find_col(["low pos"], hdrs_lower)

        if driver_idx < 0 or rating_idx < 0:
            time.sleep(0.3)
            continue

        # Extract race name from page title
        race_name = ""
        title_tag = soup.find("title")
        if title_tag:
            title_text = title_tag.get_text(strip=True)
            m = re.search(r"Loop Data: \d{4}\s+(.+?)\s*\|", title_text)
            if m:
                race_name = m.group(1).strip()

        track_code = _race_name_to_code(race_name) if race_name else "OTHER"

        race_drivers = []
        for row in rows[3:]:  # Start from Row 3 (first data row)
            cells = row.find_all("td")
            if len(cells) < 10:  # Need at least 10 cells to be valid
                continue
            
            texts = [c.get_text(strip=True) for c in cells]
            
            driver_name = texts[driver_idx] if driver_idx < len(texts) else ""
            if not driver_name or len(driver_name) > 40:
                continue
            if driver_name.lower() in ["driver", "loop data for this race:", ""]:
                continue

            rating = safe_float(texts[rating_idx] if rating_idx < len(texts) else "0")
            avg_run = safe_float(texts[avg_run_idx] if avg_run_idx < len(texts) else "20")
            laps_led = safe_int(texts[laps_led_idx] if laps_led_idx < len(texts) else "0")
            fast_laps = safe_int(texts[fast_laps_idx] if fast_laps_idx < len(texts) else "0")
            top15 = safe_int(texts[top15_idx] if top15_idx < len(texts) else "0")
            quality_passes = safe_int(texts[qual_idx] if qual_idx < len(texts) else "0")
            finish = safe_int(texts[finish_idx] if finish_idx < len(texts) else "0")
            start = safe_int(texts[start_idx] if start_idx < len(texts) else "0")
            total_laps = safe_int(texts[total_laps_idx] if total_laps_idx < len(texts) else "0")
            mid_race = safe_int(texts[mid_race_idx] if mid_race_idx < len(texts) else "0")
            green_passes = safe_int(texts[green_pass_idx] if green_pass_idx < len(texts) else "0")
            pass_diff = safe_int(texts[pass_diff_idx] if pass_diff_idx < len(texts) else "0")
            high_pos = safe_int(texts[high_pos_idx] if high_pos_idx < len(texts) else "0")
            low_pos = safe_int(texts[low_pos_idx] if low_pos_idx < len(texts) else "0")

            pct_quality = safe_float(texts[pct_qual_idx] if pct_qual_idx < len(texts) else "0")
            pct_top15 = safe_float(texts[pct_top15_idx] if pct_top15_idx < len(texts) else "0")

            race_drivers.append({
                "race_number": race_no,
                "driver_name": driver_name,
                "driver_norm": _normalize_driver(driver_name),
                "track_code": track_code,
                "track_name": race_name,
                "driver_rating": rating,
                "avg_running_position": avg_run,
                "laps_led": laps_led,
                "fastest_laps": fast_laps,
                "top15_laps": top15,
                "quality_passes": quality_passes,
                "finish_position": finish,
                "start_position": start,
                "mid_race_position": mid_race,
                "high_position": high_pos,
                "low_position": low_pos,
                "green_flag_passes": green_passes,
                "passing_differential": pass_diff,
                "total_laps": total_laps,
                "pct_quality_passes": pct_quality,
                "pct_top15_laps": pct_top15,
                "season": str(year),
            })

        if race_drivers:
            df_race = pd.DataFrame(race_drivers)
            all_races.append(df_race)
            top_driver = max(race_drivers, key=lambda x: x["driver_rating"])
            print(f"    Race {race_no:2d}: {len(race_drivers)} drivers, best rating={top_driver['driver_rating']:.1f} ({top_driver['driver_name']})", flush=True)
        else:
            print(f"    Race {race_no:2d}: no drivers parsed", flush=True)

        time.sleep(0.5)  # Be respectful to the server

    if not all_races:
        print(f"    No loop data found for {year}", flush=True)
        return pd.DataFrame()

    result = pd.concat(all_races, ignore_index=True)
    result.to_parquet(cache_file, index=False)
    print(f"  Saved {len(result)} driver-race rows for {year} ({result['driver_norm'].nunique()} drivers, "
          f"{result['race_number'].nunique()} races)", flush=True)
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
        resp = requests.get(
            "https://cf.nascar.com/cacher/2026/qualifying/qualifying_results.json",
            headers=HEADERS,
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
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

    # Fallback: try live-feed endpoint for current race qualifying position
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
                # If driver name is missing, use vehicle number
                if not name and "vehicle_number" in v:
                    name = f"#{v['vehicle_number']}"
                if not name:
                    continue
                norm_name = _normalize_driver(name)
                results[norm_name] = {
                    "position": int(v.get("starting_position", v.get("running_position", 0))),
                    "speed": float(v.get("best_lap_speed", 0)),
                    "best_time": float(v.get("best_lap_time", 0)),
                    "avg_running_pos": float(v.get("average_running_position", 0)),
                    "avg_speed": float(v.get("average_speed", 0)),
                }
            return results
    except Exception:
        pass

    return {}


def merge_loop_data_with_standings(loop_df: pd.DataFrame, standings_df: pd.DataFrame) -> pd.DataFrame:
    """Merge Racing-Reference loop data into the existing standings DataFrame.
    
    Adds columns: driver_rating, avg_running_position, laps_led_loop, fastest_laps,
    quality_passes, top15_laps, and their rolling averages.
    """
    if loop_df.empty:
        return standings_df

    result = standings_df.copy()

    # Add loop data columns with NaN defaults
    loop_cols = ["driver_rating", "avg_running_position", "laps_led_loop", "fastest_laps",
                 "quality_passes", "top15_laps", "pct_quality_passes", "pct_top15_laps",
                 "passing_differential", "avg_speed_loop"]
    for col in loop_cols:
        if col not in result.columns:
            result[col] = np.nan

    # Manual merge: for each driver-race in standings, find matching loop data
    matched = 0
    for idx in result.index:
        driver_norm = _normalize_driver(result.at[idx, "driver_name"] if "driver_name" in result.columns else "")
        race_no = result.at[idx, "race_number"] if "race_number" in result.columns else 0
        if not driver_norm or not race_no:
            continue

        match = loop_df[(loop_df["driver_norm"] == driver_norm) & (loop_df["race_number"] == race_no)]
        if match.empty:
            continue

        row = match.iloc[0]
        result.at[idx, "driver_rating"] = row.get("driver_rating", np.nan)
        result.at[idx, "avg_running_position"] = row.get("avg_running_position", np.nan)
        result.at[idx, "laps_led_loop"] = row.get("laps_led", 0)
        result.at[idx, "fastest_laps"] = row.get("fastest_laps", 0)
        result.at[idx, "quality_passes"] = row.get("quality_passes", 0)
        result.at[idx, "top15_laps"] = row.get("top15_laps", 0)
        result.at[idx, "pct_quality_passes"] = row.get("pct_quality_passes", 0)
        result.at[idx, "pct_top15_laps"] = row.get("pct_top15_laps", 0)
        result.at[idx, "passing_differential"] = row.get("passing_differential", 0)
        matched += 1

    print(f"  Merged loop data: {matched}/{len(result)} rows matched", flush=True)
    return result


def compute_rolling_loop_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling averages of loop data features for each driver.
    
    Adds feature columns with shift(1) to prevent look-ahead bias.
    """
    if df.empty or "driver_rating" not in df.columns:
        return df
    
    result = df.copy()
    result = result.sort_values(["driver_norm", "race_number"]).reset_index(drop=True)
    
    # Rolling driver rating (key feature: r≈0.614)
    for window in [3, 5, 10]:
        col = f"roll_driver_rating_{window}"
        result[col] = (
            result.groupby("driver_norm")["driver_rating"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
    
    # Rolling average running position (smaller = better at that track)
    for window in [3, 5, 10]:
        col = f"roll_avg_running_pos_{window}"
        result[col] = (
            result.groupby("driver_norm")["avg_running_position"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
    
    # Rolling laps led rate
    for window in [3, 5, 10]:
        col = f"roll_laps_led_rate_{window}"
        # laps_led / total_laps capped at 1.0
        laps_pct = result["laps_led"] / result["total_laps"].replace(0, 1)
        result[col] = (
            pd.Series(laps_pct).groupby(result["driver_norm"])
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
    
    # Rolling top15 laps rate
    for window in [3, 5, 10]:
        col = f"roll_top15_rate_{window}"
        result[col] = (
            result.groupby("driver_norm")["top15_laps"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
    
    # Rolling quality passes
    for window in [3, 5, 10]:
        col = f"roll_quality_passes_{window}"
        result[col] = (
            result.groupby("driver_norm")["quality_passes"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
    
    # Fill NaN with reasonable defaults
    fill_defaults = {
        "roll_driver_rating_3": 70.0, "roll_driver_rating_5": 70.0, "roll_driver_rating_10": 70.0,
        "roll_avg_running_pos_3": 20.0, "roll_avg_running_pos_5": 20.0, "roll_avg_running_pos_10": 20.0,
    }
    for col, default in fill_defaults.items():
        if col in result.columns:
            result[col] = result[col].fillna(default)
    
    # Binary rates default to 0
    for col in result.columns:
        if col.startswith("roll_laps_led_rate_") or col.startswith("roll_top15_rate_") or col.startswith("roll_quality_passes_"):
            result[col] = result[col].fillna(0)
    
    return result


if __name__ == "__main__":
    import sys
    
    # Fetch loop data for recent seasons
    years = [2024, 2025]
    if len(sys.argv) > 1:
        years = [int(y) for y in sys.argv[1:]]
    
    df = fetch_multiyear_loop_data(years)
    if not df.empty:
        print(f"\n=== LOOP DATA SUMMARY ===")
        print(f"Total rows: {len(df)}")
        print(f"Seasons: {sorted(df['season'].unique())}")
        print(f"Races: {df.groupby('season')['race_number'].nunique().to_dict()}")
        print(f"Drivers: {df['driver_norm'].nunique()}")
        print(f"\nDriver Rating stats:")
        print(f"  Mean: {df['driver_rating'].mean():.1f}")
        print(f"  Std:  {df['driver_rating'].std():.1f}")
        print(f"  Min:  {df['driver_rating'].min():.1f}")
        print(f"  Max:  {df['driver_rating'].max():.1f}")
        print(f"\nAvg Running Position stats:")
        print(f"  Mean: {df['avg_running_position'].mean():.1f}")
        print(f"  Std:  {df['avg_running_position'].std():.1f}")
        print(f"\nTop drivers by avg Driver Rating:")
        top = df.groupby("driver_name")["driver_rating"].mean().sort_values(ascending=False).head(10)
        for name, rating in top.items():
            print(f"  {name:30s} {rating:.1f}")
        
        # Compute rolling features and show
        print(f"\nComputing rolling loop data features...")
        roll_df = compute_rolling_loop_features(df)
        feat_cols = [c for c in roll_df.columns if c.startswith("roll_")]
        print(f"  Created {len(feat_cols)} rolling features: {feat_cols}")
        
        # Cache the full dataset with features
        out_file = CACHE_DIR / "loop_data_with_features.parquet"
        roll_df.to_parquet(out_file, index=False)
        print(f"  Saved to {out_file}")
    else:
        print("No loop data found")
