"""MLB stadium locations and weather data fetcher (open-meteo, no API key).

Phase 2 of MLB plan: weather features for HR and TB models. Per NotebookLM:
wind speed/direction affects HR rate (15+ mph heuristic). Temperature and
humidity also matter.

Fetches hourly forecast for the next 7 days for each MLB stadium. Cached
in data/cache/mlb/weather/ as parquet keyed by (park, date, hour).

Usage:
    from src.data.mlb_weather import fetch_weather_for_park
    df = fetch_weather_for_park("NYY", lat=40.8296, lon=-73.9262)
"""
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache" / "mlb" / "weather"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 30 MLB stadium coordinates
MLB_PARKS = {
    "ARI": {"lat": 33.4455, "lon": -112.0667, "name": "Chase Field",         "roof": True},
    "ATL": {"lat": 33.7348, "lon": -84.3897, "name": "Truist Park",         "roof": False},
    "BAL": {"lat": 39.2838, "lon": -76.6217, "name": "Camden Yards",        "roof": False},
    "BOS": {"lat": 42.3467, "lon": -71.0972, "name": "Fenway Park",         "roof": False},
    "CHC": {"lat": 41.9484, "lon": -87.6553, "name": "Wrigley Field",       "roof": False},
    "CIN": {"lat": 39.0975, "lon": -84.5067, "name": "Great American",      "roof": False},
    "CLE": {"lat": 41.4962, "lon": -81.6852, "name": "Progressive Field",   "roof": False},
    "COL": {"lat": 39.7559, "lon": -104.9942,"name": "Coors Field",         "roof": False},
    "CWS": {"lat": 41.8299, "lon": -87.6338, "name": "Guaranteed Rate",     "roof": False},
    "DET": {"lat": 42.3390, "lon": -83.0485, "name": "Comerica Park",       "roof": False},
    "HOU": {"lat": 29.7572, "lon": -95.3552, "name": "Minute Maid Park",    "roof": True},
    "KC":  {"lat": 39.0517, "lon": -94.4803, "name": "Kauffman Stadium",    "roof": False},
    "LAA": {"lat": 33.8003, "lon": -117.8827,"name": "Angel Stadium",       "roof": False},
    "LAD": {"lat": 34.0739, "lon": -118.2400,"name": "Dodger Stadium",      "roof": False},
    "MIA": {"lat": 25.7780, "lon": -80.2197, "name": "loanDepot park",      "roof": True},
    "MIL": {"lat": 43.0280, "lon": -87.9712, "name": "American Family",     "roof": True},
    "MIN": {"lat": 44.9817, "lon": -93.2773, "name": "Target Field",        "roof": False},
    "NYM": {"lat": 40.7571, "lon": -73.8458, "name": "Citi Field",          "roof": False},
    "NYY": {"lat": 40.8296, "lon": -73.9262, "name": "Yankee Stadium",      "roof": False},
    "OAK": {"lat": 37.7516, "lon": -122.2007,"name": "Oakland Coliseum",    "roof": False},
    "ATH": {"lat": 37.7516, "lon": -122.2007,"name": "Oakland Coliseum (ATH)","roof": False},
    "PHI": {"lat": 39.9061, "lon": -75.1665, "name": "Citizens Bank",       "roof": False},
    "PIT": {"lat": 40.4469, "lon": -80.0057, "name": "PNC Park",            "roof": False},
    "SD":  {"lat": 32.7073, "lon": -117.1566,"name": "Petco Park",          "roof": False},
    "SEA": {"lat": 47.5914, "lon": -122.3325,"name": "T-Mobile Park",       "roof": True},
    "SF":  {"lat": 37.7783, "lon": -122.3893,"name": "Oracle Park",         "roof": False},
    "STL": {"lat": 38.6226, "lon": -90.1928, "name": "Busch Stadium",       "roof": False},
    "TB":  {"lat": 27.7682, "lon": -82.6534, "name": "Tropicana Field",     "roof": True},
    "TEX": {"lat": 32.7473, "lon": -97.0841, "name": "Globe Life Field",    "roof": True},
    "TOR": {"lat": 43.6414, "lon": -79.3894, "name": "Rogers Centre",       "roof": True},
    "WSH": {"lat": 38.8730, "lon": -77.0074, "name": "Nationals Park",      "roof": False},
}

# Park orientation: degrees from home plate to center field.
# Wind blowing FROM this direction = wind blowing OUT to CF.
# Used to compute "wind out to CF" component.
PARK_ORIENTATION = {
    "ARI": 0,   "ATL": 0,   "BAL": 30,  "BOS": 60,
    "CHC": 30,  "CIN": 30,  "CLE": 30,  "COL": 30,
    "CWS": 30,  "DET": 30,  "HOU": 0,   "KC":  30,
    "LAA": 30,  "LAD": 30,  "MIA": 0,   "MIL": 0,
    "MIN": 30,  "NYM": 30,  "NYY": 60,  "OAK": 30,
    "ATH": 30,  "PHI": 30,  "PIT": 30,  "SD":  30,
    "SEA": 60,  "SF":  30,  "STL": 30,  "TB":  0,
    "TEX": 0,   "TOR": 0,   "WSH": 30,
}


def fetch_hourly_weather(park_code: str, lat: float, lon: float,
                          start_date: str, end_date: str,
                          force: bool = False) -> pd.DataFrame:
    """Fetch hourly weather from open-meteo for one park.

    Returns DataFrame with columns: park, time, temp_f, wind_speed_mph,
    wind_dir_deg, humidity_pct, precipitation_in, is_retractable_roof.
    """
    cache_path = CACHE_DIR / f"{park_code}_{start_date}_{end_date}.parquet"
    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m,"
                  "relative_humidity_2m,precipitation",
        "wind_speed_unit": "mph",
        "temperature_unit": "fahrenheit",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "America/New_York",
    }
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  Weather fetch failed for {park_code}: {e}")
        return pd.DataFrame()

    hourly = data.get("hourly", {})
    if not hourly or "time" not in hourly:
        return pd.DataFrame()

    park = MLB_PARKS.get(park_code, {})
    df = pd.DataFrame({
        "time": pd.to_datetime(hourly["time"]),
        "temp_f": hourly.get("temperature_2m", [None] * len(hourly["time"])),
        "wind_speed_mph": hourly.get("wind_speed_10m", [None] * len(hourly["time"])),
        "wind_dir_deg": hourly.get("wind_direction_10m", [None] * len(hourly["time"])),
        "humidity_pct": hourly.get("relative_humidity_2m", [None] * len(hourly["time"])),
        "precipitation_in": hourly.get("precipitation", [None] * len(hourly["time"])),
    })
    df["park"] = park_code
    df["is_retractable_roof"] = bool(park.get("roof", False))

    # Wind out to CF: positive = wind blowing out (helps HR)
    # If wind_dir_deg is the direction wind is coming FROM, then wind is
    # blowing TO (wind_dir_deg + 180) % 360. Compare to park orientation.
    park_orient = PARK_ORIENTATION.get(park_code, 30)
    wind_to = (df["wind_dir_deg"] + 180) % 360
    # Component of wind blowing in the park_orient direction
    # cos(angle_diff) > 0 means blowing out
    angle_diff = np.radians(wind_to - park_orient)
    df["wind_out_to_cf_mph"] = df["wind_speed_mph"] * np.cos(angle_diff)
    df["wind_out_flag"] = (df["wind_out_to_cf_mph"] > 0).astype(int)
    df["strong_wind_out_flag"] = (df["wind_out_to_cf_mph"] >= 15).astype(int)

    df.to_parquet(cache_path, index=False)
    return df


def fetch_weather_for_all_parks(start_date: str = None, end_date: str = None,
                                 force: bool = False) -> pd.DataFrame:
    """Fetch weather for all 30 MLB parks (next 7 days by default)."""
    if start_date is None:
        start_date = datetime.now().strftime("%Y-%m-%d")
    if end_date is None:
        end_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")

    frames = []
    for code, info in MLB_PARKS.items():
        df = fetch_hourly_weather(code, info["lat"], info["lon"],
                                    start_date, end_date, force=force)
        if not df.empty:
            frames.append(df)
        time.sleep(0.2)  # be nice to open-meteo
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    print("Fetching weather for all MLB parks (next 7 days)...")
    df = fetch_weather_for_all_parks(force=True)
    if df.empty:
        print("No data returned.")
    else:
        print(f"\nFetched {len(df):,} hourly records across {df['park'].nunique()} parks")
        print(f"  Date range: {df['time'].min()} -> {df['time'].max()}")
        print(f"  Avg wind speed: {df['wind_speed_mph'].mean():.1f} mph")
        print(f"  Pct with strong wind out (>=15 mph to CF): "
              f"{df['strong_wind_out_flag'].mean():.1%}")
        print(f"  Sample (first 5 rows):")
        print(df.head().to_string(index=False))
