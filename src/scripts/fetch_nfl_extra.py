#!/usr/bin/env python3
"""Fetch extra NFL data: weather (Open-Meteo), injuries, betting lines."""
import sys, warnings, requests, time
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config.settings import PROJECT_ROOT

CACHE_DIR = PROJECT_ROOT / "data" / "nfl_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

STADIUMS = {
    "ARI": (33.5276, -112.2626), "ATL": (33.7554, -84.4008),
    "BAL": (39.2780, -76.6225), "BUF": (42.7738, -78.7871),
    "CAR": (35.2258, -80.8528), "CHI": (41.8625, -87.6167),
    "CIN": (39.0954, -84.5161), "CLE": (41.5061, -81.6995),
    "DAL": (32.7473, -97.0922), "DEN": (39.7439, -105.0201),
    "DET": (42.3400, -83.0456), "GB": (44.5013, -88.0622),
    "HOU": (29.6847, -95.4109), "IND": (39.7601, -86.1639),
    "JAX": (30.3240, -81.6373), "KC": (39.0489, -94.4839),
    "LA": (33.9536, -118.3392), "LAC": (33.9536, -118.3392),
    "LV": (36.0905, -115.1836), "MIA": (25.9580, -80.2389),
    "MIN": (44.9738, -93.2580), "NE": (42.0909, -71.2644),
    "NO": (29.9509, -90.0812), "NYG": (40.8135, -74.0745),
    "NYJ": (40.8135, -74.0745), "PHI": (39.9008, -75.1675),
    "PIT": (40.4468, -80.0158), "SEA": (47.5952, -122.3316),
    "SF": (37.4033, -121.9700), "TB": (27.9759, -82.5033),
    "TEN": (36.1665, -86.7713), "WAS": (38.9077, -76.8645),
}
DOMES = {"ATL", "DAL", "DET", "HOU", "IND", "LV", "MIN", "NO", "LA", "LAC"}


def load_cached(path):
    if path.exists():
        return pd.read_parquet(path)
    return None


def fetch_schedule(seasons: list) -> pd.DataFrame:
    path = CACHE_DIR / "schedule.parquet"
    cached = load_cached(path)
    if cached is not None:
        print("  Loading cached schedule")
        return cached
    import nfl_data_py as nfl
    all_s = []
    for s in seasons:
        try:
            sch = nfl.import_schedules([s])
            if sch is not None and not sch.empty:
                all_s.append(sch)
                print(f"    {s}: {len(sch)} games")
        except Exception as e:
            print(f"    {s}: unavailable - {e}")
    if not all_s:
        return pd.DataFrame()
    sched = pd.concat(all_s, ignore_index=True)
    sched["game_date"] = pd.to_datetime(sched.get("gameday", sched.get("game_date")))
    sched.to_parquet(path)
    print(f"  Schedule: {len(sched)} games cached")
    return sched


def fetch_weather(seasons: list, sched: pd.DataFrame) -> pd.DataFrame:
    path = CACHE_DIR / "weather.parquet"
    cached = load_cached(path)
    if cached is not None:
        print("  Loading cached weather")
        return cached

    if sched is not None and not sched.empty and "home_team" in sched.columns:
        games = sched[["game_date", "home_team"]].drop_duplicates().copy()
    else:
        print("  WARNING: No schedule, weather may be inaccurate")
        return pd.DataFrame()

    games["game_date"] = pd.to_datetime(games["game_date"])
    games = games.dropna(subset=["home_team"])
    games = games[games["home_team"].isin(STADIUMS)]

    records = []
    url = "https://archive-api.open-meteo.com/v1/archive"

    for date in sorted(games["game_date"].unique()):
        ds = date.strftime("%Y-%m-%d")
        for team in games[games["game_date"] == date]["home_team"].unique():
            if team not in STADIUMS:
                continue
            lat, lon = STADIUMS[team]
            params = {"latitude": lat, "longitude": lon,
                      "start_date": ds, "end_date": ds,
                      "hourly": "temperature_2m,wind_speed_10m,precipitation,weather_code",
                      "timezone": "America/New_York"}
            try:
                resp = requests.get(url, params=params, timeout=15)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                if "hourly" not in data:
                    continue
                times = data["hourly"]["time"]
                temps = data["hourly"]["temperature_2m"]
                winds = data["hourly"]["wind_speed_10m"]
                precips = data["hourly"]["precipitation"]
                codes = data["hourly"]["weather_code"]
                best = 0
                for i, t in enumerate(times):
                    h = int(t.split("T")[1].split(":")[0])
                    if h == 13:
                        best = i
                        break
                    if abs(h - 13) < abs(int(times[best].split("T")[1].split(":")[0]) - 13):
                        best = i
                records.append({
                    "game_date": date, "home_team": team,
                    "temp_f": temps[best] if temps[best] is not None else np.nan,
                    "wind_mph": winds[best] if winds[best] is not None else np.nan,
                    "precip_mm": precips[best] if precips[best] is not None else 0.0,
                    "weather_code": codes[best] if codes[best] is not None else 0,
                })
            except Exception as e:
                print(f"    Weather err {team} {ds}: {e}")
            time.sleep(0.05)

    if not records:
        return pd.DataFrame()

    wdf = pd.DataFrame(records)
    wdf["is_dome"] = wdf["home_team"].isin(DOMES).astype(int)
    wdf["wind_severe"] = np.where(wdf["wind_mph"] >= 20, 2, np.where(wdf["wind_mph"] >= 15, 1, 0))
    rc = wdf["weather_code"]
    pr = wdf["precip_mm"]
    wdf["is_rain"] = ((rc >= 95) | ((rc >= 61) & (rc <= 67) & (pr > 0))).astype(int)
    wdf["is_snow"] = (((rc >= 85) & (rc <= 86)) | ((rc >= 71) & (rc <= 77))).astype(int)
    tf = wdf["temp_f"]
    wdf["is_freezing"] = (tf < 25).astype(int)
    wdf["is_cold"] = ((tf >= 25) & (tf < 50)).astype(int)
    adj = np.ones(len(wdf))
    out = wdf["is_dome"] == 0
    adj = np.where(out & (wdf["wind_mph"] >= 20), adj * 0.85, adj)
    adj = np.where(out & (wdf["wind_mph"] >= 15) & (wdf["wind_mph"] < 20), adj * 0.93, adj)
    adj = np.where(out & (wdf["wind_mph"] >= 10) & (wdf["wind_mph"] < 15), adj * 0.97, adj)
    adj = np.where(out & wdf["is_rain"].astype(bool), adj * 0.88, adj)
    adj = np.where(out & wdf["is_snow"].astype(bool), adj * 0.75, adj)
    adj = np.where(out & wdf["is_freezing"].astype(bool), adj * 0.92, adj)
    adj = np.where(out & wdf["is_cold"].astype(bool), adj * 0.95, adj)
    wdf["weather_adj"] = adj
    wdf.to_parquet(path)
    print(f"  Weather: {len(wdf)} records cached")
    return wdf


def fetch_injuries(seasons: list) -> pd.DataFrame:
    path = CACHE_DIR / "injuries.parquet"
    cached = load_cached(path)
    if cached is not None:
        print("  Loading cached injuries")
        return cached
    import nfl_data_py as nfl
    all_i = []
    for s in seasons:
        try:
            inj = nfl.import_injuries([s])
            if inj is not None and not inj.empty:
                all_i.append(inj)
                print(f"    {s}: {len(inj)} records")
        except Exception as e:
            print(f"    {s}: {e}")
    if not all_i:
        return pd.DataFrame()
    inj = pd.concat(all_i, ignore_index=True)
    inj["gsis_id"] = inj["gsis_id"].astype(str).str.strip()
    try:
        ids = nfl.import_ids()
        im = ids[ids["gsis_id"].notna()][["gsis_id", "name"]].copy()
        im["gsis_id"] = im["gsis_id"].astype(str).str.strip()
        inj = inj.merge(im, on="gsis_id", how="left")
    except:
        pass
    rs = inj["report_status"].str.lower()
    ps = inj["practice_status"].str.lower()
    inj["is_out"] = rs.str.contains("out", na=False).astype(int)
    inj["is_q"] = rs.str.contains("questionable", na=False).astype(int)
    inj["dnp"] = ps.str.contains("did not participate", na=False).astype(int)
    inj["limited_p"] = ps.str.contains("limited", na=False).astype(int)
    inj.to_parquet(path)
    print(f"  Injuries: {len(inj)} records cached")
    return inj


def fetch_betting(seasons: list, sched: pd.DataFrame) -> pd.DataFrame:
    path = CACHE_DIR / "betting_lines.parquet"
    cached = load_cached(path)
    if cached is not None:
        print("  Loading cached betting lines")
        return cached
    if sched is not None and not sched.empty:
        bc = [c for c in ["spread_line", "total_line", "home_spread_odds", "away_spread_odds",
                          "over_odds", "under_odds", "home_moneyline", "away_moneyline",
                          "game_date", "home_team", "away_team", "season", "week"]
              if c in sched.columns]
        if len(bc) >= 5:
            bet = sched[bc].copy()
            bet.to_parquet(path)
            print(f"  Betting: {len(bet)} records from schedule")
            return bet
    import nfl_data_py as nfl
    all_l = []
    for s in seasons:
        try:
            sc = nfl.import_sc_lines([s])
            if sc is not None and not sc.empty:
                all_l.append(sc)
                print(f"    {s}: {len(sc)} sc_lines")
                continue
        except:
            pass
        try:
            sd = nfl.import_seasonal_data([s])
            if sd is not None and not sd.empty:
                all_l.append(sd)
                print(f"    {s}: {len(sd)} seasonal")
        except:
            pass
    if not all_l:
        return pd.DataFrame()
    bet = pd.concat(all_l, ignore_index=True)
    bet.to_parquet(path)
    print(f"  Betting: {len(bet)} records cached")
    return bet


def main():
    print("Fetching NFL extra data...")
    weekly = load_cached(CACHE_DIR / "weekly.parquet")
    if weekly is None:
        print("No weekly data found")
        return
    seasons = sorted(weekly["season"].unique().tolist())
    print(f"Seasons: {seasons}")
    print()
    print("0. Schedule...")
    sched = fetch_schedule(seasons)
    print()
    print("1. Weather (Open-Meteo)...")
    weather = fetch_weather(seasons, sched)
    print()
    print("2. Injuries...")
    injuries = fetch_injuries(seasons)
    print()
    print("3. Betting lines...")
    betting = fetch_betting(seasons, sched)
    print()
    print("Done.")
    w = len(weather) if weather is not None and not weather.empty else 0
    i = len(injuries) if injuries is not None and not injuries.empty else 0
    b = len(betting) if betting is not None and not betting.empty else 0
    print(f"  Weather: {w} records")
    print(f"  Injuries: {i} records")
    print(f"  Betting: {b} records")

if __name__ == "__main__":
    main()
