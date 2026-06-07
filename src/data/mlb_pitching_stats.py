"""
MLB Pitching Stats Fetcher — from MLB Stats API.

Fetches season-level pitching stats (K, BB, HR, IP) and computes:
  - FIP: (13*HR + 3*BB - 2*K) / IP + lgConstant
  - K/9, BB/9, HR/9, K-BB%

Caches results as parquet for fast reloading.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache" / "mlb" / "pitching_stats"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
}


def fetch_season_pitching_stats(season: int, min_ip: float = 30) -> pd.DataFrame:
    """Fetch season-level pitching stats for all qualified pitchers from MLB Stats API.
    
    Returns DataFrame with columns:
        player_id (MLBAM ID), player_name, season,
        ip, k, bb, hr, era, whip, fip, k9, bb9, hr9, k_bb_pct
    """
    cache_file = CACHE_DIR / f"pitching_{season}.parquet"
    if cache_file.exists():
        df = pd.read_parquet(cache_file)
        print(f"  Loaded cached pitching stats for {season} ({len(df)} pitchers)", flush=True)
        return df

    print(f"  Fetching pitching stats for {season} from MLB API...", flush=True)
    
    url = f"https://statsapi.mlb.com/api/v1/stats?stats=season&group=pitching&season={season}&playerPool=qualified&limit=400"
    
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            print(f"    MLB API returned {resp.status_code}", flush=True)
            return pd.DataFrame()
        
        data = resp.json()
        splits = data.get('stats', [{}])[0].get('splits', [])
    except Exception as e:
        print(f"    Error fetching data: {e}", flush=True)
        return pd.DataFrame()

    if not splits:
        print(f"    No pitchers found for {season}", flush=True)
        return pd.DataFrame()

    rows = []
    for sp in splits:
        s = sp.get('stat', {})
        player = sp.get('player', {})
        
        player_id = player.get('id')
        player_name = player.get('fullName', '')
        
        ip_str = str(s.get('inningsPitched', '0'))
        # Parse innings pitched (MLB API returns 123.1 for 123.1 IP)
        ip = 0.0
        try:
            if '.' in ip_str:
                whole, frac = ip_str.split('.')
                ip = float(whole) + float(frac) / 3.0
            else:
                ip = float(ip_str)
        except (ValueError, TypeError):
            continue
        
        if ip < min_ip:
            continue
        
        k = float(s.get('strikeOuts', 0) or 0)
        bb = float(s.get('baseOnBalls', 0) or 0)
        hr = float(s.get('homeRuns', 0) or 0)
        hbp = float(s.get('hitByPitch', 0) or 0)
        era = float(s.get('era', 0) or 0)
        whip = float(s.get('whip', 0) or 0)
        hits = float(s.get('hits', 0) or 0)
        games_started = int(s.get('gamesStarted', 0) or 0)
        
        rows.append({
            "player_id": player_id,
            "player_name": player_name,
            "season": season,
            "ip": round(ip, 1),
            "k": int(k),
            "bb": int(bb),
            "hr": int(hr),
            "hbp": int(hbp),
            "era": era,
            "whip": whip,
            "hits": int(hits),
            "games_started": games_started,
        })
    
    df = pd.DataFrame(rows)
    
    if df.empty:
        return df
    
    # Compute rate stats
    df["k9"] = df["k"] / df["ip"] * 9
    df["bb9"] = df["bb"] / df["ip"] * 9
    df["hr9"] = df["hr"] / df["ip"] * 9
    df["k_bb_pct"] = (df["k"] - df["bb"]) / df["ip"] * 9
    
    # Compute FIP constant (league average)
    total_hr = df["hr"].sum()
    total_bb = df["bb"].sum()
    total_k = df["k"].sum()
    total_ip = df["ip"].sum()
    avg_era = df["era"].mean()
    
    avg_fip_component = (13 * total_hr + 3 * total_bb - 2 * total_k) / total_ip
    fip_constant = avg_era - avg_fip_component
    
    df["fip"] = (13 * df["hr"] + 3 * df["bb"] - 2 * df["k"]) / df["ip"] + fip_constant
    df["fip_constant"] = round(fip_constant, 4)
    
    # Sort by FIP
    df = df.sort_values("fip").reset_index(drop=True)
    
    df.to_parquet(cache_file, index=False)
    print(f"  Saved {len(df)} pitchers for {season} (FIP constant={fip_constant:.3f})", flush=True)
    return df


def fetch_multiyear_pitching_stats(years: list[int], min_ip: float = 30) -> pd.DataFrame:
    """Fetch pitching stats for multiple seasons."""
    frames = []
    for year in years:
        df = fetch_season_pitching_stats(year, min_ip=min_ip)
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def merge_pitching_stats_into_pa(pa_df: pd.DataFrame, pitching_df: pd.DataFrame) -> pd.DataFrame:
    """Merge MLB pitching stats into PA DataFrame by matching pitcher IDs.
    
    Adds features: pitcher_fip, pitcher_k9, pitcher_bb9, pitcher_k_bb_pct
    """
    if pa_df.empty or pitching_df.empty:
        return pa_df
    
    result = pa_df.copy()
    
    # Match on pitcher (MLBAM ID in Statcast) = player_id (in pitching_df)
    # Both use the same MLBAM player IDs
    pitching_key = pitching_df[["player_id", "season", "fip", "k9", "bb9", "hr9", "k_bb_pct", "whip"]].copy()
    pitching_key["season"] = pitching_key["season"].astype(int)
    pitching_key["pitcher"] = pitching_key["player_id"]
    
    # Extract season from game_date (could be string or datetime)
    if hasattr(result["game_date"], "dt"):
        result["season_num"] = result["game_date"].dt.year.astype(int)
    else:
        result["season_num"] = result["game_date"].str[:4].astype(int)
    
    result = result.merge(
        pitching_key[["pitcher", "season", "fip", "k9", "bb9", "hr9", "k_bb_pct", "whip"]],
        left_on=["pitcher", "season_num"],
        right_on=["pitcher", "season"],
        how="left",
        suffixes=("", "_pitch")
    )
    
    # Fill missing FIP with average values
    median_fip = result["fip"].median() if "fip" in result.columns and result["fip"].notna().any() else 4.5
    median_k9 = result["k9"].median() if "k9" in result.columns and result["k9"].notna().any() else 8.0
    median_bb9 = result["bb9"].median() if "bb9" in result.columns and result["bb9"].notna().any() else 3.0
    median_hr9 = result["hr9"].median() if "hr9" in result.columns and result["hr9"].notna().any() else 1.2
    median_whip = result["whip"].median() if "whip" in result.columns and result["whip"].notna().any() else 1.3
    
    result["pitcher_fip"] = result["fip"].fillna(median_fip)
    result["pitcher_k9"] = result["k9"].fillna(median_k9)
    result["pitcher_bb9"] = result["bb9"].fillna(median_bb9)
    result["pitcher_hr9"] = result["hr9"].fillna(median_hr9)
    result["pitcher_whip"] = result["whip"].fillna(median_whip)
    result["pitcher_k_bb_pct"] = result["k_bb_pct"].fillna(0.0)
    
    # Clean up intermediate columns from the merge
    for c in ["fip", "k9", "bb9", "hr9", "k_bb_pct", "whip", "season", "season_num"]:
        if c in result.columns:
            result.drop(columns=[c], inplace=True)
    
    matched = result["pitcher_fip"].notna().sum()
    print(f"  Merged FIP data: {matched}/{len(result)} PAs matched ({matched/len(result):.1%})", flush=True)
    
    return result


if __name__ == "__main__":
    # Test: fetch 2024 + 2025 stats
    df = fetch_multiyear_pitching_stats([2024, 2025], min_ip=30)
    if not df.empty:
        print(f"\n=== PITCHING STATS SUMMARY ===")
        print(f"Total: {len(df)} pitcher-seasons")
        print(f"\nTop 10 by FIP (2025):")
        d25 = df[df["season"] == 2025].head(10)
        for _, r in d25.iterrows():
            print(f"  {r['player_name']:25s} FIP={r['fip']:.2f} K/9={r['k9']:.1f} BB/9={r['bb9']:.1f}")
        
        print(f"\nBottom 5 by FIP (2025):")
        d25_bottom = df[df["season"] == 2025].tail(5)
        for _, r in d25_bottom.iterrows():
            print(f"  {r['player_name']:25s} FIP={r['fip']:.2f} K/9={r['k9']:.1f} BB/9={r['bb9']:.1f}")
