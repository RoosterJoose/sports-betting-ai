#!/usr/bin/env python3
"""Fetch Statcast data for MLB games and cache per-player per-game aggregates."""
import sys, warnings, gc
warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config.settings import PROJECT_ROOT

CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "mlb"
STATCAST_DIR = CACHE_DIR / "statcast"
STATCAST_DIR.mkdir(parents=True, exist_ok=True)


def fetch_statcast_for_season(season: int) -> pd.DataFrame:
    """Fetch all Statcast data for a season, one 2-week chunk at a time."""
    from pybaseball import statcast

    cache_file = STATCAST_DIR / f"statcast_{season}.parquet"
    if cache_file.exists():
        print(f"  Loading cached Statcast for {season}...", flush=True)
        return pd.read_parquet(cache_file)

    print(f"  Fetching Statcast for {season}...", flush=True)
    all_dfs = []
    start_date = datetime(season, 3, 1)
    end_date = datetime(season, 11, 1)

    cursor = start_date
    while cursor < end_date:
        chunk_end = min(cursor + timedelta(days=14), end_date)
        s = cursor.strftime("%Y-%m-%d")
        e = chunk_end.strftime("%Y-%m-%d")
        print(f"    {s} to {e}...", flush=True)
        try:
            df = statcast(s, e)
            if df is not None and len(df) > 0:
                all_dfs.append(df)
        except Exception as ex:
            print(f"      Error: {ex}", flush=True)
        cursor = chunk_end
        gc.collect()

    if not all_dfs:
        print(f"  No data for {season}", flush=True)
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    result.to_parquet(cache_file)
    print(f"  Saved {len(result)} rows", flush=True)
    return result


def compute_player_agg(sc: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Statcast to per-player per-game stats."""
    if sc.empty:
        return pd.DataFrame()

    print(f"  Computing player-game aggregates...", flush=True)
    has_contact = sc["launch_speed"].notna() & sc["launch_angle"].notna()

    sc["barrel"] = 0
    sc.loc[has_contact & (sc["launch_speed"] >= 98) & (sc["launch_angle"] >= 26) & (sc["launch_angle"] <= 30), "barrel"] = 1
    sc["hard_hit"] = 0
    sc.loc[has_contact & (sc["launch_speed"] >= 95), "hard_hit"] = 1
    sc["sweet_spot"] = 0
    sc.loc[has_contact & (sc["launch_angle"] >= 8) & (sc["launch_angle"] <= 32), "sweet_spot"] = 1

    stats_list = []
    for (game_pk, batter), group in sc.groupby(["game_pk", "batter"]):
        n_pa = len(group)
        n_contact = group["launch_speed"].notna().sum()
        row = {
            "game_pk": game_pk,
            "batter_id": batter,
            "player_name": group["player_name"].iloc[0],
            "n_pa": n_pa,
            "n_contact": n_contact,
            "launch_speed_avg": group.loc[has_contact, "launch_speed"].mean() if n_contact > 0 else np.nan,
            "launch_angle_avg": group.loc[has_contact, "launch_angle"].mean() if n_contact > 0 else np.nan,
            "max_ev": group.loc[has_contact, "launch_speed"].max() if n_contact > 0 else np.nan,
            "barrel_rate": float(group["barrel"].sum() / n_pa) if n_pa > 0 else 0.0,
            "hard_hit_rate": float(group["hard_hit"].sum() / n_pa) if n_pa > 0 else 0.0,
            "sweet_spot_rate": float(group["sweet_spot"].sum() / n_pa) if n_pa > 0 else 0.0,
            "avg_hit_distance": group.loc[has_contact, "hit_distance_sc"].mean() if n_contact > 0 else np.nan,
            "xba": group["estimated_ba_using_speedangle"].mean() if "estimated_ba_using_speedangle" in group.columns else np.nan,
            "xslg": group["estimated_slg_using_speedangle"].mean() if "estimated_slg_using_speedangle" in group.columns else np.nan,
            "xwoba": group["estimated_woba_using_speedangle"].mean() if "estimated_woba_using_speedangle" in group.columns else np.nan,
        }
        stats_list.append(row)

    return pd.DataFrame(stats_list)


def add_rolling_features(pg: pd.DataFrame) -> pd.DataFrame:
    """Add season-to-date rolling averages."""
    if pg.empty:
        return pg
    result = pg.sort_values(["batter_id", "game_pk"])
    feats = ["launch_speed_avg", "launch_angle_avg", "max_ev",
             "barrel_rate", "hard_hit_rate", "sweet_spot_rate",
             "avg_hit_distance", "xba", "xslg", "xwoba"]
    for feat in feats:
        if feat not in result.columns:
            continue
        # Convert to float, coercing errors to NaN
        vals = pd.to_numeric(result[feat], errors='coerce')
        result[f"{feat}_avg5"] = vals.groupby(result["batter_id"]).transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        result[f"{feat}_avg15"] = vals.groupby(result["batter_id"]).transform(
            lambda x: x.shift(1).rolling(15, min_periods=1).mean())
        result[f"{feat}_ewm"] = vals.groupby(result["batter_id"]).transform(
            lambda x: x.shift(1).ewm(alpha=0.3, min_periods=1).mean())
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int, default=[2024, 2025, 2026])
    args = parser.parse_args()

    for season in args.seasons:
        print(f"\n=== Season {season} ===", flush=True)
        sc = fetch_statcast_for_season(season)
        if sc.empty:
            continue
        pg = compute_player_agg(sc)
        del sc; gc.collect()
        print(f"  {len(pg)} player-games", flush=True)
        pg_rolled = add_rolling_features(pg)
        out = STATCAST_DIR / f"statcast_agg_{season}.parquet"
        pg_rolled.to_parquet(out)
        print(f"  Saved to {out}", flush=True)

    print(f"\nDone!")


if __name__ == "__main__":
    main()
