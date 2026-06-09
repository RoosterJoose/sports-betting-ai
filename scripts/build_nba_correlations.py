#!/usr/bin/env python3
"""Build NBA stat-to-stat correlation database for parlay probability adjustment.

Computes pairwise Pearson correlations between NBA statistics from player
game log data.  Saves as JSON at models/nba/stat_correlations.json.

Usage:
    python scripts/build_nba_correlations.py
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config.settings import PROJECT_ROOT

DATA_PATH = PROJECT_ROOT / "data" / "nba_cache" / "game_logs_v14.parquet"
OUT_PATH = PROJECT_ROOT / "models" / "nba" / "stat_correlations.json"

# NBA stats that correspond to Kalshi prop markets
STAT_COLS = [
    "pts",
    "reb",
    "ast",
    "stl",
    "blk",
    "tov",
    "fg3m",
    "fg3a",
    "fgm",
    "fga",
    "ftm",
    "fta",
    "min",
]

# Combined stats computed from raw columns
COMPUTED_STATS = {
    "pra":  lambda df: df["pts"].fillna(0) + df["reb"].fillna(0) + df["ast"].fillna(0),
    "pa":   lambda df: df["pts"].fillna(0) + df["ast"].fillna(0),
    "pr":   lambda df: df["pts"].fillna(0) + df["reb"].fillna(0),
    "ra":   lambda df: df["reb"].fillna(0) + df["ast"].fillna(0),
    "fpts": lambda df: df["pts"].fillna(0) + df["reb"].fillna(0) * 1.2 + df["ast"].fillna(0) * 1.5
                       + df["stl"].fillna(0) * 3 + df["blk"].fillna(0) * 3 + df["tov"].fillna(0) * (-1),
}


def compute_correlations(df: pd.DataFrame, label: str) -> dict:
    """Compute pairwise Pearson correlations for all stat columns.

    Returns nested dict: {stat_a: {stat_b: ρ, ...}, ...}
    Only includes off-diagonal entries (i != j), sorted by absolute ρ.
    """
    corr_matrix = df.corr(method="pearson")

    result = {}
    for col_a in corr_matrix.columns:
        row = {}
        for col_b in corr_matrix.columns:
            if col_a == col_b:
                continue
            val = corr_matrix.loc[col_a, col_b]
            if not (np.isnan(val) or np.isinf(val)):
                row[col_b] = round(float(val), 4)
        if row:
            row = dict(sorted(row.items(), key=lambda x: abs(x[1]), reverse=True))
            result[col_a] = row

    print(f"  {label}: {len(df)} rows, {len(result)} stats", flush=True)
    return result


def main():
    print("Loading NBA game logs...", flush=True)
    df = pd.read_parquet(DATA_PATH)
    print(f"  {len(df):,} total rows", flush=True)
    print(f"  Columns: {[c for c in df.columns if c in STAT_COLS][:10]}...", flush=True)

    available = [c for c in STAT_COLS if c in df.columns]
    missing = [c for c in STAT_COLS if c not in df.columns]
    if missing:
        print(f"  WARNING: missing columns: {missing}", flush=True)

    # Add computed stats
    for name, fn in COMPUTED_STATS.items():
        try:
            df[name] = fn(df)
            available.append(name)
        except Exception as e:
            print(f"  WARNING: could not compute {name}: {e}", flush=True)

    print(f"  Computing correlations for {len(available)} stats", flush=True)

    # Build database
    db = {
        "metadata": {
            "source": "NBA player game logs (nba_api PlayerGameLogs)",
            "n_rows": len(df),
            "stat_columns": available,
        },
    }

    # 1. All players
    db["all_players"] = compute_correlations(df[available], "All players")

    # 2. Guards (PG, SG)
    if "position" in df.columns:
        guards = df[df["position"].str.upper().isin(["PG", "SG", "G"])].copy()
        db["guards"] = compute_correlations(guards[available], "Guards") if len(guards) > 100 else {}
    else:
        db["guards"] = {}

    # 3. Forwards (SF, PF, F)
    if "position" in df.columns:
        forwards = df[df["position"].str.upper().isin(["SF", "PF", "F"])].copy()
        db["forwards"] = compute_correlations(forwards[available], "Forwards") if len(forwards) > 100 else {}
    else:
        db["forwards"] = {}

    # 4. Centers (C)
    if "position" in df.columns:
        centers = df[df["position"].str.upper().isin(["C"])].copy()
        db["centers"] = compute_correlations(centers[available], "Centers") if len(centers) > 100 else {}
    else:
        db["centers"] = {}

    # 5. Starters only (higher volume -> more stable correlations)
    if "min" in df.columns:
        starters = df[df["min"].fillna(0) >= 20].copy()
        db["starters"] = compute_correlations(starters[available], "Starters (>=20 min)") if len(starters) > 100 else {}
    else:
        db["starters"] = {}

    # Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(db, f, indent=2)
    print(f"\nSaved {OUT_PATH}")
    print(f"  Groups: {[k for k in db.keys() if k != 'metadata']}", flush=True)

    # Print key correlations for parlay construction
    print(f"\n=== Key NBA correlations for parlay construction ===")
    important_pairs = [
        ("pts", "reb"), ("pts", "ast"), ("pts", "stl"), ("pts", "blk"),
        ("pts", "fg3m"), ("pts", "ftm"),
        ("reb", "ast"), ("reb", "stl"), ("reb", "blk"),
        ("ast", "stl"), ("ast", "tov"),
        ("stl", "blk"),
        ("pts", "pra"), ("pra", "pa"), ("pra", "pr"),
        ("fg3m", "fg3a"), ("fgm", "fga"), ("ftm", "fta"),
    ]
    for group_key in ["all_players", "guards", "forwards", "centers", "starters"]:
        group = db.get(group_key, {})
        if not group:
            continue
        print(f"\n  {group_key}:")
        for a, b in important_pairs:
            if a in group and b in group[a]:
                rho = group[a][b]
                print(f"    {a:10s} ↔ {b:10s}: ρ={rho:>6.3f}")


if __name__ == "__main__":
    main()
