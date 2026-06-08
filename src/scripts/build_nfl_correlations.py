#!/usr/bin/env python3
"""Build NFL stat-to-stat correlation database for parlay probability adjustment.

Computes pairwise Pearson correlations between NFL statistics from weekly
game log data.  Saves as JSON at models/nfl/stat_correlations.json in the
same format as the MLB correlation database.

Usage:
    python -m src.scripts.build_nfl_correlations
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config.settings import PROJECT_ROOT

DATA_PATH = PROJECT_ROOT / "data" / "nfl_cache" / "weekly.parquet"
OUT_PATH = PROJECT_ROOT / "models" / "nfl" / "stat_correlations.json"

# Stats that map to Kalshi NFL market types
# Internal column names from nfl_data_py
STAT_COLS = [
    "passing_yards",
    "passing_tds",
    "pass_attempts",    # derived from `attempts` column
    "interceptions",
    "rushing_yards",
    "rushing_tds",
    "carries",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "targets",
    "touchdowns",
]

# Additional computed stats for cross-market correlation
COMPUTED_STATS = {
    "rush_rec_yds": lambda df: df["rushing_yards"].fillna(0) + df["receiving_yards"].fillna(0),
    "pass_yds_td":  lambda df: df["passing_yards"].fillna(0) + df["passing_tds"].fillna(0) * 10,
}


def compute_correlations(df: pd.DataFrame, label: str) -> dict:
    """Compute pairwise Pearson correlations for all stat columns in df.

    Returns nested dict: {stat_a: {stat_b: ρ, ...}, ...}
    Only includes off-diagonal entries (i != j), sorted by absolute ρ.
    """
    # Compute Pearson correlation matrix
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
            # Sort by absolute correlation descending
            row = dict(sorted(row.items(), key=lambda x: abs(x[1]), reverse=True))
            result[col_a] = row

    print(f"  {label}: {len(df)} rows, {len(result)} stats", flush=True)
    return result


def main():
    print("Loading NFL weekly data...", flush=True)
    df = pd.read_parquet(DATA_PATH)
    print(f"  {len(df):,} total rows, {df['season'].min()}-{df['season'].max()}", flush=True)

    # Derive pass_attempts from attempts column
    if "attempts" in df.columns and "pass_attempts" not in df.columns:
        df["pass_attempts"] = df["attempts"].fillna(0)

    # Ensure we have the stat columns we need
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

    print(f"  Computing correlations for {len(available)} stats: {available}", flush=True)

    # Build database
    db = {
        "metadata": {
            "source": "NFL weekly game logs (nfl_data_py)",
            "n_rows": len(df),
            "seasons": [int(s) for s in sorted(df["season"].unique())],
            "stat_columns": available,
        },
    }

    # 1. All offensive players (exclude defensive / special teams)
    offensive = df[df["position"].isin(["QB", "RB", "WR", "TE", "FB"])].copy()
    db["all_offense"] = compute_correlations(offensive[available], "All offense")

    # 2. QBs only
    qbs = df[df["position"] == "QB"].copy()
    db["qbs"] = compute_correlations(qbs[available], "QBs")

    # 3. RBs only
    rbs = df[df["position"] == "RB"].copy()
    db["rbs"] = compute_correlations(rbs[available], "RBs")

    # 4. WRs only
    wrs = df[df["position"] == "WR"].copy()
    db["wrs"] = compute_correlations(wrs[available], "WRs")

    # 5. TEs only
    tes = df[df["position"] == "TE"].copy()
    db["tes"] = compute_correlations(tes[available], "TEs")

    # 6. Receivers (WR + TE)
    receivers = df[df["position"].isin(["WR", "TE"])].copy()
    db["receivers"] = compute_correlations(receivers[available], "Receivers")

    # Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(db, f, indent=2)
    print(f"\nSaved {OUT_PATH}")
    print(f"  Groups: {[k for k in db.keys() if k != 'metadata']}", flush=True)

    # Print key correlations
    print(f"\n=== Key correlations for parlay construction ===")
    important_pairs = [
        ("passing_yards", "passing_tds"),
        ("passing_yards", "interceptions"),
        ("rushing_yards", "rushing_tds"),
        ("rushing_yards", "receiving_yards"),
        ("receiving_yards", "receptions"),
        ("receiving_yards", "receiving_tds"),
        ("receptions", "receiving_tds"),
        ("passing_yards", "rushing_yards"),
        ("touchdowns", "passing_tds"),
        ("touchdowns", "rushing_tds"),
        ("touchdowns", "receiving_tds"),
        ("rush_rec_yds", "touchdowns"),
        ("rush_rec_yds", "receptions"),
    ]
    for group_key in ["all_offense", "qbs", "rbs", "wrs", "receivers"]:
        group = db.get(group_key, {})
        if not group:
            continue
        print(f"\n  {group_key}:")
        for a, b in important_pairs:
            if a in group and b in group[a]:
                rho = group[a][b]
                print(f"    {a:20s} ↔ {b:20s}: ρ={rho:>6.3f}")


if __name__ == "__main__":
    main()
