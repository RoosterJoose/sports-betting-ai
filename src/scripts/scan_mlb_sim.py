#!/usr/bin/env python3
"""MLB Monte Carlo Player Prop Scanner.

Fetches today's schedule, builds pitcher/batter feature vectors from
cached Statcast data and MLB game logs, runs 500+ Monte Carlo simulations
per game, and outputs P(K >= X), P(TB >= Y), P(HR >= Z) for each player.

Usage:
    python -m src.scripts.scan_mlb_sim                          # today's games
    python -m src.scripts.scan_mlb_sim --date 2026-06-08        # specific date
    python -m src.scripts.scan_mlb_sim --sims 2000              # more sims
    python -m src.scripts.scan_mlb_sim --compare                # compare vs Kalshi lines
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.settings import PROJECT_ROOT
from src.mlb.mlb_simulator import MLBSimulator, DEFAULT_BATTER, DEFAULT_RELIEVER, compute_max_bf

MODEL_DIR = PROJECT_ROOT / "models" / "mlb"
CACHE_DIR = PROJECT_ROOT / "data/cache/mlb"
STATCAST_DIR = CACHE_DIR / "statcast"

# MLB Stats API team code → 3-letter code mapping
TEAM_MAP = {
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHC": "CHC", "CIN": "CIN", "CLE": "CLE", "COL": "COL",
    "CWS": "CWS", "DET": "DET", "HOU": "HOU", "KC": "KC",
    "LAA": "LAA", "LAD": "LAD", "MIA": "MIA", "MIL": "MIL",
    "MIN": "MIN", "NYM": "NYM", "NYY": "NYY", "OAK": "OAK",
    "PHI": "PHI", "PIT": "PIT", "SD": "SD", "SEA": "SEA",
    "SF": "SF", "STL": "STL", "TB": "TB", "TEX": "TEX",
    "TOR": "TOR", "WSH": "WSH",
}

# Mapping MLB Stats API team IDs to 3-letter codes
MLB_API_TEAM_IDS = {
    108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS",
    112: "CHC", 113: "CIN", 114: "CLE", 115: "COL",
    116: "DET", 117: "HOU", 118: "KC", 119: "LAD",
    120: "WSH", 121: "NYM", 133: "OAK", 134: "PIT",
    135: "SD", 136: "SEA", 137: "SF", 138: "STL",
    139: "TB", 140: "TEX", 141: "TOR", 142: "MIN",
    143: "PHI", 144: "ATL", 145: "CWS", 146: "MIA",
    147: "NYY", 158: "MIL",
}

# Park factors for K and HR
PARK_FACTOR_K = {
    "SD": 1.08, "SEA": 1.06, "NYM": 1.04, "MIA": 1.03, "CLE": 1.02,
    "OAK": 1.02, "TB": 1.01, "SF": 1.01, "WSH": 1.00, "DET": 1.00,
    "MIL": 0.99, "BAL": 0.99, "KC": 0.99, "MIN": 0.99, "PIT": 0.99,
    "LAA": 0.98, "PHI": 0.98, "CIN": 0.98, "ATL": 0.97, "CHC": 0.97,
    "TEX": 0.97, "BOS": 0.97, "TOR": 0.96, "HOU": 0.96, "STL": 0.96,
    "AZ": 0.95, "NYY": 0.95, "LAD": 0.94, "CWS": 0.93, "COL": 0.88,
}

PARK_FACTOR_HR = {
    "COL": 1.28, "CIN": 1.14, "BOS": 1.12, "NYY": 1.10, "BAL": 1.09,
    "CHC": 1.07, "MIL": 1.06, "MIN": 1.05, "TEX": 1.04, "HOU": 1.04,
    "CLE": 1.04, "LAA": 1.03, "PHI": 1.03, "AZ": 1.02, "TB": 1.01,
    "ATL": 1.01, "WSH": 1.00, "DET": 1.00, "KC": 1.00, "STL": 1.00,
    "PIT": 0.99, "MIA": 0.98, "SEA": 0.98, "LAD": 0.97,
    "SD": 0.96, "SF": 0.95, "OAK": 0.95, "TOR": 0.94, "NYM": 0.93,
}


# ── Schedule fetching ────────────────────────────────────────────

def fetch_schedule(date_str: str | None = None) -> list[dict]:
    """Fetch MLB schedule with probable pitchers for a given date."""
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    url = f"https://statsapi.mlb.com/api/v1/schedule"
    params = {
        "sportId": 1,
        "date": date_str,
        "hydrate": "probablePitcher,team,lineup",
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  Schedule fetch error: {e}")
        return []

    games = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            if game.get("status", {}).get("codedState") == "I":
                continue  # Skip in-progress games

            teams = game.get("teams", {})
            away = teams.get("away", {}).get("team", {})
            home = teams.get("home", {}).get("team", {})

            away_id = away.get("id")
            home_id = home.get("id")
            away_abbr = MLB_API_TEAM_IDS.get(away_id, "")
            home_abbr = MLB_API_TEAM_IDS.get(home_id, "")

            away_p = teams.get("away", {}).get("probablePitcher", {})
            home_p = teams.get("home", {}).get("probablePitcher", {})

            games.append({
                "game_pk": game.get("gamePk"),
                "away_team": away_abbr,
                "home_team": home_abbr,
                "away_pitcher_name": away_p.get("fullName", "") if away_p else "",
                "away_pitcher_id": away_p.get("id") if away_p else None,
                "home_pitcher_name": home_p.get("fullName", "") if home_p else "",
                "home_pitcher_id": home_p.get("id") if home_p else None,
                "start_time": game.get("gameDate", ""),
                "venue": game.get("venue", {}).get("name", ""),
            })

    return games


# ── Feature builders ─────────────────────────────────────────────

def load_team_bullpen_profiles(
    game_logs_df: pd.DataFrame | None,
    pitching_df: pd.DataFrame,
    team_abbr: str,
) -> dict[str, float] | None:
    """Build a team-specific bullpen profile from game logs and pitching stats.

    Looks at all reliever appearances for this team (position="P", gs=0 or
    missing gs) and computes average K rate, BB rate, HR rate, FIP, K/9, etc.

    Falls back to DEFAULT_RELIEVER if insufficient data.
    """
    bp = dict(DEFAULT_RELIEVER)

    if game_logs_df is not None and not game_logs_df.empty and "team_abbr" in game_logs_df.columns:
        # Get team's pitchers who are NOT starters (gs=0 or gs is missing)
        team_pitchers = game_logs_df[
            (game_logs_df["team_abbr"].str.upper() == team_abbr.upper()) &
            (game_logs_df["position"] == "P")
        ]
        if "gs" in team_pitchers.columns:
            relievers = team_pitchers[team_pitchers["gs"].fillna(0) == 0]
        else:
            relievers = team_pitchers

        if len(relievers) >= 10:
            if "so" in relievers.columns and "bf" in relievers.columns:
                total_bf = relievers["bf"].sum()
                if total_bf > 0:
                    k_rate = relievers["so"].sum() / total_bf
                    bp["pitcher_k_rate_prior"] = min(max(k_rate, 0.10), 0.45)
            if "bb" in relievers.columns and "bf" in relievers.columns:
                total_bf = relievers["bf"].sum()
                if total_bf > 0:
                    bb_rate = relievers["bb"].sum() / total_bf
                    bp["pitcher_bb_rate_prior"] = min(max(bb_rate, 0.03), 0.20)
            if "hr" in relievers.columns and "bf" in relievers.columns:
                total_bf = relievers["bf"].sum()
                if total_bf > 0:
                    hr_rate = relievers["hr"].sum() / total_bf
                    bp["pitcher_hr_rate_prior"] = min(max(hr_rate, 0.005), 0.08)

    # Enrich with team pitching stats from pitching_df (team-level averages)
    if pitching_df is not None and not pitching_df.empty and "team_abbr" in pitching_df.columns:
        team_stats = pitching_df[pitching_df["team_abbr"].str.upper() == team_abbr.upper()]
        if not team_stats.empty:
            # Use mean stats across all pitchers on the team
            agg = team_stats.agg({
                "k9": "mean", "bb9": "mean", "hr9": "mean",
                "fip": "mean", "whip": "mean", "k_bb_pct": "mean",
            })
            if pd.notna(agg.get("k9")):
                bp["pitcher_k9"] = float(agg["k9"])
            if pd.notna(agg.get("fip")):
                bp["pitcher_fip"] = float(agg["fip"])

    return bp


def load_statcast_agg() -> pd.DataFrame:
    """Load the most recent Statcast aggregate data with batter rolling features."""
    files = sorted(STATCAST_DIR.glob("statcast_agg_*.parquet"))
    if not files:
        print("  No Statcast aggregate data found")
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"  Loaded {len(df)} Statcast agg rows", flush=True)
    return df


def load_raw_statcast_pitcher_agg() -> pd.DataFrame:
    """Load raw Statcast data and aggregate per-pitcher per-game.

    Raw Statcast has a ``pitcher`` column (MLBAM ID).  We compute
    per-game averages for EV against, LA against, hard-hit rate, etc.,
    then build rolling averages per pitcher (like the batter agg does).
    """
    files = sorted(STATCAST_DIR.glob("statcast_202*.parquet"))
    if not files:
        print("  No raw Statcast data found for pitcher agg")
        return pd.DataFrame()

    # Only load the most recent 2 seasons (substantial data)
    recent = files[-2:]
    print(f"  Loading raw Statcast ({[f.name for f in recent]})...", end=" ", flush=True)
    chunks = []
    for f in recent:
        df = pd.read_parquet(f, columns=[
            "pitcher", "game_pk", "game_date", "player_name",
            "launch_speed", "launch_angle", "events", "description", "bb_type",
        ])
        chunks.append(df)
    raw = pd.concat(chunks, ignore_index=True)
    print(f"{len(raw)} PAs", flush=True)

    if raw.empty:
        return pd.DataFrame()

    # Filter to batted-ball events only
    has_contact = raw["launch_speed"].notna() & raw["launch_angle"].notna()

    # Compute per-pitcher per-game aggregates
    agg_list = []
    for (pitcher_id, game_pk), group in raw.groupby(["pitcher", "game_pk"]):
        n_pa = len(group)
        n_contact = int(group["launch_speed"].notna().sum())
        contact_mask = has_contact.loc[group.index]
        contact_group = group[contact_mask]

        # Batted ball type rates
        n_gb = int((group["bb_type"] == "ground_ball").sum()) if "bb_type" in group.columns else 0
        n_fb = int((group["bb_type"].str.contains("fly_ball", na=False)).sum()) if "bb_type" in group.columns else 0
        n_ld = int((group["bb_type"] == "line_drive").sum()) if "bb_type" in group.columns else 0

        row = {
            "pitcher": pitcher_id,
            "game_pk": game_pk,
            "game_date": group["game_date"].iloc[0] if "game_date" in group.columns else None,
            "n_pa_faced": n_pa,
            "launch_speed_avg": float(contact_group["launch_speed"].mean()) if n_contact > 0 else None,
            "launch_angle_avg": float(contact_group["launch_angle"].mean()) if n_contact > 0 else None,
            "hard_hit_rate": float((contact_group["launch_speed"] >= 95).sum() / n_pa) if n_pa > 0 else 0.0,
            "barrel_rate": float(
                ((contact_group["launch_speed"] >= 98) &
                 (contact_group["launch_angle"] >= 26) &
                 (contact_group["launch_angle"] <= 30)).sum() / n_pa
            ) if n_pa > 0 else 0.0,
            "gb_rate": n_gb / max(n_pa, 1),
            "fb_rate": n_fb / max(n_pa, 1),
            "ld_rate": n_ld / max(n_pa, 1),
            "k_rate_raw": float((group["events"] == "strikeout").sum() / n_pa) if n_pa > 0 else 0.0,
            "bb_rate_raw": float((group["events"] == "walk").sum() / n_pa) if n_pa > 0 else 0.0,
        }
        agg_list.append(row)

    pitcher_game = pd.DataFrame(agg_list)
    if pitcher_game.empty:
        return pitcher_game

    # Build rolling features per pitcher (game-by-game)
    pitcher_game = pitcher_game.sort_values(["pitcher", "game_pk"])
    roll_cols = ["launch_speed_avg", "launch_angle_avg", "hard_hit_rate",
                 "barrel_rate", "gb_rate", "fb_rate", "ld_rate",
                 "k_rate_raw", "bb_rate_raw"]
    for col in roll_cols:
        if col not in pitcher_game.columns:
            continue
        vals = pd.to_numeric(pitcher_game[col], errors="coerce")
        pitcher_game[f"{col}_ewm"] = vals.groupby(pitcher_game["pitcher"]).transform(
            lambda x: x.shift(1).ewm(alpha=0.3, min_periods=1).mean()
        )

    print(f"  Pitcher agg: {len(pitcher_game)} pitcher-games, "
          f"{pitcher_game['pitcher'].nunique()} pitchers", flush=True)
    return pitcher_game


def load_pitching_stats() -> pd.DataFrame:
    """Load cached pitcher FIP/K9/BB9 data."""
    f = CACHE_DIR / "pitching_stats_2024_2025_2026.parquet"
    if not f.exists():
        # Try single-year files
        files = sorted(CACHE_DIR.glob("pitching_stats_*.parquet"))
        if not files:
            return pd.DataFrame()
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    else:
        df = pd.read_parquet(f)
    print(f"  Loaded {len(df)} pitching stats rows", flush=True)
    return df


# ── Handedness lookup (loaded from raw Statcast) ─────────────────

_HANDEDNESS_CACHE: dict[str, str] = {}  # player_id -> "R" or "L"
_OF_HANDEDNESS_CACHE: dict[str, str] = {}  # opposite: batter_id -> "R" or "L"

def _load_handedness() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load pitcher and batter handedness from raw Statcast."""
    if _HANDEDNESS_CACHE:
        return _HANDEDNESS_CACHE, _OF_HANDEDNESS_CACHE

    files = sorted(STATCAST_DIR.glob("statcast_202*.parquet"))
    if not files:
        return {}, {}

    # Load most recent season's hand data (one row per pitcher/batter is enough)
    df = pd.read_parquet(files[-1], columns=["pitcher", "p_throws", "batter", "stand"])
    if not df.empty:
        # Per-pitcher: take the first p_throws value (should be consistent)
        pitcher_hand = df.groupby("pitcher")["p_throws"].first().to_dict()
        _HANDEDNESS_CACHE.update({str(k): v for k, v in pitcher_hand.items()})
        # Per-batter: take the first stand value
        batter_hand = df.groupby("batter")["stand"].first().to_dict()
        _OF_HANDEDNESS_CACHE.update({str(k): v for k, v in batter_hand.items()})

    print(f"  Loaded handedness: {len(_HANDEDNESS_CACHE)} pitchers, "
          f"{len(_OF_HANDEDNESS_CACHE)} batters", flush=True)
    return _HANDEDNESS_CACHE, _OF_HANDEDNESS_CACHE


def build_pitcher_features(
    pitcher_name: str,
    pitcher_id: int | None,
    statcast_pitcher_df: pd.DataFrame,
    pitching_df: pd.DataFrame,
    game_logs_df: pd.DataFrame | None = None,
    pitcher_hand_map: dict[str, str] | None = None,
) -> dict:
    """Build the feature vector for a starting pitcher.

    Combines data from raw Statcast pitcher aggregates (EV against, LA against,
    hard-hit rate), MLB pitching stats (FIP, K/9, BB/9), and game logs
    (rolling K/BB/HR rates). Includes handedness (p_throws).
    """
    features = dict(DEFAULT_RELIEVER)
    features["name"] = pitcher_name

    if pitcher_id is None:
        return features

    pid_str = str(int(pitcher_id))

    # ── Handedness ──
    if pitcher_hand_map is not None and pid_str in pitcher_hand_map:
        features["p_throws"] = pitcher_hand_map[pid_str]

    # ── Raw Statcast: pitcher's rolling stats (what batters do against them) ──
    if not statcast_pitcher_df.empty and "pitcher" in statcast_pitcher_df.columns:
        try:
            p_rows = statcast_pitcher_df[statcast_pitcher_df["pitcher"] == int(pid_str)]
        except (ValueError, TypeError):
            p_rows = pd.DataFrame()
        if not p_rows.empty:
            last = (p_rows.sort_values("game_pk").iloc[-1]
                    if "game_pk" in p_rows.columns else p_rows.iloc[-1])
            for key, col in [
                ("pitcher_avg_ev_against", "launch_speed_avg_ewm"),
                ("pitcher_avg_la_against", "launch_angle_avg_ewm"),
                ("pitcher_hard_hit_against", "hard_hit_rate_ewm"),
                ("pitcher_gb_rate_against", "gb_rate_ewm"),
                ("pitcher_fb_rate_against", "fb_rate_ewm"),
            ]:
                if col in last.index and pd.notna(last[col]):
                    features[key] = float(last[col])

    # ── MLB Pitching Stats (FIP, K/9, BB/9, HR/9) ──
    if not pitching_df.empty:
        if "player_id" in pitching_df.columns:
            match = pitching_df[pitching_df["player_id"].astype(str) == pid_str]
        else:
            match = pd.DataFrame()

        if match.empty and "player_name" in pitching_df.columns:
            match = pitching_df[
                pitching_df["player_name"].str.contains(
                    pitcher_name.split()[-1], case=False, na=False
                )
            ]

        if not match.empty:
            last = match.sort_values("season").iloc[-1] if "season" in match.columns else match.iloc[-1]
            for key, col in [
                ("pitcher_fip", "fip"),
                ("pitcher_k9", "k9"),
                ("pitcher_bb9", "bb9"),
                ("pitcher_hr9", "hr9"),
                ("pitcher_whip", "whip"),
                ("pitcher_k_bb_pct", "k_bb_pct"),
            ]:
                if col in last.index and pd.notna(last[col]):
                    features[key] = float(last[col])

    # ── Game logs: rolling K/BB/HR rates + K/9 fallback ──
    if game_logs_df is not None and not game_logs_df.empty:
        if "player_id" in game_logs_df.columns:
            gl = game_logs_df[game_logs_df["player_id"].astype(str) == pid_str]
            if not gl.empty:
                gl = gl.sort_values("game_date")
                bf_col = "bf" if "bf" in gl.columns else "outs"
                if "so" in gl.columns and bf_col in gl.columns:
                    recent_bf = gl[bf_col].iloc[-min(14, len(gl)):].sum()
                    if recent_bf > 0:
                        k_rate = gl["so"].iloc[-min(14, len(gl)):].sum() / recent_bf
                        features["pitcher_k_rate_prior"] = min(max(k_rate, 0.01), 0.50)
                        # Compute K/9 from game logs if pitching_df didn't provide it
                        # K/9 = K per PA * 27 outs
                        if features.get("pitcher_k9", 8.5) == 8.5 and "outs" in gl.columns:
                            total_outs = gl["outs"].sum()
                            if total_outs > 0:
                                k9_from_logs = gl["so"].sum() / total_outs * 27
                                features["pitcher_k9"] = min(max(k9_from_logs, 3.0), 15.0)
                if "bb" in gl.columns:
                    recent_bf = gl[bf_col].iloc[-min(14, len(gl)):].sum()
                    if recent_bf > 0:
                        bb_rate = gl["bb"].iloc[-min(14, len(gl)):].sum() / recent_bf
                        features["pitcher_bb_rate_prior"] = min(max(bb_rate, 0.01), 0.25)
                if "hr" in gl.columns:
                    recent_bf = gl[bf_col].iloc[-min(14, len(gl)):].sum()
                    if recent_bf > 0:
                        hr_rate = gl["hr"].iloc[-min(14, len(gl)):].sum() / recent_bf
                        features["pitcher_hr_rate_prior"] = min(max(hr_rate, 0.001), 0.10)

    return features


def _resolve_batter_id(row) -> int | None:
    """Get the numeric batter ID from a data row, regardless of column name."""
    for col in ["batter", "batter_id", "player_id"]:
        if col in row.index:
            v = row[col]
            if pd.notna(v):
                try:
                    return int(float(v))
                except (ValueError, TypeError):
                    pass
    return None


def build_batter_features(
    batter_name: str,
    batter_id: int | None,
    statcast_df: pd.DataFrame,
    batter_hand_map: dict[str, str] | None = None,
) -> dict:
    """Build the feature vector for a batter from Statcast aggregates.

    Uses the most recent rolling averages (EWMA preferred, then avg15, then avg5).
    Includes batter handedness (stand).
    """
    features = dict(DEFAULT_BATTER)
    features["name"] = batter_name

    if batter_id is None:
        return features

    bid = int(batter_id)
    bid_str = str(bid)

    # ── Handedness ──
    if batter_hand_map is not None and bid_str in batter_hand_map:
        features["stand"] = batter_hand_map[bid_str]

    if statcast_df.empty:
        return features

    # Statcast agg data uses 'batter' column (int)
    id_col = None
    for col in ["batter", "batter_id"]:
        if col in statcast_df.columns:
            id_col = col
            break

    if id_col is None:
        return features

    # Match on int ID (avoid dtype issues)
    try:
        b_rows = statcast_df[statcast_df[id_col].astype(int) == bid]
    except (ValueError, TypeError):
        b_rows = pd.DataFrame()

    if b_rows.empty:
        # Try matching by name
        name_parts = batter_name.lower().split()
        if len(name_parts) >= 2 and "player_name" in statcast_df.columns:
            b_rows = statcast_df[
                statcast_df["player_name"].str.lower().str.contains(
                    name_parts[-1], na=False
                )
            ]
    if b_rows.empty:
        return features

    last = b_rows.sort_values("game_date").iloc[-1] if "game_date" in b_rows.columns else b_rows.iloc[-1]

    # Map Statcast columns to PA model features
    # EWMA preferred (most recent), then avg15, then avg5
    col_map = {
        "batter_k_rate_prior": ["k_rate_ewm", "k_rate_avg15", "k_rate_avg5"],
        "batter_bb_rate_prior": ["bb_rate_ewm", "bb_rate_avg15", "bb_rate_avg5"],
        "batter_avg_ev_prior": ["launch_speed_avg_ewm", "launch_speed_avg_avg15", "launch_speed_avg_avg5"],
        "batter_avg_la_prior": ["launch_angle_avg_ewm", "launch_angle_avg_avg15", "launch_angle_avg_avg5"],
        "batter_hard_hit_rate_prior": ["hard_hit_rate_ewm", "hard_hit_rate_avg15", "hard_hit_rate_avg5"],
        "batter_gb_rate_prior": ["gb_rate_ewm", "gb_rate_avg15", "gb_rate_avg5"],
        "batter_fb_rate_prior": ["fb_rate_ewm", "fb_rate_avg15", "fb_rate_avg5"],
        "batter_ld_rate_prior": ["ld_rate_ewm", "ld_rate_avg15", "ld_rate_avg5"],
    }

    for feature_key, col_options in col_map.items():
        for col in col_options:
            if col in last.index and pd.notna(last[col]):
                features[feature_key] = float(last[col])
                break

    return features


def build_team_roster_from_game_logs(
    game_logs_df: pd.DataFrame,
    team_abbr: str,
    n_batters: int = 9,
) -> list[dict]:
    """Build a lineup for a team from game logs (has team_abbr via feature engineering).

    Finds the most recent batters (position != 'P') for this team,
    ordered by games played (most frequent = likely starter).
    Falls back to default batters if no data.
    """
    batters = []

    if game_logs_df is None or game_logs_df.empty:
        return batters

    # Filter to this team's hitters (non-pitcher)
    team_upper = team_abbr.upper()
    if "team_abbr" in game_logs_df.columns:
        hitters = game_logs_df[
            (game_logs_df["team_abbr"].str.upper() == team_upper) &
            (game_logs_df["position"] != "P")
        ]
    else:
        return batters

    if hitters.empty:
        return batters

    # Get the most recent game date for this team
    if "game_date" in hitters.columns:
        max_date = pd.to_datetime(hitters["game_date"]).max()
        # Get games within the last 14 days, preferring most recent
        cutoff = max_date - pd.Timedelta(days=14)
        recent = hitters[pd.to_datetime(hitters["game_date"]) >= cutoff]
    else:
        recent = hitters

    # Count games per player and get the most frequent starters
    if "player_id" in recent.columns and "game_date" in recent.columns:
        # Count unique game appearances per player
        player_games = recent.groupby("player_id").agg(
            n_games=("game_date", "nunique"),
            player_name=("player_name", "first"),
            last_game=("game_date", "max"),
        ).sort_values(["n_games", "last_game"], ascending=[False, False])

        for pid, p_row in player_games.head(n_batters).iterrows():
            name = p_row.get("player_name", f"Player {pid}")
            batter_id = int(float(pid)) if pid is not None else None
            # Build features from Statcast (passed separately)
            feats = dict(DEFAULT_BATTER)
            feats["name"] = name
            feats["_player_id"] = batter_id
            batters.append(feats)

    return batters


def build_lineup(
    team_abbr: str,
    statcast_df: pd.DataFrame,
    roster_batters: list[dict] | None = None,
    n_batters: int = 9,
    batter_hand_map: dict[str, str] | None = None,
) -> list[dict]:
    """Build a batting order of N batters for a team.

    Uses the roster (built from game logs) to identify which batters play
    for this team, then enriches each with Statcast aggregate features
    and handedness data. Falls back to Statcast-based matching when no
    game log roster is available.
    """
    batters = []

    # Method 1: Use roster from game logs (has real player IDs + team info)
    if roster_batters and len(roster_batters) >= 3:
        for b in roster_batters[:n_batters]:
            name = b.get("name", "Unknown")
            bid = b.get("_player_id")
            feats = build_batter_features(name, bid, statcast_df, batter_hand_map)
            if feats:
                batters.append(feats)

    # Method 2: Fallback — use Statcast data (only works if team_abbr exists there)
    if len(batters) < n_batters and not statcast_df.empty and "team_abbr" in statcast_df.columns:
        team_rows = statcast_df[
            statcast_df["team_abbr"].str.upper() == team_abbr.upper()
        ]
        if "n_pa" in team_rows.columns:
            team_rows = team_rows.sort_values("n_pa", ascending=False)
        for _, row in team_rows.head(n_batters).iterrows():
            bname = row.get("player_name", "")
            bid = _resolve_batter_id(row)
            feats = build_batter_features(bname, bid, statcast_df, batter_hand_map)
            if feats:
                batters.append(feats)

    # Fill remaining slots with defaults
    while len(batters) < n_batters:
        b = dict(DEFAULT_BATTER)
        b["name"] = f"Batter #{len(batters) + 1}"
        batters.append(b)

    return batters[:n_batters]


# ── Display ──────────────────────────────────────────────────────

def display_player_probs(
    game_info: dict,
    result: dict,
    n_sims: int,
) -> None:
    """Display player prop probabilities from simulation results."""
    away = game_info["away_team"]
    home = game_info["home_team"]
    start = game_info.get("start_time", "")[:16].replace("T", " ")
    venue = game_info.get("venue", "")

    print(f"\n{'='*75}")
    print(f"  {away} @ {home}  [{start}]  {venue}")
    print(f"  {n_sims} simulations")
    print(f"{'='*75}")

    # Pitcher K props — show both empirical and distribution-based
    for side, pitcher_key in [("AWAY", "away_pitcher"), ("HOME", "home_pitcher")]:
        p = result[pitcher_key]
        name = p["name"]
        k_mean = p["k_mean"]
        k_probs = p["k_probs"]
        k_probs_dist = p.get("k_probs_dist", {})

        print(f"\n  {side:4s} SP: {name}")
        print(f"  {'':4s} K/game: {k_mean:.1f}")

        # Show empirical probabilities
        print(f"  {'':4s} P(K) [empirical ]:", end="")
        for line in [3, 4, 5, 6, 7, 8, 9, 10]:
            prob = k_probs.get(line, 0)
            if prob > 0.01:
                print(f"  ≥{line}: {prob:.0%}", end="")
        print()

        # Show distribution-based probabilities (NB/Poisson-smoothed)
        print(f"  {'':4s} P(K) [NB-smooth]:", end="")
        for line in [3, 4, 5, 6, 7, 8, 9, 10]:
            prob = k_probs_dist.get(line, 0)
            if prob > 0.01:
                print(f"  ≥{line}: {prob:.0%}", end="")
        print()

    # Batter TB/HR/H props (top 3 by TB mean)
    for side_key, label in [("away_batters", "AWAY"), ("home_batters", "HOME")]:
        batters = result[side_key]
        # Sort by TB mean descending
        sorted_batters = sorted(batters, key=lambda b: -b.get("tb_mean", 0))

        print(f"\n  {label:4s} Batters (top 3 by TB):")
        for b in sorted_batters[:3]:
            name = b["name"]
            tb_mean = b.get("tb_mean", 0)
            hr_mean = b.get("hr_mean", 0)
            h_mean = b.get("h_mean", 0)
            tb_prob = b.get("tb_probs", {}).get(1, 0)
            hr_prob = b.get("hr_probs", {}).get(1, 0)

            print(f"  {'':4s} {name[:25]:25s} TB={tb_mean:.2f}  H={h_mean:.2f}  "
                  f"HR={hr_mean:.2f}")
            print(f"  {'':4s} {'':25s} P(TB≥1)={tb_prob:.0%}  "
                  f"P(HR≥1)={hr_prob:.0%}")


# ── Main ─────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="MLB Monte Carlo Sim Scanner")
    parser.add_argument("--date", type=str, default=None, help="Date (YYYY-MM-DD)")
    parser.add_argument("--sims", type=int, default=500, help="Simulations per game")
    parser.add_argument("--compare", action="store_true", help="Compare vs Kalshi lines (NYI)")
    args = parser.parse_args()

    print(f"MLB Monte Carlo Sim Scanner — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")

    # ── 1. Load simulator ──
    print("\n1. Loading PA outcome model...")
    try:
        sim = MLBSimulator()
    except FileNotFoundError as e:
        print(f"  {e}")
        return

    # ── 2. Fetch schedule ──
    print(f"\n2. Fetching schedule for {args.date or 'today'}...")
    games = fetch_schedule(args.date)
    if not games:
        print("  No games found.")
        return
    print(f"  {len(games)} games")

    # ── 3. Load data ──
    print("\n3. Loading feature data...")
    statcast_df = load_statcast_agg()
    statcast_pitcher_df = load_raw_statcast_pitcher_agg()
    pitching_df = load_pitching_stats()

    # Load game logs for pitcher rolling stats AND team roster building
    cache_files = sorted(CACHE_DIR.glob("game_logs_*.parquet"))
    game_logs_df = None
    if cache_files:
        columns_needed = ["player_id", "game_date", "so", "bb", "hr", "bf", "outs",
                          "position", "team_id", "player_name", "gs"]
        # Check available columns via schema (reads metadata only, no data)
        import pyarrow.parquet as pq
        schema = pq.read_schema(cache_files[-1])
        available = [c for c in columns_needed if c in schema.names]
        game_logs_df = pd.concat(
            [pd.read_parquet(f, columns=available) for f in cache_files[-2:]],
            ignore_index=True,
        )
        # Map team_id (string) to team_abbr (3-letter code) — needed for roster building
        if "team_id" in game_logs_df.columns and "team_abbr" not in game_logs_df.columns:
            from src.features.mlb import TEAM_IDS
            game_logs_df["team_abbr"] = (
                game_logs_df["team_id"].astype(int).map(TEAM_IDS).fillna("UNK")
            )
        print(f"  Loaded {len(game_logs_df)} game log rows", flush=True)

    # Build team rosters from game logs for lineup creation
    print("\n4. Building team rosters from game logs...")
    team_rosters: dict[str, list[dict]] = {}
    if game_logs_df is not None and "team_abbr" in game_logs_df.columns:
        for abbr in game_logs_df["team_abbr"].dropna().unique():
            roster = build_team_roster_from_game_logs(game_logs_df, abbr)
            if roster:
                team_rosters[abbr.upper()] = roster
        print(f"  Built rosters for {len(team_rosters)} teams")

    # ── Load handedness data ──
    print("\n5. Loading handedness data...")
    pitcher_hand_map, batter_hand_map = _load_handedness()

    # ── 6. Run simulations ──
    print(f"\n6. Running simulations ({args.sims} per game)...")

    for game in games:
        away = game["away_team"]
        home = game["home_team"]
        ap_name = game["away_pitcher_name"]
        hp_name = game["home_pitcher_name"]

        print(f"\n  {away} @ {home}")

        # Build pitcher features (with handedness)
        ap_feats = build_pitcher_features(
            ap_name, game["away_pitcher_id"], statcast_pitcher_df, pitching_df, game_logs_df,
            pitcher_hand_map=pitcher_hand_map,
        )
        hp_feats = build_pitcher_features(
            hp_name, game["home_pitcher_id"], statcast_pitcher_df, pitching_df, game_logs_df,
            pitcher_hand_map=pitcher_hand_map,
        )

        # Build lineups (with handedness)
        away_roster = team_rosters.get(away.upper())
        home_roster = team_rosters.get(home.upper())
        away_batters = build_lineup(away, statcast_df, roster_batters=away_roster,
                                     batter_hand_map=batter_hand_map)
        home_batters = build_lineup(home, statcast_df, roster_batters=home_roster,
                                     batter_hand_map=batter_hand_map)

        # Park factors
        pf_k = PARK_FACTOR_K.get(home, 1.0)
        pf_hr = PARK_FACTOR_HR.get(home, 1.0)

        # Build team-specific bullpen profiles
        away_bullpen = load_team_bullpen_profiles(game_logs_df, pitching_df, away)
        home_bullpen = load_team_bullpen_profiles(game_logs_df, pitching_df, home)

        # Compute per-starter BF limits for display
        away_max_bf = compute_max_bf(ap_feats)
        home_max_bf = compute_max_bf(hp_feats)

        print(f"    SP: {ap_name[:20]:20s} (max {away_max_bf} BF) vs {hp_name[:20]:20s} (max {home_max_bf} BF)")
        print(f"    Lineups: {len(away_batters)} away × {len(home_batters)} home batters")
        print(f"    Park: K={pf_k:.2f} HR={pf_hr:.2f}")
        print(f"    Simulating...", end=" ", flush=True)

        result = sim.simulate_game(
            ap_feats, hp_feats,
            away_batters, home_batters,
            n_sims=args.sims,
            away_bullpen=away_bullpen,
            home_bullpen=home_bullpen,
            home_park_k=pf_k,
            home_park_hr=pf_hr,
        )

        display_player_probs(game, result, args.sims)

    print(f"\n{'='*60}")
    print(f"  Done at {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
