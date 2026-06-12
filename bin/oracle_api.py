#!/usr/bin/env python3
"""Oracle API — thin FastAPI wrapper for the researcher agent's tool-use pattern.

Exposes two read-only endpoints that read directly from the existing parquet
caches. Designed to be called by the `researcher-web` agent (or any LLM agent)
as a custom tool instead of relying on the public web, which has no 2026
game data.

Endpoints
---------
GET /health
    Liveness check. Returns {"status": "ok", "timestamp": ...}.

GET /schema
    OpenAPI-style schema for the two endpoints below.

GET /oracle/mlb/{date}
    MLB slate for `date` (YYYY-MM-DD). Reads `data/cache/mlb/game_logs_*.parquet`
    and aggregates player-game logs to game-level. Returns matchup, probable
    pitchers (if identifiable from the data), score, players with key stats.

GET /oracle/wc/{date}
    World Cup matches for `date` (YYYY-MM-DD). Reads
    `data/cache/worldcup/all_matches.parquet` and returns home/away team,
    scores, Elo ratings, tournament code.

Why a tool, not a file
----------------------
The static-file oracle (`bin/research_oracle.py` → `data/oracle/YYYY-MM-DD.md`)
is the short-term "context-first" approach: inline the file in the prompt. The
API is the long-term scalable approach: the agent calls a function and gets a
JSON response, just like any tool. The two are complementary — the static file
is for ad-hoc dispatches; the API is for automated/orchestrated workflows.

Run
---
    # Local dev (auto-reload):
    python -m bin.oracle_api --reload

    # Production:
    uvicorn bin.oracle_api:app --host 0.0.0.0 --port 8000

Dependencies
------------
    pip install fastapi 'uvicorn[standard]'

The project uses pydantic v1 (via cfbd transitive dep). FastAPI 0.110+ requires
pydantic v2. If you need pydantic v1, pin `fastapi<0.110` and `pydantic<2`.

Auth
----
This API has no auth by default — it reads public game data. If exposing
externally, run behind a reverse proxy with API key auth, or set
`ORACLE_API_TOKEN` env var to require `Authorization: Bearer <token>`.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

warnings.filterwarnings("ignore")

# Make `bin` importable as a package; fall back to direct path insertion
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from fastapi import Depends, FastAPI, HTTPException, Query, status
    from fastapi.responses import JSONResponse
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
except ImportError as e:
    print("FastAPI is not installed. Run: pip install fastapi 'uvicorn[standard]'",
          file=sys.stderr)
    raise

import pandas as pd

# ── Config ──────────────────────────────────────────────────────────────────

MLB_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "mlb"
WC_CACHE_PATH = PROJECT_ROOT / "data" / "cache" / "worldcup" / "all_matches.parquet"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
API_TOKEN = os.environ.get("ORACLE_API_TOKEN", "")  # if set, require Bearer auth

logger = logging.getLogger("oracle_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Sports Betting Oracle API",
    description=(
        "Read-only API over the project's parquet caches. "
        "Designed to be called as a custom tool by the researcher agent."
    ),
    version="0.1.0",
)

# Optional Bearer auth — only enforced if ORACLE_API_TOKEN is set
_bearer_scheme = HTTPBearer(auto_error=False) if API_TOKEN else None


def _verify_token(creds: Optional["HTTPAuthorizationCredentials"] = None) -> None:
    """Verify Bearer token if API_TOKEN is set. Raises 401 on mismatch."""
    if not API_TOKEN:
        return  # no auth configured
    if creds is None or creds.scheme.lower() != "bearer" or creds.credentials != API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── Helpers ────────────────────────────────────────────────────────────────


def _parse_date(date_str: str) -> str:
    """Validate YYYY-MM-DD and return the canonical form. Raises 400 on bad input."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date '{date_str}'. Expected YYYY-MM-DD.",
        )


@lru_cache(maxsize=1)
def _load_mlb_cache() -> Optional[pd.DataFrame]:
    """Load the most recent MLB game_logs parquet, cached for the process lifetime."""
    candidates = sorted(MLB_CACHE_DIR.glob("game_logs_*.parquet"))
    if not candidates:
        return None
    # Prefer the 2026 file (most current) if present, else fall back to newest
    preferred = [c for c in candidates if "2026" in c.name]
    chosen = preferred[0] if preferred else candidates[-1]
    logger.info(f"  Loading MLB cache: {chosen.name} ({chosen.stat().st_size // 1024} KB)")
    return pd.read_parquet(chosen)


@lru_cache(maxsize=1)
def _load_wc_cache() -> Optional[pd.DataFrame]:
    """Load the WC all_matches parquet, cached for the process lifetime."""
    if not WC_CACHE_PATH.exists():
        return None
    logger.info(f"  Loading WC cache: {WC_CACHE_PATH.name} ({WC_CACHE_PATH.stat().st_size // 1024} KB)")
    return pd.read_parquet(WC_CACHE_PATH)


# ── MLB team / position lookups (mirror scan_mlb_sim.py) ───────────────────

TEAM_ID_TO_ABBR = {
    108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS",
    112: "CHC", 113: "CIN", 114: "CLE", 115: "COL",
    116: "DET", 117: "HOU", 118: "KC", 119: "LAD",
    120: "WSH", 121: "NYM", 133: "OAK", 134: "PIT",
    135: "SD", 136: "SEA", 137: "SF", 138: "STL",
    139: "TB", 140: "TEX", 141: "TOR", 142: "MIN",
    143: "PHI", 144: "ATL", 145: "CWS", 146: "MIA",
    147: "NYY", 158: "MIL",
}
TEAM_FULL_NAME = {
    "ARI": "Arizona Diamondbacks", "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles", "BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs", "CIN": "Cincinnati Reds",
    "CLE": "Cleveland Guardians", "COL": "Colorado Rockies",
    "CWS": "Chicago White Sox", "DET": "Detroit Tigers",
    "HOU": "Houston Astros", "KC": "Kansas City Royals",
    "LAA": "Los Angeles Angels", "LAD": "Los Angeles Dodgers",
    "MIA": "Miami Marlins", "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins", "NYM": "New York Mets",
    "NYY": "New York Yankees", "OAK": "Oakland Athletics",
    "PHI": "Philadelphia Phillies", "PIT": "Pittsburgh Pirates",
    "SD": "San Diego Padres", "SEA": "Seattle Mariners",
    "SF": "San Francisco Giants", "STL": "St. Louis Cardinals",
    "TB": "Tampa Bay Rays", "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays", "WSH": "Washington Nationals",
}


# ── Routes ──────────────────────────────────────────────────────────────────


@app.get("/health")
def health() -> dict:
    """Liveness check. Always returns 200 unless the process is dead."""
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version": app.version,
        "mlb_cache_loaded": _load_mlb_cache() is not None,
        "wc_cache_loaded": _load_wc_cache() is not None,
    }


@app.get("/schema")
def schema() -> dict:
    """Lightweight schema for the two main endpoints. Use this to teach an
    agent the function signatures before it dispatches the calls.
    """
    return {
        "endpoints": [
            {
                "method": "GET",
                "path": "/oracle/mlb/{date}",
                "params": {"date": "YYYY-MM-DD (required)"},
                "returns": {
                    "date": "string (echo)",
                    "games": [
                        {
                            "game_pk": "int",
                            "date": "string",
                            "away_team": "string (3-letter abbr)",
                            "away_team_full": "string",
                            "home_team": "string (3-letter abbr)",
                            "home_team_full": "string",
                            "home_or_away": "string (per player)",
                            "players": [
                                {
                                    "player_id": "int",
                                    "player_name": "string",
                                    "position": "string (P/OF/IF/...)",
                                    "opponent": "string (3-letter abbr)",
                                    "stats": {
                                        "ip": "float", "so": "int", "bb": "int",
                                        "h": "int", "r": "int", "er": "int",
                                        "hr": "int", "tb": "int", "rbi": "int",
                                        "sb": "int", "ab": "int",
                                    },
                                }
                            ],
                        }
                    ],
                    "n_games": "int",
                },
            },
            {
                "method": "GET",
                "path": "/oracle/wc/{date}",
                "params": {"date": "YYYY-MM-DD (required)"},
                "returns": {
                    "date": "string (echo)",
                    "matches": [
                        {
                            "match_date": "string",
                            "home_team": "string (3-letter WC code)",
                            "away_team": "string (3-letter WC code)",
                            "home_score": "int or null",
                            "away_score": "int or null",
                            "tournament_code": "string (WC / FRIENDLY / ...)",
                            "elo_home_pre": "float or null",
                            "elo_away_pre": "float or null",
                            "result": "string (home / draw / away / upcoming)",
                        }
                    ],
                    "n_matches": "int",
                },
            },
        ],
    }


@app.get("/oracle/mlb/{date}")
def oracle_mlb(
    date: str,
    include_stats: bool = Query(True, description="Include per-player stat lines"),
    creds: Optional[Any] = Depends(_bearer_scheme) if API_TOKEN else None,
) -> dict:
    """MLB slate for `date` from the game_logs parquet cache.

    Returns one entry per game with the player-game-log rows for that game.
    Pitchers (position='P') and batters (any other position) are returned
    together so the caller can identify probable SPs (pitchers with gs=1).
    """
    _verify_token(creds)
    date_canon = _parse_date(date)
    df = _load_mlb_cache()
    if df is None:
        raise HTTPException(
            status_code=503,
            detail="MLB cache not loaded. Run bin/refresh_mlb_cache.sh first.",
        )
    if "game_date" not in df.columns:
        raise HTTPException(
            status_code=500,
            detail="MLB cache is missing 'game_date' column — schema mismatch.",
        )
    # Filter to the date. game_date may be string or datetime.
    game_date_str = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")
    day_df = df[game_date_str == date_canon]
    if day_df.empty:
        return {
            "date": date_canon,
            "n_games": 0,
            "games": [],
            "note": f"No MLB games in cache for {date_canon}.",
        }

    # Group by game_pk → one entry per game
    games: list[dict] = []
    for game_pk, gdf in day_df.groupby("game_pk"):
        # Identify home/away: 'home_or_away' column has 'home' / 'away'
        if "home_or_away" in gdf.columns:
            home_players = gdf[gdf["home_or_away"].str.lower() == "home"]
            away_players = gdf[gdf["home_or_away"].str.lower() == "away"]
        else:
            # Fallback: derive from team_id matching opponents
            home_players = gdf
            away_players = gdf.iloc[0:0]

        # Team abbr: take the first non-null team_abbr or fall back to opponent
        def _team_of(subset: pd.DataFrame) -> str:
            if "team_abbr" in subset.columns and not subset.empty:
                v = subset["team_abbr"].dropna()
                if not v.empty:
                    return str(v.iloc[0])
            if "opponent" in subset.columns and not subset.empty:
                return str(subset["opponent"].dropna().iloc[0]) if not subset["opponent"].dropna().empty else ""
            return ""

        away_abbr = _team_of(away_players) or (gdf["opponent"].dropna().iloc[0] if "opponent" in gdf.columns and not gdf["opponent"].dropna().empty else "")
        home_abbr = _team_of(home_players) or (gdf["opponent"].dropna().iloc[0] if "opponent" in gdf.columns and not gdf["opponent"].dropna().empty else "")

        # Find the home SP (pitcher with gs=1 on the home side)
        def _sp_of(side_df: pd.DataFrame) -> Optional[str]:
            if side_df.empty or "position" not in side_df.columns:
                return None
            sps = side_df[(side_df["position"] == "P") & (side_df.get("gs", 0) == 1)]
            if sps.empty:
                return None
            return str(sps["player_name"].iloc[0])

        away_sp = _sp_of(away_players)
        home_sp = _sp_of(home_players)

        # Build player rows
        player_rows: list[dict] = []
        if include_stats:
            for _, row in gdf.iterrows():
                stats_cols = ["ip", "so", "bb", "h", "r", "er", "hr", "tb", "rbi", "sb", "ab", "hbp", "sf"]
                stats = {}
                for c in stats_cols:
                    if c in row.index and pd.notna(row[c]):
                        v = row[c]
                        stats[c] = float(v) if isinstance(v, (int, float)) else v
                player_rows.append({
                    "player_id": int(row["player_id"]) if pd.notna(row.get("player_id")) else None,
                    "player_name": str(row.get("player_name", "")),
                    "position": str(row.get("position", "")),
                    "home_or_away": str(row.get("home_or_away", "")),
                    "opponent": str(row.get("opponent", "")),
                    "stats": stats,
                })

        # Compute home/away score (sum of team runs if present)
        def _score(side_df: pd.DataFrame) -> Optional[int]:
            if side_df.empty or "r" not in side_df.columns:
                return None
            try:
                return int(side_df["r"].sum())
            except (TypeError, ValueError):
                return None

        games.append({
            "game_pk": int(game_pk),
            "date": date_canon,
            "away_team": away_abbr,
            "away_team_full": TEAM_FULL_NAME.get(away_abbr, ""),
            "home_team": home_abbr,
            "home_team_full": TEAM_FULL_NAME.get(home_abbr, ""),
            "away_sp": away_sp,
            "home_sp": home_sp,
            "away_score": _score(away_players),
            "home_score": _score(home_players),
            "n_players": int(len(gdf)),
            "players": player_rows,
        })

    # Sort by first-pitch (game_pk is roughly chronological within a day)
    games.sort(key=lambda g: g["game_pk"])
    return {
        "date": date_canon,
        "n_games": len(games),
        "games": games,
    }


@app.get("/oracle/wc/{date}")
def oracle_wc(
    date: str,
    creds: Optional[Any] = Depends(_bearer_scheme) if API_TOKEN else None,
) -> dict:
    """World Cup matches for `date` from the all_matches parquet cache.

    Returns one entry per match with scores, Elo ratings, and tournament code.
    """
    _verify_token(creds)
    date_canon = _parse_date(date)
    df = _load_wc_cache()
    if df is None:
        raise HTTPException(
            status_code=503,
            detail="WC cache not loaded. Expected at data/cache/worldcup/all_matches.parquet",
        )
    if "match_date" not in df.columns:
        raise HTTPException(
            status_code=500,
            detail="WC cache is missing 'match_date' column — schema mismatch.",
        )
    match_date_str = pd.to_datetime(df["match_date"]).dt.strftime("%Y-%m-%d")
    day_df = df[match_date_str == date_canon]
    if day_df.empty:
        return {
            "date": date_canon,
            "n_matches": 0,
            "matches": [],
            "note": f"No WC matches in cache for {date_canon}.",
        }

    matches: list[dict] = []
    for _, row in day_df.iterrows():
        hs = row.get("home_score")
        as_ = row.get("away_score")
        if pd.isna(hs):
            hs = None
        else:
            hs = int(hs)
        if pd.isna(as_):
            as_ = None
        else:
            as_ = int(as_)
        result = None
        if hs is not None and as_ is not None:
            if hs > as_:
                result = "home"
            elif hs < as_:
                result = "away"
            else:
                result = "draw"
        else:
            result = "upcoming"

        matches.append({
            "match_date": str(row["match_date"])[:10],
            "home_team": str(row.get("home_team", "")),
            "away_team": str(row.get("away_team", "")),
            "home_score": hs,
            "away_score": as_,
            "tournament_code": str(row.get("tournament_code", "")),
            "elo_home_pre": float(row["elo_home_pre"]) if pd.notna(row.get("elo_home_pre")) else None,
            "elo_away_pre": float(row["elo_away_pre"]) if pd.notna(row.get("elo_away_pre")) else None,
            "result": result,
        })
    matches.sort(key=lambda m: m["match_date"])
    return {
        "date": date_canon,
        "n_matches": len(matches),
        "matches": matches,
    }


# ── Entrypoint ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Oracle API — FastAPI wrapper over parquet caches.")
    parser.add_argument("--host", type=str, default=os.environ.get("ORACLE_API_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("ORACLE_API_PORT", DEFAULT_PORT)))
    parser.add_argument("--reload", action="store_true", help="Auto-reload on file changes (dev only).")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed. Run: pip install 'uvicorn[standard]'", file=sys.stderr)
        sys.exit(1)

    print(f"  Oracle API — listening on http://{args.host}:{args.port}")
    print(f"  Try: curl http://{args.host}:{args.port}/health")
    print(f"  Try: curl http://{args.host}:{args.port}/oracle/mlb/2026-06-12")
    print(f"  Try: curl http://{args.host}:{args.port}/oracle/wc/2026-06-12")
    if API_TOKEN:
        print(f"  Auth: Bearer token required (ORACLE_API_TOKEN env var set)")
    else:
        print(f"  Auth: none (set ORACLE_API_TOKEN to require auth)")
    uvicorn.run(
        "bin.oracle_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
