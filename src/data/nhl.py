from datetime import datetime
from pathlib import Path
import time

import pandas as pd
import httpx

from src.data.base import DataSource

NHLE_BASE = "https://api-web.nhle.com/v1"
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "nhl_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _season_str(season: str) -> str:
    y = int(str(season)[:4])
    return f"{y}{y+1}"


def _all_teams() -> list[str]:
    r = httpx.get(f"{NHLE_BASE}/standings/now", follow_redirects=True, timeout=15)
    r.raise_for_status()
    teams = set()
    for entry in r.json().get("standings", []):
        abbrev = entry.get("teamAbbrev", {}).get("default", "")
        if abbrev:
            teams.add(abbrev)
    return sorted(teams)


def _fetch_games(api_season: str) -> pd.DataFrame:
    teams = _all_teams()
    print(f"  Fetching NHL {api_season} for {len(teams)} teams...")

    games = []
    for team in teams:
        try:
            r = httpx.get(f"{NHLE_BASE}/roster/{team}/{api_season}", timeout=10)
            if r.status_code != 200:
                continue
            roster = r.json()
            for pos_label in ["forwards", "defensemen", "goalies"]:
                pos_code = "F" if pos_label == "forwards" else "D" if pos_label == "defensemen" else "G"
                for player in roster.get(pos_label, []):
                    pid = player["id"]
                    try:
                        gl = httpx.get(
                            f"{NHLE_BASE}/player/{pid}/game-log/{api_season}/2",
                            timeout=5,
                        )
                        if gl.status_code != 200:
                            continue
                        entries = gl.json().get("gameLog", [])
                        if not entries:
                            continue
                        name = f"{player.get('firstName', {}).get('default', '')} {player.get('lastName', {}).get('default', '')}"
                        for g in entries:
                            games.append({
                                "player_id": str(pid),
                                "player_name": name,
                                "team": g.get("teamAbbrev", ""),
                                "opponent": g.get("opponentAbbrev", ""),
                                "game_id": str(g["gameId"]),
                                "game_date": g.get("gameDate", ""),
                                "home_away": g.get("homeRoadFlag", ""),
                                "goals": g.get("goals", 0),
                                "assists": g.get("assists", 0),
                                "points": g.get("points", 0),
                                "shots": g.get("shots", 0),
                                "hits": g.get("hits", 0),
                                "blocks": g.get("blocks", 0),
                                "plus_minus": g.get("plusMinus", 0),
                                "pim": g.get("pim", 0),
                                "giveaways": g.get("giveaways", 0),
                                "takeaways": g.get("takeaways", 0),
                                "faceoff_win_pct": g.get("faceoffWinningPctg", 0),
                                "toi": g.get("toi", "0:00"),
                                "position": pos_code,
                                "season": api_season,
                            })
                    except Exception:
                        continue
                    time.sleep(0.05)
        except Exception:
            continue
        if len(games) % 2000 == 0:
            print(f"    {len(games)} game entries collected...")

    return pd.DataFrame(games)


class NHLDataSource(DataSource):
    def fetch_player_stats(self, player_id: str, start_date, end_date) -> pd.DataFrame:
        df = self.fetch_player_game_logs([])
        return df[df["player_id"].astype(str) == player_id].copy()

    def fetch_team_stats(self, team_id: str, start_date, end_date) -> pd.DataFrame:
        df = self.fetch_player_game_logs([])
        return df[df["team"] == team_id].copy()

    def fetch_schedule(self, season: str) -> pd.DataFrame:
        return self.fetch_player_game_logs([season])

    def fetch_player_game_logs(self, seasons: list[str]) -> pd.DataFrame:
        cache_path = CACHE_DIR / "game_logs_v2.parquet"
        if cache_path.exists():
            print(f"  Loading NHL from cache: {cache_path}")
            return pd.read_parquet(cache_path)

        if not seasons:
            seasons = ["2025"]
        api_seasons = [_season_str(s) for s in seasons]

        frames = [_fetch_games(s) for s in api_seasons]
        frames = [f for f in frames if not f.empty]

        if frames:
            df = pd.concat(frames, ignore_index=True)
            df["game_date"] = pd.to_datetime(df["game_date"])
            for col in ["goals", "assists", "points", "shots", "hits", "blocks",
                        "plus_minus", "pim", "giveaways", "takeaways"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
            if "toi" in df.columns:
                df["icetime"] = df["toi"].apply(
                    lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1])
                    if isinstance(x, str) and ":" in x else 0
                )
                df = df.drop(columns=["toi"])

            print(f"  NHL: {len(df)} player-game rows ({df['player_name'].nunique()} players, {df['team'].nunique()} teams)")
            df.to_parquet(cache_path)
            return df

        print("  No NHL data available")
        return pd.DataFrame()
