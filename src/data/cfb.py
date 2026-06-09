import os
from pathlib import Path
from datetime import datetime

import pandas as pd

from src.data.base import DataSource

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cfb_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

STAT_CATEGORY_MAP = {
    "totalYards": "total_yards",
    "netPassingYards": "passing_yards",
    "rushingYards": "rushing_yards",
    "firstDowns": "first_downs",
    "turnovers": "turnovers",
    "penaltyYards": "penalty_yards",
    "sacks": "sacks",
    "tacklesForLoss": "tackles_for_loss",
    "possessionTime": "possession_seconds",
    "thirdDownEff": "third_down_eff",
    "fourthDownEff": "fourth_down_eff",
    "passCompletions": "pass_completions",
    "passAttempts": "pass_attempts",
    "fumblesLost": "fumbles_lost",
    "interceptions": "interceptions_thrown",
    "penalties": "penalties",
}

# Convert "3-10" or "3-10-0" format to a float rate (0.0-1.0)
def _parse_eff_rate(raw: str) -> float:
    try:
        parts = raw.replace("-", "/").split("/")
        if len(parts) >= 2:
            made, att = int(parts[0]), int(parts[1])
            return made / att if att > 0 else 0.0
    except (ValueError, IndexError):
        pass
    return 0.0

# Convert "3:15" or "3-15" format to total seconds
def _parse_possession(raw: str) -> int:
    try:
        sep = ":" if ":" in raw else "-"
        parts = raw.split(sep)
        if len(parts) >= 2:
            return int(parts[0]) * 60 + int(parts[1])
    except (ValueError, IndexError):
        pass
    return 0


class CFBDataSource(DataSource):
    """Data source for College Football using the CFBD (collegefootballdata.com) API.

    Returns per-team per-game stat lines suitable for feature engineering
    and model training. Team-level data (unlike NBA/NFL player props).
    """

    def __init__(self):
        self._cache = {}
        self._games_api = None
        self._stats_api = None
        self._betting_api = None
        self._setup_client()

    def _setup_client(self):
        api_key = os.environ.get("CFBD_API_KEY", "")
        if not api_key:
            print("  CFB: no CFBD_API_KEY set — data fetching will fail")
            return

        import cfbd
        configuration = cfbd.Configuration()
        api_client = cfbd.ApiClient(configuration)
        # cfbd v5.x doesn't auto-configure auth via api_key dict;
        # set the Authorization header directly on the ApiClient.
        api_client.set_default_header("Authorization", f"Bearer {api_key}")
        self._games_api = cfbd.GamesApi(api_client)
        self._stats_api = cfbd.StatsApi(api_client)
        self._betting_api = cfbd.BettingApi(api_client)

    def fetch_player_game_logs(self, seasons: list[str]) -> pd.DataFrame:
        """Fetch per-team per-game stats for the given seasons.

        Returns a DataFrame with one row per team per game, including:
          team, opponent, game_date, season, week,
          home, win, points_for, points_against,
          total_yards, passing_yards, rushing_yards, first_downs,
          turnovers, penalty_yards, sacks, tackles_for_loss,
          third_down_eff, fourth_down_eff, possession_seconds,
          spread_line, total_line
        """
        cache_path = CACHE_DIR / "game_logs.parquet"
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            self._cache["game_logs"] = df
            print(f"  CFB: {len(df)} rows from cache")
            return df

        if self._games_api is None:
            print("  CFB: API client not configured (missing CFBD_API_KEY)")
            return pd.DataFrame()

        frames = []
        for s in seasons:
            year = int(s) if s.isdigit() else datetime.now().year
            try:
                season_df = self._fetch_season(year)
                if not season_df.empty:
                    frames.append(season_df)
                    print(f"    {year}: {len(season_df)} team-game rows")
            except Exception as e:
                print(f"    {year}: error — {e}")
                continue

        if not frames:
            print("  CFB: no data fetched")
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        df.to_parquet(cache_path)
        self._cache["game_logs"] = df
        print(f"  CFB: {len(df)} team-game rows, {df['team'].nunique()} teams, "
              f"{df['game_date'].min()} to {df['game_date'].max()}")
        return df

    def _fetch_season(self, year: int) -> pd.DataFrame:
        """Fetch all games + team stats + lines for a single season."""
        # Step 1: Fetch games (schedule + scores)
        games_dict = self._fetch_games(year)
        if not games_dict:
            return pd.DataFrame()

        # Step 2: Fetch team-level stats for each game
        stats_dict = self._fetch_team_stats(year)

        # Step 3: Fetch betting lines
        lines_dict = self._fetch_lines(year)

        # Build per-team per-game rows
        rows = []
        for game_id, g in games_dict.items():
            home = g["home_team"]
            away = g["away_team"]
            home_pts = g.get("home_points", 0) or 0
            away_pts = g.get("away_points", 0) or 0
            is_neutral = g.get("neutral_site", False)
            game_date = self._parse_date(g.get("start_date", ""))
            week = g.get("week", 0)
            season_type = g.get("season_type", "regular")

            # Spread and total lines
            # Convention: spread_line is from the home team's perspective (negative = home favored).
            # Both home and away rows store the same raw value; the `home` flag (0/1) tells
            # the model which side the spread applies to.
            lines_info = lines_dict.get(game_id, {})
            spread_line = lines_info.get("spread", float("nan"))
            total_line = lines_info.get("over_under", float("nan"))

            # Team stats
            home_stats = stats_dict.get(game_id, {}).get(home, {})
            away_stats = stats_dict.get(game_id, {}).get(away, {})

            # Home team row
            rows.append(self._make_team_row(
                team=home, opponent=away,
                points_for=home_pts, points_against=away_pts,
                stats=home_stats, game_date=game_date,
                season=year, week=week, season_type=season_type,
                home=1 if not is_neutral else 0,
                spread_line=spread_line, total_line=total_line,
            ))

            # Away team row
            rows.append(self._make_team_row(
                team=away, opponent=home,
                points_for=away_pts, points_against=home_pts,
                stats=away_stats, game_date=game_date,
                season=year, week=week, season_type=season_type,
                home=0 if not is_neutral else 0,
                spread_line=spread_line, total_line=total_line,
            ))

        return pd.DataFrame(rows)

    def _fetch_games(self, year: int) -> dict:
        """Fetch games for a season. Returns dict of game_id -> game info dict (snake_case keys)."""
        try:
            games = self._games_api.get_games(year=year)
        except Exception as e:
            print(f"    CFB get_games({year}) failed: {e}")
            return {}

        result = {}
        for g in games:
            # Use snake_case attributes from the raw game objects (NOT to_dict which returns camelCase)
            gid = g.id
            if not gid:
                continue
            result[gid] = {
                "id": gid,
                "home_team": getattr(g, "home_team", "") or "",
                "away_team": getattr(g, "away_team", "") or "",
                "home_points": getattr(g, "home_points", None),
                "away_points": getattr(g, "away_points", None),
                "neutral_site": getattr(g, "neutral_site", False),
                "start_date": str(getattr(g, "start_date", "") or ""),
                "week": getattr(g, "week", 0) or 0,
                "season_type": str(getattr(g, "season_type", "regular") or "regular"),
            }
        return result

    def _fetch_team_stats(self, year: int) -> dict:
        """Fetch team stats for a season.
        Returns dict of game_id -> team_name -> {stat_category: value}
        Uses get_game_team_stats with week-by-week iteration.
        """
        # Get calendar to find available weeks
        try:
            calendar = self._games_api.get_calendar(year=year)
            weeks = []
            for c in calendar:
                d = c.to_dict() if hasattr(c, "to_dict") else {"week": getattr(c, "week", 0)}
                w = d.get("week", 0)
                if w:
                    weeks.append(w)
            weeks = sorted(set(weeks))
        except Exception:
            # Fallback: weeks 1-16
            weeks = list(range(1, 17))

        result = {}
        for week in weeks:
            try:
                stats_list = self._games_api.get_game_team_stats(year=year, week=week)
            except Exception as e:
                continue

            for entry in stats_list:
                d = entry.to_dict() if hasattr(entry, "to_dict") else {}
                gid = d.get("id", "")
                if not gid:
                    continue
                teams_data = d.get("teams", [])
                game_stats = {}
                for t in teams_data:
                    team_name = t.get("team", "")
                    raw_stats = t.get("stats", [])
                    parsed = {}
                    for s in raw_stats:
                        category = s.get("category", "")
                        value = s.get("stat", "0")
                        mapped_col = STAT_CATEGORY_MAP.get(category, "")
                        if mapped_col:
                            parsed[mapped_col] = value
                    if team_name:
                        game_stats[team_name] = parsed
                if game_stats:
                    result[gid] = game_stats

        return result

    def _fetch_lines(self, year: int) -> dict:
        """Fetch betting lines for a season.
        Returns dict of game_id -> {spread, over_under, home_moneyline, away_moneyline}
        Uses the first available provider (usually consensus or Vegas).
        """
        try:
            lines = self._betting_api.get_lines(year=year)
        except Exception as e:
            print(f"    CFB get_lines({year}) failed: {e}")
            return {}

        result = {}
        for entry in lines:
            d = entry.to_dict() if hasattr(entry, "to_dict") else {}
            gid = d.get("id", "")
            if not gid:
                continue

            lines_data = d.get("lines", [])
            if not lines_data:
                continue

            # Use the first line provider (usually consensus)
            first_line = lines_data[0] if isinstance(lines_data, list) else lines_data
            result[gid] = {
                "spread": first_line.get("spread"),
                "over_under": first_line.get("overUnder"),
                "home_moneyline": first_line.get("homeMoneyline"),
                "away_moneyline": first_line.get("awayMoneyline"),
            }
        return result

    def _make_team_row(self, team, opponent, points_for, points_against,
                       stats, game_date, season, week, season_type,
                       home, spread_line, total_line) -> dict:
        """Build a single row for one team in one game."""
        row = {
            "team": team,
            "opponent": opponent,
            "game_date": game_date,
            "season": season,
            "week": week,
            "season_type": season_type,
            "home": home,
            "win": 1 if points_for > points_against else 0,
            "points_for": points_for,
            "points_against": points_against,
            "spread_margin": points_for - points_against,
            "total_points": points_for + points_against,
            "spread_line": spread_line,
            "total_line": total_line,
        }

        # Map stat categories to columns
        for col in STAT_CATEGORY_MAP.values():
            val = stats.get(col, 0)
            if col in ("third_down_eff", "fourth_down_eff"):
                val = _parse_eff_rate(str(val))
            elif col == "possession_seconds":
                val = _parse_possession(str(val))
            else:
                try:
                    val = float(str(val).replace(",", ""))
                except (ValueError, TypeError):
                    val = 0.0
            row[col] = val

        return row

    @staticmethod
    def _parse_date(raw: str):
        """Parse various date formats from the CFBD API."""
        if not raw:
            return pd.NaT
        for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ",
                    "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%d"):
            try:
                return datetime.strptime(raw, fmt)
            except ValueError:
                continue
        return pd.NaT

    # Stub implementations for abstract base class
    def fetch_player_stats(self, player_id: str, start_date, end_date) -> pd.DataFrame:
        return pd.DataFrame()

    def fetch_team_stats(self, team_id: str, start_date, end_date) -> pd.DataFrame:
        return pd.DataFrame()

    def fetch_schedule(self, season: str) -> pd.DataFrame:
        return pd.DataFrame()
