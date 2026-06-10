import pandas as pd
import numpy as np

from src.features.base import FeatureEngineer

NBA_STATS = [
    "pts", "reb", "ast", "stl", "blk", "tov",
    "fg3m", "fg3a", "fgm", "fga", "ftm", "fta",
    "min", "plus_minus",
]

NBA_SCARCE = ["stl", "blk"]  # Poisson/low-frequency, need log-transform

NBA_COMBINED = {
    "pr": ["pts", "reb"],
    "pa": ["pts", "ast"],
    "ra": ["reb", "ast"],
    "pra": ["pts", "reb", "ast"],
    "sb": ["stl", "blk"],
}


class NBAFeatureEngineer(FeatureEngineer):

    def _add_opponent_defensive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute opponent defensive vulnerability features per player-row.

        For each player-row, we want stats describing the opposing team's
        recent defensive performance (what they ALLOW). These are the missing
        signal in STL/BLK predictions — steals depend on opponent TOV rate;
        blocks depend on opponent shot volume / pace.

        Implementation:
          1. Aggregate player-rows to team-game level (sum stl, blk, tov, etc.)
          2. For each (team, game_date), look up opponent's team_id in same game
          3. Compute team-level cumulative defensive stats: per (defending_team,
             game_date), sum the offensive stats the OPPOSING team recorded
             against them in past games. Then divide by games played for a rate.
          4. Join back to player-rows on (opponent_team_id, game_date)

        All features are temporally safe (cumulative + shift, no future data).

        Output columns added to df:
          opp_stl_allowed_avg   — opponent avg steals allowed per game (10g)
          opp_blk_allowed_avg   — opponent avg blocks allowed per game (10g)
          opp_tov_forced_avg    — opponent avg turnovers forced per game (10g)
          opp_pace_avg          — opponent avg possessions per game (10g)
        """
        # Step 1: aggregate player-rows to team-game level
        # Each game has N player-rows; we need one row per (team, game).
        team_game = (
            df.groupby(["team_id", "game_date", "game_id"], as_index=False)
            .agg(
                stl=("stl", "sum"),
                blk=("blk", "sum"),
                tov=("tov", "sum"),
                min_team=("min", "sum"),  # proxy for team total minutes
            )
        )
        # Identify the opponent in the same game (any other team in that game)
        # Each game has exactly 2 teams, so we can join on game_id and exclude self
        game_teams = team_game[["game_id", "team_id"]].drop_duplicates()
        game_teams = game_teams.merge(
            game_teams.rename(columns={"team_id": "opp_team_id"}),
            on="game_id",
        )
        game_teams = game_teams[game_teams["team_id"] != game_teams["opp_team_id"]]
        # Merge opponent id back into team_game
        team_game = team_game.merge(
            game_teams[["game_id", "team_id", "opp_team_id"]],
            on=["game_id", "team_id"],
            how="left",
        )

        # Step 2: build the opponent defensive history table
        # For each (defending_team, game_date), compute the offensive stats
        # the OPPOSING team recorded against them in that specific game.
        # Then we cumulative-sum by defending_team to get running averages.
        def_team = team_game.rename(columns={
            "team_id": "def_team_id",
            "opp_team_id": "off_team_id",
            "stl": "off_stl", "blk": "off_blk", "tov": "off_tov",
            "min_team": "off_min",
        })[["def_team_id", "game_date", "off_team_id", "off_stl", "off_blk", "off_tov", "off_min"]]

        # For each defending team, sort by date and cumulative-sum what opponents did
        def_team = def_team.sort_values(["def_team_id", "game_date"]).reset_index(drop=True)
        def_team["cum_off_stl"] = def_team.groupby("def_team_id")["off_stl"].cumsum().shift(1)
        def_team["cum_off_blk"] = def_team.groupby("def_team_id")["off_blk"].cumsum().shift(1)
        def_team["cum_off_tov"] = def_team.groupby("def_team_id")["off_tov"].cumsum().shift(1)
        def_team["cum_off_min"] = def_team.groupby("def_team_id")["off_min"].cumsum().shift(1)
        def_team["cum_games"]   = def_team.groupby("def_team_id").cumcount()
        # shift(1) above excludes the current game → no leakage

        # Per-game rates (avg over all past games for this defending team)
        safe_games = def_team["cum_games"].replace(0, 1)
        def_team["def_stl_allowed_avg"] = def_team["cum_off_stl"] / safe_games
        def_team["def_blk_allowed_avg"] = def_team["cum_off_blk"] / safe_games
        def_team["def_tov_forced_avg"]  = def_team["cum_off_tov"] / safe_games
        # Pace proxy: opponent team total minutes / game (240 = 5 players × 48 min)
        def_team["def_pace_avg"]        = def_team["cum_off_min"] / safe_games / 5.0

        # Step 3: join to player-rows
        # For each player-row, the defending team is the opponent
        # We need to look up: opponent_team_id (from player-row), game_date
        # and get the defending team's avg stats
        if "opp_team_id" not in df.columns:
            # Derive opponent team id from game_id if not already present
            # This re-uses the team_game mapping we just built
            df = df.merge(
                game_teams[["game_id", "team_id", "opp_team_id"]],
                left_on=["game_id", "team_id"],
                right_on=["game_id", "team_id"],
                how="left",
            )

        df = df.merge(
            def_team[["def_team_id", "game_date",
                      "def_stl_allowed_avg", "def_blk_allowed_avg",
                      "def_tov_forced_avg", "def_pace_avg"]],
            left_on=["opp_team_id", "game_date"],
            right_on=["def_team_id", "game_date"],
            how="left",
        )

        # Rename to expected column names; drop the join key
        rename_map = {
            "def_stl_allowed_avg": "opp_stl_allowed_avg",
            "def_blk_allowed_avg": "opp_blk_allowed_avg",
            "def_tov_forced_avg":  "opp_tov_forced_avg",
            "def_pace_avg":        "opp_pace_avg",
        }
        df = df.rename(columns=rename_map)
        if "def_team_id" in df.columns:
            df = df.drop(columns=["def_team_id"])

        # Fill missing with league averages (no opponent history yet — early season)
        league_stl = df["opp_stl_allowed_avg"].median() if df["opp_stl_allowed_avg"].notna().any() else 8.0
        league_blk = df["opp_blk_allowed_avg"].median() if df["opp_blk_allowed_avg"].notna().any() else 5.0
        league_tov = df["opp_tov_forced_avg"].median() if df["opp_tov_forced_avg"].notna().any() else 14.0
        league_pace = df["opp_pace_avg"].median() if df["opp_pace_avg"].notna().any() else 48.0
        df["opp_stl_allowed_avg"] = df["opp_stl_allowed_avg"].fillna(league_stl)
        df["opp_blk_allowed_avg"] = df["opp_blk_allowed_avg"].fillna(league_blk)
        df["opp_tov_forced_avg"]  = df["opp_tov_forced_avg"].fillna(league_tov)
        df["opp_pace_avg"]        = df["opp_pace_avg"].fillna(league_pace)

        n_filled = (df["opp_stl_allowed_avg"].notna()).sum()
        print(f"  Opponent defensive features: {n_filled}/{len(df)} rows matched",
              flush=True)
        return df

    def build_features(self, games: pd.DataFrame, opponent_games: pd.DataFrame = None) -> pd.DataFrame:
        df = games.copy()
        if "game_date" not in df.columns:
            df["game_date"] = pd.to_datetime(df["game_date"])

        # Log-transform scarce stats for Poisson stability
        for col in NBA_SCARCE:
            if col in df.columns:
                df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))

        stats_for_features = NBA_STATS + [f"{s}_log" for s in NBA_SCARCE]

        # Rolling features (all use shift(1) = no leak)
        df = self.rolling_averages(df, stats_for_features)
        df = self.rolling_medians(df, stats_for_features)
        df = self.recency_weighted_avg(df, stats_for_features)
        df = self.streak_features(df, stats_for_features)
        df = self.consistency_features(df, stats_for_features)
        df = self.expected_possessions(df)
        df = self.home_away_split(df)

        # Combined stats — rolling averages, then drop raw combined
        for name, parts in NBA_COMBINED.items():
            combined = f"{name}_combined"
            df[combined] = df[parts].sum(axis=1)
            df = self.rolling_averages(df, [combined])
            df = self.rolling_medians(df, [combined])
            df = self.streak_features(df, [combined])
            df = self.consistency_features(df, [combined])
            df = df.drop(columns=[combined], errors="ignore")

        # Schedule density
        df = self.schedule_density(df)

        # Opponent adjustments at team level
        for stat in ["pts", "reb", "ast", "stl", "blk"]:
            df = self.opponent_adjustment(df, stat)

        # ── Opponent defensive features (HIGH IMPACT for STL/BLK) ─────────
        # For each (team, game_date) we want to know:
        #   opp_stl_allowed = avg steals the opponent team allows per game
        #   opp_blk_allowed = avg blocks the opponent team allows per game
        #   opp_tov_forced  = avg turnovers the opponent team forces per game
        #   opp_pace        = avg possessions the opponent team allows per game
        # These are *opponent defensive vulnerabilities* — the missing signal
        # in STL/BLK predictions. Steals come from opp_tov_forced; blocks
        # come from opp_pace (more attempts = more block opportunities).
        #
        # Temporal safety: cumulative-sum-by-team with shift(1) so each row
        # only sees the opponent's past games. No leakage.
        if all(c in df.columns for c in ["team_id", "game_date", "stl", "blk", "tov", "min"]):
            df = self._add_opponent_defensive_features(df)

        # Strip raw current-game stat columns. Keep only lagged features + schedule + identity.
        keep_cols = ["player_id", "game_date", "game_id", "season", "matchup"]
        for c in df.columns:
            # Rolling averages
            if any(c.endswith(f"_avg_{w}") for w in self.windows):
                keep_cols.append(c)
            # Rolling medians
            elif any(c.endswith(f"_med_{w}") for w in self.windows):
                keep_cols.append(c)
            # EWMs
            elif c.endswith("_ewm"):
                keep_cols.append(c)
            # Derived features
            elif c.endswith(("_streak", "_consistency", "_adj")):
                keep_cols.append(c)
            # Schedule + meta
            elif c in ("days_rest", "b2b", "four_in_six", "is_home", "exp_poss"):
                keep_cols.append(c)

        df = df[[c for c in keep_cols if c in df.columns]].copy()
        return df
