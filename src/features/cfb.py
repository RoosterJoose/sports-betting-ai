import numpy as np
import pandas as pd

from src.features.base import FeatureEngineer

# Core team-level stat columns that serve as feature building blocks
CFB_STATS = [
    "points_for", "points_against",
    "total_yards", "passing_yards", "rushing_yards",
    "first_downs", "turnovers", "penalty_yards",
    "sacks", "tackles_for_loss",
    "third_down_eff", "fourth_down_eff", "possession_seconds",
]

# All feature columns that the model will use
FEATURE_COLS = [
    # Rolling offensive averages (team)
    "points_for_avg_4", "points_for_avg_8", "points_for_avg_12",
    "total_yards_avg_4", "total_yards_avg_8", "total_yards_avg_12",
    "passing_yards_avg_4", "passing_yards_avg_8",
    "rushing_yards_avg_4", "rushing_yards_avg_8",
    "first_downs_avg_4", "turnovers_avg_4",
    "third_down_eff_avg_4", "fourth_down_eff_avg_4",
    "possession_seconds_avg_4",
    # Rolling defensive averages (what team allows)
    "points_against_avg_4", "points_against_avg_8",
    # Rolling medians
    "points_for_med_4", "points_against_med_4",
    "total_yards_med_4",
    # Recency-weighted
    "points_for_ewm", "total_yards_ewm", "passing_yards_ewm",
    # Streak (short - long average)
    "points_for_streak", "total_yards_streak",
    # Consistency
    "points_for_consistency", "total_yards_consistency",
    # Opponent quality (defensive stats from opponent's perspective)
    "opp_points_for_avg_4", "opp_total_yards_avg_4",
    "opp_passing_yards_avg_4", "opp_rushing_yards_avg_4",
    "opp_first_downs_avg_4", "opp_turnovers_avg_4",
    # Differential features (team offense vs opponent defense)
    "off_def_total_yards_diff",
    "off_def_passing_yards_diff",
    "off_def_rushing_yards_diff",
    "off_def_turnovers_diff",
    # Win/loss momentum
    "win_streak",
    "win_pct_4", "win_pct_8",
    # Schedule
    "days_rest",
    # Game context
    "home",
    "spread_line", "total_line",
]


class CFBFeatureEngineer(FeatureEngineer):
    """Feature engineer for College Football team-level predictions.

    Operates on per-team per-game DataFrames from CFBDataSource.
    Uses `team` as the group key (equivalent to `player_id` in player-sport features).
    """

    def __init__(self, config):
        super().__init__(config)
        self.group_col = "team"

    def build_features(self, games: pd.DataFrame, opponent_games: pd.DataFrame = None) -> pd.DataFrame:
        """Build features from CFB per-team per-game data.

        Input columns expected: team, opponent, game_date, season, week,
        home, win, points_for, points_against, total_yards, passing_yards,
        rushing_yards, first_downs, turnovers, penalty_yards, sacks,
        tackles_for_loss, third_down_eff, fourth_down_eff, possession_seconds,
        spread_line, total_line, season_type
        """
        df = games.copy()
        if df.empty:
            return df

        # Ensure datetime
        if "game_date" in df.columns:
            df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

        # Ensure player_id exists (pipeline expects it for merging targets)
        if "player_id" not in df.columns:
            df["player_id"] = df["team"].astype(str)

        # ── 1. Team rolling stats (offensive performance) ──────────────
        available_stats = [c for c in CFB_STATS if c in df.columns]
        if available_stats:
            df = self.rolling_averages(df, available_stats, group_col=self.group_col)
            df = self.rolling_medians(df, available_stats, group_col=self.group_col)
            df = self.recency_weighted_avg(df, available_stats, group_col=self.group_col)
            df = self.streak_features(df, available_stats)
            df = self.consistency_features(df, available_stats, group_col=self.group_col)

        # ── 2. Schedule density (days between games) ──────────────────
        df = self.schedule_density(df, group_col=self.group_col)

        # ── 3. Win/loss momentum ──────────────────────────────────────
        df = self._win_streak_features(df)

        # ── 4. Opponent quality: what does each opponent's defense allow? ──
        if "opponent" in df.columns:
            df = self._opponent_defense_features(df)

        # ── 5. Differential features (team offense vs opponent defense) ──
        df = self._differential_features(df)

        # ── 6. Game context from data source ──────────────────────────
        context_cols = ["home", "spread_line", "total_line"]
        for c in context_cols:
            if c not in df.columns:
                df[c] = 0

        # ── 7. Keep only feature columns (plus required metadata) ─────
        keep_cols = ["team", "player_id", "opponent", "game_date", "season", "week",
                     "season_type", "win", "points_for", "points_against",
                     "spread_margin", "total_points"]
        for c in df.columns:
            if c in keep_cols:
                continue
            if c in FEATURE_COLS:
                keep_cols.append(c)
                continue
            # Also keep rolling/med/ewm columns at windows not in FEATURE_COLS
            # (they'll be consumed by the model if available)
            if any(c.endswith(f"_{t}_{w}") for t in ("avg", "med") for w in self.windows):
                keep_cols.append(c)
            elif c.endswith("_ewm") or c.endswith("_streak") or c.endswith("_consistency"):
                keep_cols.append(c)
            # Opponent features
            elif c.startswith("opp_") and any(c.endswith(f"_avg_{w}") for w in self.windows):
                keep_cols.append(c)
            # Win streak / schedule
            elif c in ("win_streak", "win_pct_4", "win_pct_8", "days_rest", "b2b", "four_in_six"):
                keep_cols.append(c)
            # Differential features
            elif c.endswith("_diff"):
                keep_cols.append(c)
            # Game context
            elif c in ("home", "spread_line", "total_line"):
                keep_cols.append(c)

        df = df[[c for c in keep_cols if c in df.columns]].copy()

        # Fill remaining NaN
        df = df.fillna(0)

        # Ensure player_id is string
        df["player_id"] = df["player_id"].astype(str)

        return df

    # ── Helper: win/loss momentum ──────────────────────────────────────

    def _win_streak_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute win streaks and recent win percentages."""
        df = df.sort_values(["team", "game_date"])

        # Win streak: count consecutive wins, reset to 0 after a loss
        # Shift(1) so current game isn't included
        def _streak(series):
            streak = 0
            result = []
            for val in series:
                if val == 1:
                    streak += 1
                else:
                    streak = 0
                result.append(streak)
            return pd.Series(result, index=series.index)

        df["win_streak"] = (
            df.groupby("team")["win"]
            .transform(lambda x: _streak(x.shift(1).fillna(0)))
        )
        df["win_streak"] = df["win_streak"].fillna(0).astype(int)

        # Win rate in last 4 and 8 games
        for w in [4, 8]:
            df[f"win_pct_{w}"] = (
                df.groupby("team")["win"]
                .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            )
            df[f"win_pct_{w}"] = df[f"win_pct_{w}"].fillna(0.5)

        return df

    # ── Helper: opponent defense features ──────────────────────────────

    def _opponent_defense_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute what each opponent's defense allows on average.
        
        For each game, builds a defensive stats table from each team's
        perspective (what do they allow?), then merges it back by opponent
        so team A's row gets team B's defensive averages.
        """
        df = df.sort_values(["team", "game_date"])

        # Build defensive stats: for each team, what do they allow?
        # "points_against" = what the opponent scored on this team
        # "total_yards" from opponent = what this team's defense allowed
        def_stats = df[["team", "game_date",
                         "points_for", "points_against",
                         "total_yards", "passing_yards", "rushing_yards",
                         "first_downs", "turnovers"]].copy()

        # Rename to defensive perspective
        # When team is on defense: points_for = what opponent scored on them = points_allowed
        # When team is on offense: points_for = what they scored
        # But since each game has 2 rows (team as offense), points_against = defense allowed
        def_stats = def_stats.rename(columns={
            "points_for": "pts_scored_off",       # what this team scored (as offense)
            "points_against": "pts_allowed_def",   # what this team allowed (as defense)
            "total_yards": "yds_allowed_def",
            "passing_yards": "pass_yds_allowed_def",
            "rushing_yards": "rush_yds_allowed_def",
            "first_downs": "fd_allowed_def",
            "turnovers": "to_forced_def",
        })

        # Rolling averages of defensive stats (what this team allows)
        def_cols = ["pts_allowed_def", "yds_allowed_def",
                    "pass_yds_allowed_def", "rush_yds_allowed_def",
                    "fd_allowed_def", "to_forced_def"]
        for col in def_cols:
            if col not in def_stats.columns:
                continue
            for w in self.windows:
                def_stats[f"{col}_avg_{w}"] = (
                    def_stats.groupby("team")[col]
                    .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
                )

        # Merge defensive stats back onto main DataFrame by opponent
        # For team A vs opponent B: take B's defensive averages
        merge_cols_avgs = [c for c in def_stats.columns
                           if any(c.endswith(f"_avg_{w}") for w in self.windows)]
        if not merge_cols_avgs:
            return df

        def_merge = def_stats[["team", "game_date"] + merge_cols_avgs].copy()
        # Rename to opp_ prefix for clarity
        opp_rename = {c: f"opp_{c}" for c in merge_cols_avgs}
        def_merge = def_merge.rename(columns=opp_rename)

        df = df.merge(def_merge,
                      left_on=["opponent", "game_date"],
                      right_on=["team", "game_date"],
                      how="left",
                      suffixes=("", "_dup") if "_dup" not in df.columns else ("", f"_dup"))

        # Clean up duplicate columns from merge
        df = df.drop(columns=[c for c in df.columns if c.endswith("_dup")], errors="ignore")

        return df

    # ── Helper: differential features ─────────────────────────────────

    def _differential_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Team offensive stats vs opponent defensive stats."""
        # After _opponent_defense_features, we have columns like:
        #   team_off: total_yards_avg_4
        #   opp_def:  opp_yds_allowed_def_avg_4 (from merge)
        pairs = [
            ("total_yards", "yds_allowed"),
            ("passing_yards", "pass_yds_allowed"),
            ("rushing_yards", "rush_yds_allowed"),
            ("turnovers", "to_forced"),
        ]
        for off_col, def_key in pairs:
            for w in self.windows:
                off_avg = f"{off_col}_avg_{w}"
                def_avg = f"opp_{def_key}_def_avg_{w}"
                if off_avg in df.columns and def_avg in df.columns:
                    df[f"off_def_{off_col}_diff_{w}"] = df[off_avg] - df[def_avg]

        return df
