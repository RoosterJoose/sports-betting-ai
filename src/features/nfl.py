import numpy as np
import pandas as pd
from src.features.base import FeatureEngineer


NFL_STATS = ["passing_yards", "passing_tds", "passing_air_yards", "interceptions",
             "rushing_yards", "rushing_tds", "carries",
             "receiving_yards", "receiving_tds", "receptions", "targets",
             "touchdowns", "fantasy_points", "pass_attempts", "rush_attempts",
             "pass_yds+td", "rush+rec_yds"]


class NFLFeatureEngineer(FeatureEngineer):
    def add_game_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge weather (roof, wind, temp) and vegas (spread_line, total_line) 
        from nfl_data_py schedule into the weekly player DataFrame.

        Schedule data is fetched via nfl_data_py.import_schedules() and merged
        on (season, week, recent_team).
        """
        seasons = sorted(df["season"].dropna().unique().astype(int).tolist())
        if not seasons:
            return df
        
        import nfl_data_py as nfl
        try:
            sched = nfl.import_schedules(seasons)
        except Exception:
            return df
        
        if sched is None or sched.empty:
            return df
        
        # Ensure consistent dtypes
        sched["season"] = sched["season"].astype(int)
        sched["week"] = sched["week"].astype(int)
        
        # Keep relevant schedule columns
        sched_cols = ["season", "week", "home_team", "away_team",
                      "roof", "temp", "wind", "spread_line", "total_line"]
        sched_cols = [c for c in sched_cols if c in sched.columns]
        sched = sched[sched_cols].copy()
        
        # Create team-level lookup: for each (season, week, team), store game info
        home = sched.copy()
        away = sched.copy()
        
        home["team"] = home["home_team"]
        home["is_home"] = 1
        away["team"] = away["away_team"]
        away["is_home"] = 0
        
        team_sched = pd.concat([home, away], ignore_index=True)
        if "roof" in team_sched.columns:
            team_sched["is_dome"] = team_sched["roof"].isin(["dome", "closed"]).astype(int)
        else:
            team_sched["is_dome"] = 0
        
        # Drop raw roof (not a numeric feature)
        drop_cols = ["home_team", "away_team", "roof"]
        team_sched = team_sched.drop(columns=[c for c in drop_cols if c in team_sched.columns])
        
        # Merge into main DataFrame on recent_team
        df = df.merge(team_sched, left_on=["season", "week", "recent_team"],
                      right_on=["season", "week", "team"], how="left")
        
        # Drop redundant team column from right side
        if "team" in df.columns:
            df = df.drop(columns=["team"])
        
        # Fill missing values — use room temp (70) as default for missing, 0 for wind
        if "temp" in df.columns:
            df["temp"] = df["temp"].fillna(70.0)
        if "wind" in df.columns:
            df["wind"] = df["wind"].fillna(0.0)
        if "spread_line" in df.columns:
            df["spread_line"] = df["spread_line"].fillna(0.0)
        if "total_line" in df.columns:
            df["total_line"] = df["total_line"].fillna(0.0)
        for col in ["is_dome", "is_home"]:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)
        
        return df

    def build_features(self, games: pd.DataFrame, opponent_games: pd.DataFrame = None) -> pd.DataFrame:
        """Build features from NFL weekly data, including opponent quality and DvP adjustments.
        
        The input DataFrame must have columns from nfl_data_py import_weekly_data output,
        including: player_id, player_display_name, game_date, season, week, recent_team, 
        opponent_team, position, position_group, and stat columns.
        """
        df = games.copy()
        if df.empty:
            return df
        
        if "game_date" not in df.columns or df["game_date"].dtype == "object":
            df["game_date"] = pd.to_datetime(df["game_date"])
        
        # Standardize team column names
        if "recent_team" in df.columns and "team_abbr" not in df.columns:
            df["team_abbr"] = df["recent_team"]
        
        # Standardize opponent column
        if "opponent_team" in df.columns:
            df["opponent"] = df["opponent_team"]
        
        # ── Game context features (weather + vegas) ──────────────────────
        df = self.add_game_context(df)

        # Compute opponent quality features (defensive stats)
        if "opponent" in df.columns and {"passing_yards", "rushing_yards", "receiving_yards"}.issubset(df.columns):
            # Build opponent defensive stats: for each game, what did each defense allow?
            # Group by opponent (defense) and game to get per-game defensive totals
            def_stats = df.groupby(["game_date", "opponent"]).agg({
                "passing_yards": "sum",
                "rushing_yards": "sum",
                "receiving_yards": "sum",
                "passing_tds": "sum" if "passing_tds" in df.columns else "sum",
                "rushing_tds": "sum" if "rushing_tds" in df.columns else "sum",
                "receiving_tds": "sum" if "receiving_tds" in df.columns else "sum",
                "interceptions": "sum",
                "fantasy_points": "sum" if "fantasy_points" in df.columns else "sum",
            }).reset_index().rename(columns={
                "passing_yards": "def_pass_yds_allowed",
                "rushing_yards": "def_rush_yds_allowed",
                "receiving_yards": "def_rec_yds_allowed",
                "passing_tds": "def_pass_td_allowed",
                "rushing_tds": "def_rush_td_allowed",
                "receiving_tds": "def_rec_td_allowed",
                "interceptions": "def_int_made",
                "fantasy_points": "def_fp_allowed",
            })
            
            # Compute rolling averages for defensive stats (3-game, 5-game)
            def_stats = def_stats.sort_values(["opponent", "game_date"])
            for w in [3, 5]:
                for col in ["def_pass_yds_allowed", "def_rush_yds_allowed", "def_rec_yds_allowed",
                           "def_fp_allowed", "def_pass_td_allowed", "def_int_made"]:
                    if col in def_stats.columns:
                        def_stats[f"{col}_avg_{w}"] = (
                            def_stats.groupby("opponent")[col]
                            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
                        )
            
            # Defense vs Position (DvP): group by opponent + position
            if "position" in df.columns:
                dvp = df.groupby(["game_date", "opponent", "position"]).agg({
                    "fantasy_points": "sum" if "fantasy_points" in df.columns else "sum",
                    "receiving_yards": "sum",
                    "rushing_yards": "sum",
                    "touchdowns": "sum" if "touchdowns" in df.columns else "sum",
                }).reset_index().rename(columns={
                    "fantasy_points": "dvp_fp",
                    "receiving_yards": "dvp_rec_yds",
                    "rushing_yards": "dvp_rush_yds",
                    "touchdowns": "dvp_td",
                })
                
                # Rolling DvP averages
                dvp = dvp.sort_values(["opponent", "position", "game_date"])
                for w in [3, 5]:
                    for col in ["dvp_fp", "dvp_rec_yds", "dvp_rush_yds", "dvp_td"]:
                        if col in dvp.columns:
                            dvp[f"{col}_avg_{w}"] = (
                                dvp.groupby(["opponent", "position"])[col]
                                .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
                            )
                
                # Merge DvP back
                df = df.merge(
                    dvp[["game_date", "opponent", "position"] + [c for c in dvp.columns if c.endswith(("_avg_3", "_avg_5"))]],
                    on=["game_date", "opponent", "position"],
                    how="left"
                )
            
            # Merge defensive stats back
            merge_cols = ["game_date", "opponent"]
            def_merge = [c for c in def_stats.columns if c.endswith(("_avg_3", "_avg_5")) or c in merge_cols]
            df = df.merge(def_stats[def_merge], on=merge_cols, how="left")
        
        # Compute player-level rolling features
        available_stats = [c for c in NFL_STATS if c in df.columns]
        if available_stats:
            # Add EPA features if available
            epa_cols = [c for c in ["passing_epa", "rushing_epa", "receiving_epa"] if c in df.columns]
            all_feature_cols = available_stats + epa_cols
            
            df = self.rolling_averages(df, all_feature_cols)
            df = self.rolling_medians(df, all_feature_cols)
            df = self.recency_weighted_avg(df, all_feature_cols)
            df = self.streak_features(df, all_feature_cols)
            df = self.consistency_features(df, all_feature_cols)
            df = self.schedule_density(df)
        
        # Opponent adjustment for key stats
        for stat in ["passing_yards", "rushing_yards", "receiving_yards", "fantasy_points"]:
            if stat in df.columns:
                try:
                    df = self.opponent_adjustment(df, stat)
                except Exception:
                    pass
        
        # Keep relevant columns
        keep_cols = ["player_id", "player_display_name", "player_name", "game_date", "season", "week",
                     "recent_team", "team_abbr", "opponent", "position", "position_group"]
        for c in df.columns:
            if any(c.endswith(f"_avg_{w}") for w in self.windows):
                keep_cols.append(c)
            elif any(c.endswith(f"_med_{w}") for w in self.windows):
                keep_cols.append(c)
            elif c.endswith("_ewm"):
                keep_cols.append(c)
            elif c.endswith(("_streak", "_consistency", "_adj")):
                keep_cols.append(c)
            elif c in ("days_rest", "b2b", "four_in_six", "def_pass_yds_allowed_avg_3",
                      "def_pass_yds_allowed_avg_5", "def_rush_yds_allowed_avg_3",
                      "def_rush_yds_allowed_avg_5", "def_rec_yds_allowed_avg_3",
                      "def_rec_yds_allowed_avg_5", "def_fp_allowed_avg_3",
                      "def_fp_allowed_avg_5", "def_pass_td_allowed_avg_3",
                      "def_int_made_avg_3", "dvp_fp_avg_3", "dvp_fp_avg_5",
                      "dvp_rec_yds_avg_3", "dvp_rush_yds_avg_3",
                      "is_dome", "is_home", "temp", "wind",
                      "spread_line", "total_line"):
                keep_cols.append(c)
        
        df = df[[c for c in keep_cols if c in df.columns]].copy()
        return df
