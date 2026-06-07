import pandas as pd
import numpy as np

from src.features.base import FeatureEngineer


class NASCARFeatureEngineer(FeatureEngineer):
    def build_features(self, games: pd.DataFrame, opponent_games: pd.DataFrame = None) -> pd.DataFrame:
        df = games.copy()
        if df.empty:
            return df

        df = df.sort_values(["driver_name", "race_number"]).reset_index(drop=True)

        # Rolling form features (with shift(1) to prevent leakage)
        for window in [3, 5, 10, 20]:
            for col, default in [
                ("finish_position", 20),
                ("standings_position", 20),
            ]:
                df[f"avg_{col}_{window}"] = (
                    df.groupby("driver_name")[col]
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                    .fillna(default)
                )

        # Binary rate features
        for window in [5, 10, 20]:
            for col in ["is_winner", "pole_position", "laps_led_most"]:
                df[f"rate_{col}_{window}"] = (
                    df.groupby("driver_name")[col]
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                    .fillna(0.01)
                )

        # Track-type-specific averages
        for tt in df["track_type"].unique():
            tt_mask = df["track_type"] == tt
            filtered = df[tt_mask].copy()
            filtered["tt_avg_finish"] = (
                filtered.groupby("driver_name")["finish_position"]
                .transform(lambda x: x.shift(1).expanding().mean())
            )
            df[f"tt_{tt}_avg"] = df.merge(
                filtered[["driver_name", "race_number", "tt_avg_finish"]],
                on=["driver_name", "race_number"],
                how="left"
            )["tt_avg_finish"].fillna(20.0)

        # Team consistency features
        df["team_avg_finish"] = (
            df.groupby("team")["finish_position"]
            .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
            .fillna(20)
        )
        df["manufacturer_avg_finish"] = (
            df.groupby("manufacturer")["finish_position"]
            .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
            .fillna(20)
        )

        # Recent form (last 3 races, weighted heavier)
        df["form_recent"] = (
            df.groupby("driver_name")["finish_position"]
            .transform(lambda x: x.shift(1).rolling(3, min_periods=1)
                       .apply(lambda y: np.average(y, weights=[1, 2, 3][-len(y):]) if len(y) > 0 else 20))
            .fillna(20)
        )

        # Finish consistency (lower = more consistent)
        df["finish_std_10"] = (
            df.groupby("driver_name")["finish_position"]
            .transform(lambda x: x.shift(1).rolling(10, min_periods=1).std())
            .fillna(8)
        )

        # Season experience (how many races this season)
        df["season_experience"] = df.groupby("driver_name").cumcount()

        # Car number as categorical
        if "car_number" in df.columns:
            try:
                df["car_number_int"] = pd.to_numeric(df["car_number"], errors="coerce").fillna(0)
            except (ValueError, TypeError):
                df["car_number_int"] = 0

        # Starting position rolling averages (shift(1) prevents leakage)
        df["avg_start_pos_5"] = (
            df.groupby("driver_name")["starting_position"]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
            .fillna(20)
        )
        df["avg_start_pos_10"] = (
            df.groupby("driver_name")["starting_position"]
            .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
            .fillna(20)
        )
        df["avg_start_pos_20"] = (
            df.groupby("driver_name")["starting_position"]
            .transform(lambda x: x.shift(1).rolling(20, min_periods=1).mean())
            .fillna(20)
        )

        df["player_id"] = df["driver_name"]
        
        # Keep only feature columns + identity cols (strip raw stat columns to prevent leakage)
        keep_cols = ["player_id", "driver_name", "game_date", "race_number", "season"]
        for col in df.columns:
            if col.startswith("avg_") or col.startswith("rate_") or col.startswith("tt_"):
                keep_cols.append(col)
            elif col.endswith("_avg_finish") or col.endswith("_recent"):
                keep_cols.append(col)
            elif col.endswith("_std_10"):
                keep_cols.append(col)
            elif col in ("form_recent", "season_experience", "car_number_int"):
                keep_cols.append(col)
            elif col.startswith("avg_start_pos_"):
                keep_cols.append(col)
        return df[[c for c in keep_cols if c in df.columns]].copy()
