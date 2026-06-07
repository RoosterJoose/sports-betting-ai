import pandas as pd

from src.features.base import FeatureEngineer

NHL_STATS = [
    "goals", "assists", "points", "shots",
    "icetime", "giveaways", "takeaways",
    "faceoff_win_pct",
]

NHL_ADVANCED = [
    "corsi_for", "corsi_against", "corsi_pct",
    "fenwick_for", "fenwick_against", "fenwick_pct",
    "xgoals_for", "xgoals_against", "xgoals_pct",
    "scoring_chances_for", "scoring_chances_against",
]


class NHLFeatureEngineer(FeatureEngineer):
    def build_features(self, games: pd.DataFrame, opponent_games: pd.DataFrame = None) -> pd.DataFrame:
        df = games.copy()

        if "game_date" not in df.columns:
            df["game_date"] = pd.to_datetime(df["game_date"])

        for col_base, col_against in [
            ("corsi_for", "corsi_against"),
            ("fenwick_for", "fenwick_against"),
            ("xgoals_for", "xgoals_against"),
        ]:
            if col_base in df.columns and col_against in df.columns:
                pct_col = col_base.replace("_for", "_pct")
                df[pct_col] = df[col_base] / (df[col_base] + df[col_against]).replace(0, 1)

        available_stats = [c for c in NHL_STATS if c in df.columns]
        available_advanced = [c for c in NHL_ADVANCED if c in df.columns]

        if "situation" in df.columns:
            for situation in ["5on5", "4on5", "5on4"]:
                mask = df["situation"] == situation
                for stat in available_stats + available_advanced:
                    if stat in df.columns:
                        df[f"{stat}_{situation}"] = df[stat].where(mask, 0)

        if "icetime" in df.columns:
            df["points_per_icetime"] = df["points"] / df["icetime"].replace(0, 1)
            df["shots_per_60"] = df["shots"] / (df["icetime"] / 60).replace(0, 1) * 60


        df = self.rolling_averages(df, available_stats + available_advanced)
        df = self.rolling_medians(df, available_stats + available_advanced)
        df = self.recency_weighted_avg(df, available_stats + available_advanced)
        df = self.streak_features(df, available_stats + available_advanced)
        df = self.consistency_features(df, available_stats + available_advanced)
        df = self.schedule_density(df)

        keep_cols = ["player_id", "game_date", "season", "team"]
        for c in df.columns:
            if any(c.endswith(f"_avg_{w}") for w in self.windows):
                keep_cols.append(c)
            elif any(c.endswith(f"_med_{w}") for w in self.windows):
                keep_cols.append(c)
            elif c.endswith("_ewm"):
                keep_cols.append(c)
            elif c.endswith(("_streak", "_consistency")):
                keep_cols.append(c)
            elif c in ("days_rest", "b2b", "four_in_six"):
                keep_cols.append(c)

        df = df[[c for c in keep_cols if c in df.columns]].copy()
        return df
