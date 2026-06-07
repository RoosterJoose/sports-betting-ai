import pandas as pd

from src.features.base import FeatureEngineer

SOCCER_STATS = [
    "goals", "assists", "shots", "shots_on_target",
    "passes", "tackles", "interceptions", "fouls",
    "yellow_cards", "red_cards", "saves",
]

SOCCER_XG = [
    "xG", "xAG", "xA", "npxG", "npxG+xA",
]


class SoccerFeatureEngineer(FeatureEngineer):
    def build_features(self, games: pd.DataFrame, opponent_games: pd.DataFrame = None) -> pd.DataFrame:
        df = games.copy()

        if "date" not in df.columns:
            df["date"] = pd.to_datetime(df.get("match_date", df.index))

        df = self.rolling_averages(df, SOCCER_STATS, group_col="player_id")
        df = self.recency_weighted_avg(df, SOCCER_STATS, group_col="player_id")

        xg_available = [c for c in SOCCER_XG if c in df.columns]
        if xg_available:
            df = self.rolling_averages(df, xg_available, group_col="player_id")
            df["xG_overperformance"] = df["goals"] - df["xG"]
            df["xG_overperformance_avg_5"] = (
                df.groupby("player_id")["xG_overperformance"]
                .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
            )

        if "home" in df.columns:
            df["is_home"] = df["home"].astype(int)

        if "mins_played" in df.columns:
            df["goals_per_90"] = df["goals"] / (df["mins_played"] / 90).replace(0, 1)
            df["assists_per_90"] = df["assists"] / (df["mins_played"] / 90).replace(0, 1)
            df["shots_per_90"] = df["shots"] / (df["mins_played"] / 90).replace(0, 1)

        df["goal_contribution"] = df["goals"] + df["assists"]
        df = self.rolling_averages(df, ["goal_contribution"], group_col="player_id")

        df = self.schedule_density(df)
        df = df
        return df
