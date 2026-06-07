import pandas as pd

from src.features.base import FeatureEngineer
from src.data.golf import GOLF_STAT_MAP

GOLF_STATS = [
    "driving_dist_avg", "driving_acc_avg",
    "gir_avg", "scrambling_avg", "putting_avg_avg",
    "sg_putt_avg", "sg_ott_avg", "sg_app_avg",
    "sg_total_points",
    "birdie_avg_avg",
    "scoring_avg_avg",
]


class GolfFeatureEngineer(FeatureEngineer):
    def build_features(self, games: pd.DataFrame, opponent_games: pd.DataFrame = None) -> pd.DataFrame:
        df = games.copy().sort_values(["player_id", "game_date"])

        for col in ["sg_total_avg", "sg_putt_avg", "sg_app_avg", "sg_ott_avg"]:
            if col in df.columns:
                df = self.rolling_averages(df, [col], group_col="player_id")

        if "sg_total_avg" in df.columns:
            df["sg_total_lag"] = df.groupby("player_id")["sg_total_avg"].shift(1)
            df["sg_trend"] = df.groupby("player_id")["sg_total_avg"].transform(
                lambda x: x.shift(1).rolling(3, min_periods=1).mean()
            )

        keep_cols = ["player_id", "game_date", "season", "tournament_name"]
        for col in df.columns:
            if any(col.endswith(f"_avg_{w}") for w in self.windows):
                keep_cols.append(col)
            elif col.endswith("_lag"):
                keep_cols.append(col)
            elif col.endswith("_trend"):
                keep_cols.append(col)
        # Keep raw stat columns for direct use (no rolling features available)
        # Exclude columns that are source data for target aliases (data leakage)
        if not self.windows:
            stat_source_cols = set(GOLF_STAT_MAP.values())
            for col in GOLF_STATS:
                if col in df.columns and col not in stat_source_cols:
                    keep_cols.append(col)
        df = df[[c for c in keep_cols if c in df.columns]].copy()
        return df
