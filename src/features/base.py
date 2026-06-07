from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
import pandas as pd


class FeatureEngineer(ABC):
    def __init__(self, config):
        self.config = config
        self.windows = config.rolling_windows
        self.decay = getattr(config, 'recency_decay', 0.001)

    @abstractmethod
    def build_features(self, games: pd.DataFrame, opponent_games: pd.DataFrame = None) -> pd.DataFrame:
        ...

    def rolling_averages(self, df: pd.DataFrame, stat_cols: list[str], group_col: str = "player_id") -> pd.DataFrame:
        """Rolling averages via shift(1). Batch all column creations to avoid fragmentation."""
        df = df.sort_values(["game_date", group_col])
        new_cols = {}
        for w in self.windows:
            for col in stat_cols:
                if col not in df.columns:
                    continue
                new_cols[f"{col}_avg_{w}"] = (
                    df.groupby(group_col)[col]
                    .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
                )
        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        return df

    def rolling_medians(self, df: pd.DataFrame, stat_cols: list[str], group_col: str = "player_id") -> pd.DataFrame:
        """Rolling medians via shift(1). Batch all column creations."""
        df = df.sort_values(["game_date", group_col])
        new_cols = {}
        for w in self.windows:
            for col in stat_cols:
                if col not in df.columns:
                    continue
                new_cols[f"{col}_med_{w}"] = (
                    df.groupby(group_col)[col]
                    .transform(lambda x: x.shift(1).rolling(w, min_periods=1).median())
                )
        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        return df

    def recency_weighted_avg(self, df: pd.DataFrame, stat_cols: list[str], group_col: str = "player_id") -> pd.DataFrame:
        """EWMA via shift(1). Batch all column creations."""
        df = df.sort_values("game_date")
        new_cols = {}
        for col in stat_cols:
            if col not in df.columns:
                continue
            new_cols[f"{col}_ewm"] = (
                df.groupby(group_col)[col]
                .transform(lambda x: x.shift(1).ewm(alpha=self.decay).mean())
            )
        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        return df

    def streak_features(self, df: pd.DataFrame, stat_cols: list[str]) -> pd.DataFrame:
        """Streak = short_avg - long_avg."""
        windows = sorted(set(self.windows))
        if len(windows) < 2:
            return df
        c_short, c_long = f"_avg_{windows[0]}", f"_avg_{windows[-1]}"
        new_cols = {}
        for col in stat_cols:
            s = f"{col}{c_short}"
            l = f"{col}{c_long}"
            if s in df.columns and l in df.columns:
                new_cols[f"{col}_streak"] = df[s] - df[l]
        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        return df

    def consistency_features(self, df: pd.DataFrame, stat_cols: list[str], group_col: str = "player_id") -> pd.DataFrame:
        """Consistency = L5_median / expanding_mean._shift(1)."""
        df = df.sort_values(["game_date", group_col])
        windows = sorted(self.windows)
        short_w = windows[0] if windows else 5
        new_cols = {}
        for col in stat_cols:
            med_col = f"{col}_med_{short_w}"
            if med_col in df.columns:
                season_avg = (
                    df.groupby(group_col)[col]
                    .transform(lambda x: x.shift(1).expanding().mean())
                )
                new_cols[f"{col}_consistency"] = df[med_col] / season_avg.replace(0, np.nan)
        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        return df

    def expected_possessions(self, df: pd.DataFrame, group_col: str = "player_id") -> pd.DataFrame:
        """Simplified: L5_min / league_avg_min as usage proxy."""
        min_col = f"min_avg_5" if 5 in self.windows else None
        if min_col and min_col in df.columns:
            avg_min = df[min_col].mean()
            if avg_min > 0:
                df["exp_poss"] = df[min_col] / avg_min
        return df

    def home_away_split(self, df: pd.DataFrame) -> pd.DataFrame:
        if "matchup" in df.columns:
            df["is_home"] = df["matchup"].str.contains("vs", case=False, na=False).astype(int)
        return df

    def opponent_adjustment(self, df: pd.DataFrame, stat_col: str) -> pd.DataFrame:
        if "opponent" in df.columns:
            baseline = df[stat_col].mean()
            opp_allowed = df.groupby("opponent")[stat_col].transform("mean")
            df[f"{stat_col}_adj"] = df[stat_col] - (opp_allowed - baseline)
        return df

    def schedule_density(self, df: pd.DataFrame, group_col: str = "player_id") -> pd.DataFrame:
        df = df.sort_values("game_date")
        df["days_rest"] = df.groupby(group_col)["game_date"].diff().dt.days
        df["b2b"] = (df["days_rest"] == 1).astype(int)
        df["four_in_six"] = (
            df.groupby(group_col)["game_date"]
            .transform(lambda x: x.rolling(6).count() >= 4)
            .astype(int)
        )
        return df
