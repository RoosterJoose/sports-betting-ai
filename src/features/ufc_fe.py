import pandas as pd

from src.features.base import FeatureEngineer


class UFCFeatureEngineer(FeatureEngineer):
    def build_features(self, games: pd.DataFrame, opponent_games: pd.DataFrame = None) -> pd.DataFrame:
        from src.features.ufc import build_ufc_features, FEATURE_COLS
        result = build_ufc_features(games)
        if result.empty:
            return result
        # Keep only feature columns + identity cols
        keep_cols = ["player_id", "game_date", "season"] + [c for c in FEATURE_COLS if c in result.columns]
        return result[[c for c in keep_cols if c in result.columns]].copy()
