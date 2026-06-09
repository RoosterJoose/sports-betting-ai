from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config.settings import SportConfig, PROJECT_ROOT

SPORT_REGISTRY = {
    "nba":   ("src.data.nba.NBADataSource",   "src.features.nba.NBAFeatureEngineer"),
    "nfl":   ("src.data.nfl.NFLDataSource",   "src.features.nfl.NFLFeatureEngineer"),
    "mlb":   ("src.data.mlb.MLBDataSource",   "src.features.mlb.MLBFeatureEngineer"),
    "nhl":   ("src.data.nhl.NHLDataSource",   "src.features.nhl.NHLFeatureEngineer"),
    "golf":  ("src.data.golf.GolfDataSource", "src.features.golf.GolfFeatureEngineer"),
    "soccer":("src.data.soccer.SoccerDataSource", "src.features.soccer.SoccerFeatureEngineer"),
    "nascar":("src.data.nascar.NASCARDataSource", "src.features.nascar.NASCARFeatureEngineer"),
    "tennis":("src.data.tennis.TennisDataSource", "src.features.tennis.TennisFeatureEngineer"),
    "wnba":  ("src.data.wnba.WNBADataSource",   "src.features.wnba.WNBAFeatureEngineer"),
    "worldcup":("src.data.world_cup.WorldCupDataSource", "src.features.worldcup.WorldCupFeatureEngineer"),
    "ufc":   ("src.data.ufc.UFCDataSource",   "src.features.ufc_fe.UFCFeatureEngineer"),
    "cfb":   ("src.data.cfb.CFBDataSource",   "src.features.cfb.CFBFeatureEngineer"),
}

HISTORICAL_SEASONS = {
    "nba": 4, "nfl": 5, "mlb": 3, "nhl": 4,
    "golf": 2, "soccer": 3, "nascar": 3, "tennis": 3, "wnba": 3, "ufc": 5, "cfb": 5,
}

# Combined stat name -> component column mapping for target resolution
COMBINED_STAT_MAP = {
    "pr": ["pts", "reb"],
    "pa": ["pts", "ast"],
    "ra": ["reb", "ast"],
    "pra": ["pts", "reb", "ast"],
    "sb": ["stl", "blk"],
    "fpts": ["pts", "reb", "ast", "stl", "blk", "tov", "fg3m"],
}

MODEL_DIR = PROJECT_ROOT / "models"


class DataPipeline:
    def __init__(self, sport_config: SportConfig):
        self.config = sport_config
        self.name = sport_config.name
        self.seasons = HISTORICAL_SEASONS.get(self.name, 3)
        self._data_source = None
        self._feature_engineer = None
        self._raw_games: pd.DataFrame | None = None
        self._cached_featured: pd.DataFrame | None = None
        self._feature_cache: dict[str, tuple[pd.DataFrame, pd.Series]] = {}

    def _import_source(self):
        path = SPORT_REGISTRY[self.name][0]
        module_path, class_name = path.rsplit(".", 1)
        mod = __import__(module_path, fromlist=[class_name])
        cls = getattr(mod, class_name)
        self._data_source = cls()

    def _import_features(self):
        path = SPORT_REGISTRY[self.name][1]
        module_path, class_name = path.rsplit(".", 1)
        mod = __import__(module_path, fromlist=[class_name])
        cls = getattr(mod, class_name)
        self._feature_engineer = cls(self.config)

    def fetch_all_games(self) -> pd.DataFrame:
        self._import_source()
        seasons = [str(datetime.now().year - offset) for offset in range(self.seasons)]
        try:
            games = self._data_source.fetch_player_game_logs(seasons)
            if isinstance(games, pd.DataFrame) and not games.empty:
                self._raw_games = games.copy()
                return games
        except Exception as e:
            print(f"  [{self.name}] fetch_player_game_logs failed: {e}")

        print(f"  [{self.name}] Falling back to fetch_schedule...")
        frames = []
        for season in seasons:
            try:
                sched = self._data_source.fetch_schedule(season)
                if isinstance(sched, pd.DataFrame) and not sched.empty:
                    frames.append(sched)
            except Exception as e:
                print(f"  [{self.name}] Could not fetch {season}: {e}")
        if frames:
            games = pd.concat(frames, ignore_index=True)
            self._raw_games = games.copy()
            return games
        return pd.DataFrame()

    def _resolve_target_col(self, stat_type: str, raw_df: pd.DataFrame):
        col = stat_type.lower()
        if col in raw_df.columns:
            return col
        if col in COMBINED_STAT_MAP:
            parts = COMBINED_STAT_MAP[col]
            if all(p in raw_df.columns for p in parts):
                return parts
        if "+" in stat_type:
            parts = [p.strip().lower() for p in stat_type.split("+")]
            if all(p in raw_df.columns for p in parts):
                return parts
        return None

    def build_training_data(self, stat_type: str) -> Optional[tuple[pd.DataFrame, pd.Series]]:
        if stat_type in self._feature_cache:
            return self._feature_cache[stat_type]

        if self._cached_featured is None:
            self._import_source()
            self._import_features()

            raw = self.fetch_all_games()
            if raw.empty:
                return None

            featured = self._feature_engineer.build_features(raw)
            if featured.empty:
                return None
            self._cached_featured = featured

        featured = self._cached_featured.copy()

        # Get raw data for target — use cached raw_games
        raw = self._raw_games
        if raw is None or raw.empty:
            raw = self.fetch_all_games()
        if raw.empty:
            return None

        # Resolve target column
        resolved = self._resolve_target_col(stat_type, raw)
        if resolved is None:
            print(f"  Target '{stat_type}' — column not in raw data")
            return None

        if isinstance(resolved, list):
            # Combined stat: compute from parts
            combined_name = "+".join(resolved)
            raw[combined_name] = sum(raw[p] for p in resolved)
            target_col = combined_name
        else:
            target_col = resolved

        # Merge target into featured
        if "player_id" in featured.columns and "game_date" in featured.columns \
           and "player_id" in raw.columns and "game_date" in raw.columns:
            target_df = raw[["player_id", "game_date", target_col]].copy()
            target_df["game_date"] = pd.to_datetime(target_df["game_date"])
            featured["game_date"] = pd.to_datetime(featured["game_date"])
            featured = featured.merge(target_df, on=["player_id", "game_date"], how="left", suffixes=("", "_y"))
            featured = featured.drop(columns=[f"{target_col}_y"], errors="ignore")
        else:
            return None

        featured = featured.dropna(subset=[target_col]).copy()

        # Build feature columns
        X = featured.drop(columns=[
            target_col, "player_id", "game_date", "game_id", "season", "matchup"
        ], errors="ignore")

        for c in list(X.columns):
            if X[c].dtype not in ("float64", "int64", "float32", "int32", "int8", "int16", "bool"):
                X = X.drop(columns=[c], errors="ignore")

        X = X.fillna(X.median(numeric_only=True))
        y = featured[target_col]

        # Sort chronologically
        sort_idx = featured["game_date"].argsort()
        X = X.iloc[sort_idx]
        y = y.iloc[sort_idx]

        result = (X, y)
        self._feature_cache[stat_type] = result
        return result

    def prepare_training_data(self, stat_type: str):
        result = self.build_training_data(stat_type)
        if result is None:
            return None, None, None, None
        X, y = result
        split_idx = int(len(X) * 0.8)
        return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]

    def save_model(self, model, stat_type: str, metrics: dict) -> Optional[Path]:
        path = MODEL_DIR / self.name / f"{stat_type}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        import json
        import numpy as np
        
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        
        model.save_model(str(path))
        with open(path.with_suffix(".metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2, cls=NumpyEncoder)
        return path
