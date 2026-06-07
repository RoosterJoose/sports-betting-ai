from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.calibrator import PlattCalibrator


class Predictor:
    def __init__(self, model_dir: Path, sport: str, stat_type: str):
        model = xgb.XGBClassifier()
        path = model_dir / sport / f"{stat_type}.json"
        model.load_model(path)
        self.model = model

        cal_path = model_dir / sport / f"{stat_type}_calibration.json"
        self.calibrator = PlattCalibrator.load(cal_path) if cal_path.exists() else None

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        prob = self.model.predict_proba(features)[:, 1]
        if self.calibrator:
            prob = self.calibrator.calibrate(prob)
        return prob

    def predict_proba(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        prob = self.model.predict_proba(features)
        if self.calibrator:
            prob[:, 1] = self.calibrator.calibrate(prob[:, 1])
            prob[:, 0] = 1.0 - prob[:, 1]
        return prob[:, 0], prob[:, 1]
