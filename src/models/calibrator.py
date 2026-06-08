import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.special import expit, logit
from scipy.stats import norm


class PlattCalibrator:
    """Platt scaling for binary classifier probability calibration.

    Learns logistic calibration: P_calibrated = 1/(1 + exp(A * logit(P_raw) + B))
    """

    def __init__(self):
        self.A = 1.0
        self.B = 0.0

    def fit(self, probs: np.ndarray, labels: np.ndarray):
        from sklearn.linear_model import LogisticRegression
        logits = logit(np.clip(probs, 1e-6, 1 - 1e-6)).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, solver="lbfgs")
        lr.fit(logits, labels)
        self.A = float(lr.coef_[0, 0])
        self.B = float(lr.intercept_[0])

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        logits = logit(np.clip(probs, 1e-6, 1 - 1e-6)) * self.A + self.B
        return expit(logits)

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump({"A": self.A, "B": self.B}, f)

    @classmethod
    def load(cls, path: Path):
        if not path.exists():
            return cls()
        with open(path) as f:
            data = json.load(f)
        c = cls()
        c.A = data.get("A", 1.0)
        c.B = data.get("B", 0.0)
        return c

    def __call__(self, prob: float) -> float:
        if prob <= 0 or prob >= 1:
            return prob
        logit_val = np.log(prob / (1 - prob)) * self.A + self.B
        return 1.0 / (1.0 + np.exp(-logit_val))


class BetaCalibrator:
    """Beta Calibration for regressor-based probability estimates.

    Maps raw (normal-CDF) probabilities to calibrated probabilities using
    the Beta distribution family.  Fits:
        logit(p_cal) = a * log(p_raw) - b * log(1 - p_raw) + c

    Unlike Platt scaling (which assumes a sigmoidal mapping), Beta calibration
    can handle both sigmoidal and inverse-sigmoidal shapes, making it better
    suited for calibrating extreme tail probabilities in sports stats.

    Reference: Kull et al. (2017) "Beyond sigmoids"
    """

    def __init__(self):
        self.a = 1.0
        self.b = 1.0
        self.c = 0.0
        self._fitted = False

    def fit(self, probs: np.ndarray, labels: np.ndarray):
        """Fit Beta calibration via logistic regression on log-prob features."""
        from sklearn.linear_model import LogisticRegression
        p = np.clip(probs, 1e-6, 1 - 1e-6)
        X = np.column_stack([np.log(p), np.log(1 - p)])
        lr = LogisticRegression(C=1e6, solver="lbfgs")
        lr.fit(X, labels)
        self.a = float(lr.coef_[0, 0])
        self.b = -float(lr.coef_[0, 1])  # sign flip for -b*log(1-p) formulation
        self.c = float(lr.intercept_[0])
        self._fitted = True

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Calibrate probability array using the fitted Beta model."""
        if not self._fitted:
            return probs
        p = np.clip(probs, 1e-6, 1 - 1e-6)
        logit_cal = self.a * np.log(p) - self.b * np.log(1 - p) + self.c
        return expit(logit_cal)

    @classmethod
    def fit_from_df(cls, probs: np.ndarray, outcomes: np.ndarray) -> "BetaCalibrator":
        """Factory: fit and return a BetaCalibrator."""
        bc = cls()
        bc.fit(probs, outcomes)
        return bc

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump({"a": self.a, "b": self.b, "c": self.c}, f)

    @classmethod
    def load(cls, path: Path) -> "BetaCalibrator":
        bc = cls()
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            bc.a = data.get("a", 1.0)
            bc.b = data.get("b", 1.0)
            bc.c = data.get("c", 0.0)
            bc._fitted = True
        return bc

    def __call__(self, prob: float | np.ndarray) -> float | np.ndarray:
        """Calibrate probability value(s). Accepts both scalar and array inputs."""
        if not self._fitted:
            return prob
        # Use atleast_1d so indexing always works (scalar -> 1-element array)
        prob_arr = np.atleast_1d(np.asarray(prob, dtype=float))
        # Preserve boundary probabilities (0 or 1) as-is
        boundary = (prob_arr <= 0) | (prob_arr >= 1)
        p = np.clip(prob_arr, 1e-6, 1 - 1e-6)
        logit_val = self.a * np.log(p) - self.b * np.log(1 - p) + self.c
        result = expit(logit_val)
        result[boundary] = prob_arr[boundary]
        if isinstance(prob, (int, float, np.floating)):
            return float(result[0])
        return result


class EmpiricalCalibrator:
    """Empirical calibration for regressor-based probability estimates.

    Replaces theoretical normal CDF with empirical calibration tables
    built from test set outcomes.
    """

    def __init__(self, cal_dir: Path):
        self.cal_dir = Path(cal_dir)
        self.calibration: dict[str, dict] = {}
        self._load_all()

    def _load_all(self):
        for f in self.cal_dir.glob("*_empirical.json"):
            stat = f.name.replace("_empirical.json", "")
            with open(f) as fh:
                self.calibration[stat] = json.load(fh)

    def calibrate(self, stat_type: str, line: int, p_raw: float) -> float:
        """Map raw normal-CDF probability to empirically calibrated probability."""
        cal = self.calibration.get(stat_type.lower(), {})
        line_key = str(line)
        bins = cal.get(line_key, {}).get("bins", [])
        if not bins:
            return max(0.001, min(0.999, p_raw))

        for bin_ in bins:
            if bin_["p_pred_min"] <= p_raw < bin_["p_pred_max"]:
                return max(0.001, min(0.999, bin_["p_actual"]))
        if p_raw >= bins[-1]["p_pred_max"]:
            return max(0.001, min(0.999, bins[-1]["p_actual"]))
        return max(0.001, min(0.999, p_raw))

    @staticmethod
    def p_ge_line_raw(mu: float, sigma: float, line_val: float) -> float:
        """Raw probability via normal CDF (before calibration)."""
        p = 1.0 - norm.cdf((line_val - 0.5 - mu) / max(sigma, 0.3))
        return max(0.001, min(0.999, float(p)))
