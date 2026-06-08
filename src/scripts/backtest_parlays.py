#!/usr/bin/env python3
"""Parlay backtest — validate that correlation-adjusted joint probabilities match actual hit rates.

For each historical player-game in the test set, predicts P(stat >= line) for
common lines, then checks if multi-leg combos' joint probability is well-calibrated.

Key questions:
  1. Do correlated legs (same game, same player) hit at different rates than P(A)*P(B)?
  2. Does our correlation-adjusted joint probability match actual hit rates?
  3. What's the historical win rate for 2/3/4-leg parlays at various edge thresholds?
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config.settings import CONFIG_DIR, PROJECT_ROOT
from src.features.mlb import MLBFeatureEngineer
from src.models.calibrator import BetaCalibrator
from src.models.distributions import p_ge_stat
from src.execution.parlay_correlation import (
    get_correlation, joint_probability, build_correlation_pairs, compute_payout
)
import toml, lightgbm as lgb

MODEL_DIR = PROJECT_ROOT / "models" / "mlb"
CALIB_DIR = MODEL_DIR / "calibration"
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "mlb"

# Stat definitions matching scanner
STAT_DEFS = [
    ("SO",      "so",   "pitcher", None),
    ("HR",      "hr",   "hitter",  None),
    ("TB",      "tb",   "hitter",  None),
    ("H_R_RBI", None,   "hitter",  lambda df: df["h"] + df["r"] + df["rbi"]),
]


def load_features():
    """Load cached MLB data and build features."""
    cache_files = sorted(CACHE_DIR.glob("game_logs_*.parquet"))
    if not cache_files:
        print("No cached data.")
        return None
    cfg = toml.load(CONFIG_DIR / "mlb.toml")
    from src.config.settings import SportConfig
    scfg = SportConfig(
        name="mlb", display_name="MLB",
        rolling_windows=cfg["features"]["rolling_windows"],
        recency_decay=0.001,
    )
    fe = MLBFeatureEngineer(scfg)
    all_games = pd.concat([pd.read_parquet(f) for f in cache_files], ignore_index=True)
    featured = fe.build_features(all_games)
    raw_cols = ["so", "hr", "tb", "h", "r", "rbi", "1b", "2b", "3b",
                "position", "player_name", "team_abbr", "gs", "bf", "game_pk"]
    raw_keep = [c for c in raw_cols if c in all_games.columns]
    if raw_keep:
        merge_cols = ["player_id", "game_date"]
        all_games["game_date"] = pd.to_datetime(all_games["game_date"])
        featured["game_date"] = pd.to_datetime(featured["game_date"])
        featured = featured.merge(all_games[merge_cols + raw_keep], on=merge_cols, how="left")
    return featured


class ParlayBacktest:
    """Backtest parlay probability calibration.

    For each stat, loads the trained model + BetaCal, then for every player-game
    in the test set, computes P(stat >= line) for common lines.  Then simulates
    2/3/4-leg parlays by randomly pairing player-game predictions and comparing
    the joint probability to the actual outcome.
    """

    def __init__(self):
        self.featured = None
        self.models = {}  # stat_name -> (model, residual_std, BetaCal)
        self.test_data = {}  # stat_name -> list of {player_name, game_pk, mu, y_actual, probs}

    def load(self):
        """Load all models and test data."""
        print("Loading features...", flush=True)
        self.featured = load_features()
        if self.featured is None:
            return False
        print(f"  {len(self.featured)} rows", flush=True)

        for stat_name, raw_col, pos_filter, compute_fn in STAT_DEFS:
            print(f"\nLoading {stat_name}...", flush=True)
            mn = stat_name.lower()
            model_path = MODEL_DIR / f"lgb_{mn}.txt"
            meta_path = MODEL_DIR / f"lgb_{mn}.meta.json"
            cal_path = CALIB_DIR / f"{mn}_beta_cal.json"

            if not model_path.exists():
                print(f"  {stat_name}: model not found")
                continue

            model = lgb.Booster(model_file=str(model_path))
            with open(meta_path) as f:
                meta = json.load(f)
            residual_std = meta.get("residual_std", 1.0)
            model_features = meta.get("features", model.feature_name())
            beta_cal = BetaCalibrator.load(cal_path)

            # Filter and prepare test set
            df = self.featured.copy()
            if pos_filter == "pitcher" and "position" in self.featured.columns:
                df = self.featured[self.featured["position"] == "P"].copy()
            elif pos_filter == "hitter" and "position" in self.featured.columns:
                df = self.featured[self.featured["position"] != "P"].copy()

            target_col = raw_col or f"{stat_name.lower()}_computed"
            if compute_fn is not None:
                df[target_col] = compute_fn(df)
            elif raw_col:
                target_col = raw_col
            if target_col not in df.columns:
                continue
            df = df.dropna(subset=[target_col]).copy()
            if len(df) < 100:
                continue

            y = df[target_col].values
            available = [c for c in model_features if c in df.columns]
            X = df[available].copy().fillna(df[available].median()) if len(available) > 0 else pd.DataFrame()

            # Temporal 80/20 split
            dates = pd.to_datetime(df["game_date"])
            sort_idx = dates.argsort()
            split = int(len(sort_idx) * 0.8)
            test_idx = sort_idx[split:]

            X_test = X.iloc[test_idx] if not X.empty else pd.DataFrame()
            y_test = y[test_idx]
            df_test = df.iloc[test_idx]

            if X_test.empty or len(y_test) < 100:
                print(f"  {stat_name}: insufficient test data")
                continue

            # Predict mu
            test_feat = pd.DataFrame(index=range(len(X_test)))
            for c in model_features:
                if c in X_test.columns:
                    test_feat[c] = X_test[c].values
                else:
                    test_feat[c] = 0.0
            mu = model.predict(test_feat.fillna(0))

            # Compute probs for common lines
            common_lines = {
                "SO": [3, 4, 5, 6, 7],
                "HR": [1],
                "TB": [1, 2, 3, 4],
                "H_R_RBI": [1, 2, 3, 4],
            }
            target_lines = common_lines.get(stat_name, [1, 2, 3])

            records = []
            for i in range(len(y_test)):
                probs = {}
                for line_val in target_lines:
                    p_raw = float(p_ge_stat(stat_name, mu[i], max(residual_std, 0.3), line_val))
                    if beta_cal._fitted:
                        p_cal = float(beta_cal(p_raw))
                    else:
                        p_cal = p_raw
                    probs[line_val] = max(0.001, min(0.999, p_cal))
                records.append({
                    "player_name": df_test.iloc[i].get("player_name", ""),
                    "game_pk": df_test.iloc[i].get("game_pk", 0),
                    "mu": float(mu[i]),
                    "y_actual": float(y_test[i]),
                    "probs": probs,
                })

            self.test_data[stat_name] = records
            self.models[stat_name] = (model, residual_std, beta_cal)
            print(f"  {stat_name}: {len(records)} test rows, σ={residual_std:.3f}")

        return len(self.models) > 0

    def _prob_exceeds(self, stat: str, line: int, record: dict) -> float:
        """Get P(stat >= line) for a record, or 0.001 if not available."""
        return record.get("probs", {}).get(line, 0.001)

    def _actual_exceeds(self, stat: str, line: int, record: dict) -> bool:
        """Check if actual stat value >= line."""
        return record.get("y_actual", 0) >= line

    def test_independent_parlay_calibration(self, n_simulations: int = 10000):
        """Test if independent probability assumption is valid for parlay legs.

        Picks random pairs/triples/quads of legs and checks if the actual
        joint hit rate matches the product of individual probabilities vs.
        the correlation-adjusted probability.
        """
        print(f"\n{'=' * 70}")
        print(f"  PARLAY CALIBRATION BACKTEST")
        print(f"{'=' * 70}")

        # Gather all available predictions across stats
        all_records = []  # list of (stat_name, line_val, P_model, y_actual, game_pk, player_name)
        for stat_name, records in self.test_data.items():
            common_lines = {"SO": [5], "HR": [1], "TB": [2], "H_R_RBI": [3]}
            for line_val in common_lines.get(stat_name, [1]):
                for rec in records:
                    p = self._prob_exceeds(stat_name, line_val, rec)
                    if 0.01 < p < 0.99:
                        all_records.append({
                            "stat": stat_name,
                            "line": line_val,
                            "prob": p,
                            "actual": self._actual_exceeds(stat_name, line_val, rec),
                            "game_pk": rec.get("game_pk", 0),
                            "player_name": rec.get("player_name", ""),
                        })

        print(f"  Total predictions: {len(all_records):,}")
        if len(all_records) < 100:
            print("  Insufficient data")
            return

        for n_legs in [2, 3, 4]:
            print(f"\n  {'─' * 65}")
            print(f"  {n_legs}-LEG PARLAY CALIBRATION")
            print(f"  {'─' * 65}")

            # Run many random parlay simulations
            results = []
            rng = np.random.RandomState(42)

            for _ in range(min(n_simulations, 50000)):
                # Pick n_legs random predictions without replacement
                if n_legs > len(all_records):
                    break
                indices = rng.choice(len(all_records), n_legs, replace=False)
                legs = [all_records[i] for i in indices]

                # Check for same-player same-stat (invalid parlay)
                skip = False
                for i in range(n_legs):
                    for j in range(i + 1, n_legs):
                        if (legs[i]["player_name"].lower() == legs[j]["player_name"].lower()
                                and legs[i]["stat"] == legs[j]["stat"]):
                            skip = True
                            break
                    if skip:
                        break
                if skip:
                    continue

                # Independent probability
                p_ind = float(np.prod([l["prob"] for l in legs]))

                # Correlation-adjusted probability using our module
                # We need to simulate a KalshiLeg-like object for build_correlation_pairs
                class SimLeg:
                    def __init__(self, l):
                        market_map = {"SO": "KS", "HR": "HR", "TB": "TB", "H_R_RBI": "HRR"}
                        self.player_name = l["player_name"]
                        self.game_id = str(l.get("game_pk", 0))
                        self.market_type = market_map.get(l["stat"], l["stat"])
                        self.position = "pitcher" if l["stat"] == "SO" else "hitter"
                sim_legs = [SimLeg(l) for l in legs]
                pairs = build_correlation_pairs(sim_legs)
                model_probs = [l["prob"] for l in legs]
                p_adj = joint_probability(model_probs, pairs)

                # Actual joint outcome
                actual_all = all(l["actual"] for l in legs)

                results.append({
                    "p_ind": p_ind,
                    "p_adj": p_adj,
                    "actual": actual_all,
                })

            if not results:
                continue

            df = pd.DataFrame(results)
            df["p_bin"] = pd.cut(df["p_adj"], bins=[0, 0.01, 0.05, 0.10, 0.20, 0.35, 0.50, 1.0],
                                 labels=["0-1%", "1-5%", "5-10%", "10-20%", "20-35%", "35-50%", "50%+"])

            print(f"  Simulated {len(df):,} parlays")
            print(f"\n  {'Bin':>12s}  {'N':>6s}  {'P_adj':>6s}  {'Actual':>6s}  {'P_ind':>6s}  {'Bias_adj':>8s}  {'Bias_ind':>8s}")
            print(f"  {'─' * 65}")
            for label, grp in df.groupby("p_bin", observed=True):
                n = len(grp)
                mean_adj = grp["p_adj"].mean()
                mean_ind = grp["p_ind"].mean()
                actual_rate = grp["actual"].mean()
                bias_adj = mean_adj - actual_rate
                bias_ind = mean_ind - actual_rate
                print(f"  {label:>12s}  {n:>6d}  {mean_adj:>5.1%}  {actual_rate:>5.1%}  "
                      f"{mean_ind:>5.1%}  {bias_adj:>+7.1%}  {bias_ind:>+7.1%}")

            # Overall metrics
            mean_adj = df["p_adj"].mean()
            mean_ind = df["p_ind"].mean()
            actual_rate = df["actual"].mean()
            brier_adj = float(np.mean((df["p_adj"] - df["actual"]) ** 2))
            brier_ind = float(np.mean((df["p_ind"] - df["actual"]) ** 2))
            print(f"\n  Overall:")
            print(f"    Adj:   mean P={mean_adj:.1%}  actual={actual_rate:.1%}  bias={mean_adj-actual_rate:+.1%}  Brier={brier_adj:.4f}")
            print(f"    Indep: mean P={mean_ind:.1%}  actual={actual_rate:.1%}  bias={mean_ind-actual_rate:+.1%}  Brier={brier_ind:.4f}")
            if brier_adj < brier_ind:
                print(f"    ✅ Correlation adjustment beats independent by {(brier_ind - brier_adj) / brier_ind:.0%}")
            else:
                print(f"    ❌ Correlation adjustment WORSE than independent by {(brier_adj - brier_ind) / brier_ind:.0%}")

    def test_correlated_pair_hit_rate(self):
        """Test if specific correlated pairs (same game, same player) hit at different rates.

        Compares P(A∩B) from independent product vs actual joint hit rate
        for pairs with known correlation.
        """
        print(f"\n{'=' * 70}")
        print(f"  CORRELATED PAIR HIT RATES")
        print(f"{'=' * 70}")

        # Same-player pairs (TB + HRR hitter)
        tb_records = self.test_data.get("TB", [])
        hrr_records = self.test_data.get("H_R_RBI", [])

        if tb_records and hrr_records:
            print(f"\n  Same-player: TB 2+ x HRR 3+")
            tb_by_player = {}
            for r in tb_records:
                pname = r.get("player_name", "").lower()
                tb_by_player.setdefault(pname, []).append(r)
            hrr_by_player = {}
            for r in hrr_records:
                pname = r.get("player_name", "").lower()
                hrr_by_player.setdefault(pname, []).append(r)

            same_games = 0
            joint_hits = 0
            total_pairs = 0
            for pname in set(tb_by_player.keys()) & set(hrr_by_player.keys()):
                for r1 in tb_by_player[pname][:5]:  # limit per player
                    for r2 in hrr_by_player[pname][:5]:
                        if r1["game_pk"] == r2["game_pk"]:
                            total_pairs += 1
                            tb_hit = r1.get("y_actual", 0) >= 2
                            hrr_hit = r2.get("y_actual", 0) >= 3
                            if tb_hit and hrr_hit:
                                joint_hits += 1
                            same_games += 1

            if total_pairs > 0:
                p_tb = np.mean([r.get("y_actual", 0) >= 2 for r in tb_records[:1000]])
                p_hrr = np.mean([r.get("y_actual", 0) >= 3 for r in hrr_records[:1000]])
                indep_prob = p_tb * p_hrr
                actual_joint = joint_hits / total_pairs
                rho = get_correlation("TB", "HRR", same_player=True, player_role_a="hitter")
                adj_joint = joint_probability([p_tb, p_hrr], [(0, 1, rho)])
                print(f"    P(TB 2+): {p_tb:.1%}  P(HRR 3+): {p_hrr:.1%}")
                print(f"    P_indep: {indep_prob:.1%}  P_adj(ρ={rho:.3f}): {adj_joint:.1%}  P_actual: {actual_joint:.1%}")
                print(f"    N pairs: {total_pairs}")

        # Different-game pairs (KS for pitcher + HR for hitter)
        so_records = self.test_data.get("SO", [])
        hr_records = self.test_data.get("HR", [])
        if so_records and hr_records:
            print(f"\n  Different-game: KS 6+ x HR 1+")
            p_ks = np.mean([self._prob_exceeds("SO", 6, r) for r in so_records[:1000]])
            p_hr = np.mean([self._prob_exceeds("HR", 1, r) for r in hr_records[:1000]])
            rho = get_correlation("KS", "HR")
            adj_joint = joint_probability([p_ks, p_hr], [(0, 1, rho)])
            print(f"    P(KS 6+): {p_ks:.1%}  P(HR 1+): {p_hr:.1%}")
            print(f"    P_indep: {p_ks*p_hr:.1%}  P_adj(ρ={rho:.3f}): {adj_joint:.1%}")
            print(f"    Rho from lookup: {rho:.4f} (default={0.01})")


def main():
    print("=" * 70)
    print("  PARLAY BACKTEST — Correlation-Adjusted Joint Probabilities")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M"))
    print("=" * 70)

    bt = ParlayBacktest()
    if not bt.load():
        print("Failed to load models/data")
        return

    bt.test_independent_parlay_calibration(n_simulations=20000)
    bt.test_correlated_pair_hit_rate()

    print(f"\n{'=' * 70}")
    print("  Backtest complete")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
