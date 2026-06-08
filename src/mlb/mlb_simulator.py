"""
Full-Game Monte Carlo MLB Simulator — extends the F5 PA-outcome model to
9-inning games, tracks player-level stats across 5,000 simulations,
and outputs P(stat >= line) for player props (K, TB, HR, H).

Architecture:
  1. PA Outcome Model (8-class LGBM) predicts single-PA outcome
  2. Markov chain simulates each PA → resolves runners/outs/runs
  3. Tracks K per pitcher, TB/HR/H per batter across all sims
  4. Bullpen replaces starter after ~95 pitches (generic reliever)
  5. After N sims, P(stat >= X) = count(stat >= X) / N

Usage:
    from src.mlb.mlb_simulator import MLBSimulator
    sim = MLBSimulator()
    result = sim.simulate_game(away_pitcher, home_pitcher, away_batters, home_batters)
    print(result["away_pitcher"]["k_prob"][5])  # P(K >= 5)
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.settings import PROJECT_ROOT
from src.mlb.f5_simulator import resolve_outcome
from src.models.distributions import p_ge_stat

MODEL_DIR = PROJECT_ROOT / "models" / "mlb"

# 8 outcome classes
OUTCOME_NAMES = {0: "OUT", 1: "1B", 2: "2B", 3: "3B", 4: "HR", 5: "BB", 6: "HBP", 7: "K"}

# Default reliever: average MLB pitcher stats
DEFAULT_RELIEVER = {
    "pitcher_k_rate_prior": 0.22,
    "pitcher_bb_rate_prior": 0.08,
    "pitcher_hr_rate_prior": 0.03,
    "pitcher_avg_ev_against": 89.0,
    "pitcher_avg_la_against": 12.0,
    "pitcher_hard_hit_against": 0.35,
    "pitcher_gb_rate_against": 0.44,
    "pitcher_fb_rate_against": 0.26,
    "pitcher_fip": 4.20,
    "pitcher_k9": 8.5,
    "pitcher_bb9": 3.0,
    "pitcher_hr9": 1.1,
    "pitcher_whip": 1.30,
    "pitcher_k_bb_pct": 0.20,
    "p_throws": "R",  # pitcher handedness: R or L
}

# Default batter: MLB average hitter
DEFAULT_BATTER = {
    "batter_k_rate_prior": 0.22,
    "batter_bb_rate_prior": 0.08,
    "batter_avg_ev_prior": 89.0,
    "batter_avg_la_prior": 12.0,
    "batter_hard_hit_rate_prior": 0.35,
    "batter_gb_rate_prior": 0.44,
    "batter_fb_rate_prior": 0.25,
    "batter_ld_rate_prior": 0.20,
    "stand": "R",  # batter stance: R or L
}

# Average PA count per game for a starter (~27 outs + extras = ~28 PAs faced)
# Pitcher is pulled after ~95 pitches or ~25 batters faced
DEFAULT_MAX_BF = 25


def compute_max_bf(pitcher_feats: dict) -> int:
    """Determine how many batters a starter faces before being pulled.

    Uses K/9 as the primary signal — higher K/9 = more efficient =
    deeper into the game.  FIP is a secondary adjustment for on-field
    performance.

    Thresholds (based on 2024-25 MLB averages):
      - Elite (K/9 ≥ 10.0): 28 BF ~ 7 IP
      - Good   (K/9 ≥ 8.5):  25 BF ~ 6 IP
      - Avg    (K/9 ≥ 7.0):  22 BF ~ 5.5 IP
      - Below  (K/9 ≥ 6.0):  19 BF ~ 5 IP
      - Weak   (K/9 < 6.0):  16 BF ~ 4 IP
    """
    k9 = pitcher_feats.get("pitcher_k9", 8.5)
    fip = pitcher_feats.get("pitcher_fip", 4.20)

    if k9 >= 10.0:
        bf = 28
    elif k9 >= 8.5:
        bf = 25
    elif k9 >= 7.0:
        bf = 22
    elif k9 >= 6.0:
        bf = 19
    else:
        bf = 16

    # FIP penalty: if pitcher is performing badly, pull them 1-2 BF earlier
    if fip > 5.0:
        bf = max(bf - 3, 12)  # can't go below 12 BF
    elif fip > 4.5:
        bf = max(bf - 1, 14)

    return bf


class MLBSimulator:
    """Monte Carlo full-game MLB simulator using 8-class PA outcome model.

    Simulates 9-inning games play-by-play, tracking player-level stats
    across thousands of simulations for player prop estimation.
    """

    def __init__(self, model_path: Path | None = None):
        import lightgbm as lgb

        if model_path is None:
            model_path = MODEL_DIR / "f5_pa_outcome.txt"

        if not model_path.exists():
            msg = f"PA outcome model not found at {model_path}. Run: python -m src.mlb.f5_pa_outcome"
            raise FileNotFoundError(msg)

        self.model = lgb.Booster(model_file=str(model_path))

        meta_path = model_path.with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path) as f:
                self.meta = json.load(f)
        else:
            self.meta = {"feature_cols": []}

        self.feature_cols = self.meta.get("feature_cols", [])
        print(f"  PA model loaded: {self.meta.get('n_samples', '?')} samples, "
              f"{len(self.feature_cols)} features, "
              f"acc={self.meta.get('test_accuracy', '?'):.1%}", flush=True)

    # ── Feature vector building ──────────────────────────────────

    def build_pa_features(
        self,
        pitcher_feats: dict[str, float],
        batter_feats: dict[str, float],
        outs: int = 0,
        runners_on: int = 0,
        is_home: bool = False,
        park_factor_k: float = 1.0,
        park_factor_hr: float = 1.0,
        same_hand: int = 0,
    ) -> dict[str, float]:
        """Build the full 28-feature vector for a single PA."""
        features = {
            "is_home": float(is_home),
            "runners_on": float(runners_on),
            "park_factor_k": park_factor_k,
            "park_factor_hr": park_factor_hr,
            "same_hand": float(same_hand),
            "umpire_zone_factor": 1.0,
            # Pitcher FIP stats
            "pitcher_fip": pitcher_feats.get("pitcher_fip", DEFAULT_RELIEVER["pitcher_fip"]),
            "pitcher_k9": pitcher_feats.get("pitcher_k9", DEFAULT_RELIEVER["pitcher_k9"]),
            "pitcher_bb9": pitcher_feats.get("pitcher_bb9", DEFAULT_RELIEVER["pitcher_bb9"]),
            "pitcher_hr9": pitcher_feats.get("pitcher_hr9", DEFAULT_RELIEVER["pitcher_hr9"]),
            "pitcher_whip": pitcher_feats.get("pitcher_whip", DEFAULT_RELIEVER["pitcher_whip"]),
            "pitcher_k_bb_pct": pitcher_feats.get("pitcher_k_bb_pct", DEFAULT_RELIEVER["pitcher_k_bb_pct"]),
            # Pitcher rolling stats
            "pitcher_k_rate_prior": pitcher_feats.get("pitcher_k_rate_prior", DEFAULT_RELIEVER["pitcher_k_rate_prior"]),
            "pitcher_bb_rate_prior": pitcher_feats.get("pitcher_bb_rate_prior", DEFAULT_RELIEVER["pitcher_bb_rate_prior"]),
            "pitcher_hr_rate_prior": pitcher_feats.get("pitcher_hr_rate_prior", DEFAULT_RELIEVER["pitcher_hr_rate_prior"]),
            "pitcher_avg_ev_against": pitcher_feats.get("pitcher_avg_ev_against", DEFAULT_RELIEVER["pitcher_avg_ev_against"]),
            "pitcher_avg_la_against": pitcher_feats.get("pitcher_avg_la_against", DEFAULT_RELIEVER["pitcher_avg_la_against"]),
            "pitcher_hard_hit_against": pitcher_feats.get("pitcher_hard_hit_against", DEFAULT_RELIEVER["pitcher_hard_hit_against"]),
            "pitcher_gb_rate_against": pitcher_feats.get("pitcher_gb_rate_against", DEFAULT_RELIEVER["pitcher_gb_rate_against"]),
            "pitcher_fb_rate_against": pitcher_feats.get("pitcher_fb_rate_against", DEFAULT_RELIEVER["pitcher_fb_rate_against"]),
            # Batter rolling stats
            "batter_k_rate_prior": batter_feats.get("batter_k_rate_prior", DEFAULT_BATTER["batter_k_rate_prior"]),
            "batter_bb_rate_prior": batter_feats.get("batter_bb_rate_prior", DEFAULT_BATTER["batter_bb_rate_prior"]),
            "batter_avg_ev_prior": batter_feats.get("batter_avg_ev_prior", DEFAULT_BATTER["batter_avg_ev_prior"]),
            "batter_avg_la_prior": batter_feats.get("batter_avg_la_prior", DEFAULT_BATTER["batter_avg_la_prior"]),
            "batter_hard_hit_rate_prior": batter_feats.get("batter_hard_hit_rate_prior", DEFAULT_BATTER["batter_hard_hit_rate_prior"]),
            "batter_gb_rate_prior": batter_feats.get("batter_gb_rate_prior", DEFAULT_BATTER["batter_gb_rate_prior"]),
            "batter_fb_rate_prior": batter_feats.get("batter_fb_rate_prior", DEFAULT_BATTER["batter_fb_rate_prior"]),
            "batter_ld_rate_prior": batter_feats.get("batter_ld_rate_prior", DEFAULT_BATTER["batter_ld_rate_prior"]),
        }
        return features

    def predict_pa(self, features: dict[str, float]) -> np.ndarray:
        """Predict 8-class outcome probabilities for a single PA."""
        vec = np.array([features.get(c, 0) for c in self.feature_cols], dtype=float).reshape(1, -1)
        return self.model.predict(vec)[0]

    # ── Core simulation ──────────────────────────────────────────

    def simulate_game(
        self,
        away_pitcher: dict[str, float],
        home_pitcher: dict[str, float],
        away_batters: list[dict[str, float]],
        home_batters: list[dict[str, float]],
        n_sims: int = 5000,
        away_bullpen: dict[str, float] | None = None,
        home_bullpen: dict[str, float] | None = None,
        home_park_k: float = 1.0,
        home_park_hr: float = 1.0,
        max_innings: int = 12,
        use_variable_bf: bool = True,
    ) -> dict[str, Any]:
        """Simulate a full game N times and return player stat distributions.

        Supports 9-inning regulation plus extra innings (up to *max_innings*)
        with the Manfred runner rule (runner on 2nd to start each extra half-inning).

        Args:
            away_pitcher: Feature dict for away starting pitcher
            home_pitcher: Feature dict for home starting pitcher
            away_batters: List of 9 batter feature dicts for away team (batting order)
            home_batters: List of 9 batter feature dicts for home team (batting order)
            n_sims: Number of Monte Carlo simulations
            away_bullpen: Team-specific bullpen for away team (defaults to DEFAULT_RELIEVER)
            home_bullpen: Team-specific bullpen for home team (defaults to DEFAULT_RELIEVER)
            home_park_k: Park factor for Ks at home stadium
            home_park_hr: Park factor for HRs at home stadium
            max_innings: Maximum innings before declaring tie (default 12)
            use_variable_bf: If True, compute BF limit per starter from pitcher features

        Returns:
            dict with keys:
                away_pitcher: {name, k_counts, k_probs, ...}
                home_pitcher: {name, k_counts, k_probs, ...}
                away_batters: [{name, tb_probs, hr_probs, h_probs}, ...]
                home_batters: [{name, tb_probs, hr_probs, h_probs}, ...]
        """
        # Initialize stat trackers for each player
        # Per simulation: we track cumulative stats
        away_p_name = away_pitcher.get("name", "Away SP")
        home_p_name = home_pitcher.get("name", "Home SP")

        # Stat arrays: shape (n_sims,) — one value per full-game simulation
        away_k = np.zeros(n_sims, dtype=np.int32)
        home_k = np.zeros(n_sims, dtype=np.int32)

        # Batter stats: one array per batter per stat
        n_away_batters = len(away_batters)
        n_home_batters = len(home_batters)
        away_tb = np.zeros((n_away_batters, n_sims), dtype=np.int32)
        away_hr = np.zeros((n_away_batters, n_sims), dtype=np.int32)
        away_h = np.zeros((n_away_batters, n_sims), dtype=np.int32)
        home_tb = np.zeros((n_home_batters, n_sims), dtype=np.int32)
        home_hr = np.zeros((n_home_batters, n_sims), dtype=np.int32)
        home_h = np.zeros((n_home_batters, n_sims), dtype=np.int32)

        # Batter PA counts (to know which batter index for each slot)
        # For each sim, track how many times each batter slot has batted
        # This handles lineup rotation correctly

        for sim in range(n_sims):
            if sim % 1000 == 0 and sim > 0:
                print(f"    Sim {sim}/{n_sims}...", flush=True)

            # Game state
            home_score = 0
            away_score = 0

            # Compute per-pitcher BF limits (variable starter quality)
            away_max_bf = compute_max_bf(away_pitcher) if use_variable_bf else DEFAULT_MAX_BF
            home_max_bf = compute_max_bf(home_pitcher) if use_variable_bf else DEFAULT_MAX_BF

            # Resolve bullpen profiles (team-specific or default)
            away_bullpen_feats = away_bullpen if away_bullpen is not None else DEFAULT_RELIEVER
            home_bullpen_feats = home_bullpen if home_bullpen is not None else DEFAULT_RELIEVER

            # Track BF for starter replacement
            away_bf = 0
            home_bf = 0
            current_away_p = away_pitcher
            current_home_p = home_pitcher

            # Batter order tracking: index into lineup array
            away_order_idx = 0
            home_order_idx = 0

            # Per-sim stat accumulators
            sim_away_k = 0
            sim_home_k = 0
            sim_away_tb = np.zeros(n_away_batters, dtype=np.int32)
            sim_away_hr = np.zeros(n_away_batters, dtype=np.int32)
            sim_away_h = np.zeros(n_away_batters, dtype=np.int32)
            sim_home_tb = np.zeros(n_home_batters, dtype=np.int32)
            sim_home_hr = np.zeros(n_home_batters, dtype=np.int32)
            sim_home_h = np.zeros(n_home_batters, dtype=np.int32)

            # Simulate 9 innings (top + bottom each)
            # Bottom of 9th skipped if home team is winning
            # ── Innings loop (9 regulation + extra as needed) ──────────────
            max_half_innings = max_innings * 2
            for half_inning in range(max_half_innings):
                is_top = half_inning % 2 == 0
                inning = (half_inning // 2) + 1

                if inning > max_innings:
                    break

                # Skip bottom half if home is already winning (regulation and extras)
                if not is_top:
                    if home_score > away_score:
                        break  # Home already winning — no bottom half needed

                # Determine pitcher and lineup
                if is_top:
                    pitcher = current_away_p
                    lineup = away_batters
                    batter_idx = away_order_idx
                else:
                    pitcher = current_home_p
                    lineup = home_batters
                    batter_idx = home_order_idx

                # State for this half-inning
                state = {
                    "outs": 0,
                    # Manfred runner: runner on 2nd to start each extra half-inning
                    "on_1b": False,
                    "on_2b": inning > 9,  # extra innings = Manfred runner on 2nd
                    "on_3b": False,
                    "home_score": home_score, "away_score": away_score,
                    "is_top": is_top,
                }

                pa_count = 0
                while state["outs"] < 3 and pa_count < 20:
                    pa_count += 1

                    # Get current batter
                    batter_feats = lineup[batter_idx % len(lineup)]
                    batter_idx += 1

                    # Track BF for starter replacement (per-pitcher BF limit)
                    if is_top:
                        away_bf += 1
                        if away_bf > away_max_bf and current_away_p is away_pitcher:
                            current_away_p = away_bullpen_feats
                    else:
                        home_bf += 1
                        if home_bf > home_max_bf and current_home_p is home_pitcher:
                            current_home_p = home_bullpen_feats

                    # Data-driven same_hand: compare pitcher throws vs batter stands
                    p_throws = pitcher.get("p_throws", "R")
                    stand = batter_feats.get("stand", "R")
                    same_hand = 1 if p_throws == stand else 0

                    # Build features and predict
                    pa_feats = self.build_pa_features(
                        pitcher, batter_feats,
                        outs=state["outs"],
                        runners_on=int(state["on_1b"]) + int(state["on_2b"]) + int(state["on_3b"]),
                        is_home=not is_top,
                        park_factor_k=home_park_k,
                        park_factor_hr=home_park_hr,
                        same_hand=same_hand,
                    )
                    probs = self.predict_pa(pa_feats)
                    outcome = np.random.choice(8, p=probs)

                    # Track player stats before resolving state
                    batter_slot = (batter_idx - 1) % len(lineup)

                    if outcome == 7:  # K
                        if is_top:
                            sim_away_k += 1
                        else:
                            sim_home_k += 1
                    elif outcome == 1:  # 1B (single)
                        if is_top:
                            sim_away_tb[batter_slot] += 1
                            sim_away_h[batter_slot] += 1
                        else:
                            sim_home_tb[batter_slot] += 1
                            sim_home_h[batter_slot] += 1
                    elif outcome == 2:  # 2B
                        tb_add = 2
                        if is_top:
                            sim_away_tb[batter_slot] += tb_add
                            sim_away_h[batter_slot] += 1
                        else:
                            sim_home_tb[batter_slot] += tb_add
                            sim_home_h[batter_slot] += 1
                    elif outcome == 3:  # 3B
                        tb_add = 3
                        if is_top:
                            sim_away_tb[batter_slot] += tb_add
                            sim_away_h[batter_slot] += 1
                        else:
                            sim_home_tb[batter_slot] += tb_add
                            sim_home_h[batter_slot] += 1
                    elif outcome == 4:  # HR
                        tb_add = 4
                        hr_add = 1
                        if is_top:
                            sim_away_tb[batter_slot] += tb_add
                            sim_away_hr[batter_slot] += hr_add
                            sim_away_h[batter_slot] += 1
                        else:
                            sim_home_tb[batter_slot] += tb_add
                            sim_home_hr[batter_slot] += hr_add
                            sim_home_h[batter_slot] += 1

                    # Resolve PA outcome (runners, outs, runs)
                    state = resolve_outcome(outcome, state)

                # Update game score and batter order
                home_score = state["home_score"]
                away_score = state["away_score"]

                if is_top:
                    away_order_idx = batter_idx
                else:
                    home_order_idx = batter_idx

            # Store per-sim results
            away_k[sim] = sim_away_k
            home_k[sim] = sim_home_k
            away_tb[:, sim] = sim_away_tb
            away_hr[:, sim] = sim_away_hr
            away_h[:, sim] = sim_away_h
            home_tb[:, sim] = sim_home_tb
            home_hr[:, sim] = sim_home_hr
            home_h[:, sim] = sim_home_h

        # ── Build results ──────────────────────────────────────────
        def _build_probs(counts: np.ndarray) -> dict[int, float]:
            """Build empirical P(stat >= X) from simulation counts."""
            max_val = int(counts.max())
            probs = {}
            cumulative = 0
            for x in range(max_val, -1, -1):
                cumulative += int((counts == x).sum())
                probs[x] = cumulative / n_sims
            return probs

        def _build_dist_probs(mu: float, sigma: float, stat_name: str,
                              max_val: int = 20) -> dict[int, float]:
            """Build distribution-based P(stat >= X) using p_ge_stat().

            Uses NB/Poisson/Normal from the distributions module based
            on *stat_name*, giving smoother tail estimates than the
            empirical counts (which are noisy at low N_sims).
            """
            probs = {}
            for x in range(max_val, -1, -1):
                p = p_ge_stat(stat_name, mu, sigma, x)
                if p > 0.001:
                    probs[x] = p
            return probs

        result: dict[str, Any] = {
            "n_sims": n_sims,
            "away_pitcher": {
                "name": away_p_name,
                "k_counts": away_k.tolist(),
                "k_mean": float(away_k.mean()),
                "k_std": float(away_k.std()),
                "k_probs": _build_probs(away_k),           # empirical
                "k_probs_dist": _build_dist_probs(          # NB/Poisson-smoothed
                    float(away_k.mean()), float(away_k.std()), "SO"),
            },
            "home_pitcher": {
                "name": home_p_name,
                "k_counts": home_k.tolist(),
                "k_mean": float(home_k.mean()),
                "k_std": float(home_k.std()),
                "k_probs": _build_probs(home_k),
                "k_probs_dist": _build_dist_probs(
                    float(home_k.mean()), float(home_k.std()), "SO"),
            },
            "away_batters": [],
            "home_batters": [],
        }

        for i in range(n_away_batters):
            bname = away_batters[i].get("name", f"Away #{i+1}")
            tb_mu = float(away_tb[i].mean())
            tb_sigma = float(away_tb[i].std())
            hr_mu = float(away_hr[i].mean())
            hr_sigma = float(away_hr[i].std())
            h_mu = float(away_h[i].mean())
            h_sigma = float(away_h[i].std())
            result["away_batters"].append({
                "name": bname,
                "tb_probs": _build_probs(away_tb[i]),
                "hr_probs": _build_probs(away_hr[i]),
                "h_probs": _build_probs(away_h[i]),
                "tb_probs_dist": _build_dist_probs(tb_mu, tb_sigma, "TB"),
                "hr_probs_dist": _build_dist_probs(hr_mu, hr_sigma, "HR"),
                "h_probs_dist": _build_dist_probs(h_mu, h_sigma, "H"),
                "tb_mean": tb_mu,
                "hr_mean": hr_mu,
                "h_mean": h_mu,
            })

        for i in range(n_home_batters):
            bname = home_batters[i].get("name", f"Home #{i+1}")
            tb_mu = float(home_tb[i].mean())
            tb_sigma = float(home_tb[i].std())
            hr_mu = float(home_hr[i].mean())
            hr_sigma = float(home_hr[i].std())
            h_mu = float(home_h[i].mean())
            h_sigma = float(home_h[i].std())
            result["home_batters"].append({
                "name": bname,
                "tb_probs": _build_probs(home_tb[i]),
                "hr_probs": _build_probs(home_hr[i]),
                "h_probs": _build_probs(home_h[i]),
                "tb_probs_dist": _build_dist_probs(tb_mu, tb_sigma, "TB"),
                "hr_probs_dist": _build_dist_probs(hr_mu, hr_sigma, "HR"),
                "h_probs_dist": _build_dist_probs(h_mu, h_sigma, "H"),
                "tb_mean": tb_mu,
                "hr_mean": hr_mu,
                "h_mean": h_mu,
            })

        return result

    def compute_edge(
        self,
        stat_probs: dict[int, float],
        line: int,
        market_prob: float,
        side: str = "over",
    ) -> float | None:
        """Compute edge for a player prop given simulated probabilities.

        Args:
            stat_probs: dict mapping X -> P(stat >= X) from simulator
            line: The line value (e.g., 4.5 → int 4 or 5 depending on market)
            market_prob: Market-implied probability for the over
            side: 'over' or 'under'

        Returns:
            Edge as a decimal (0.05 = 5% edge), or None if no data
        """
        p_over = stat_probs.get(int(line), None)
        if p_over is None:
            return None
        if side == "over":
            return p_over - market_prob
        else:
            return (1 - p_over) - (1 - market_prob)


def demo():
    """Quick demo with synthetic data."""
    sim = MLBSimulator()

    # Synthetic average pitcher
    avg_pitcher = {k: v for k, v in DEFAULT_RELIEVER.items()}
    avg_pitcher["name"] = "Average SP"

    # 9 synthetic batters
    batters = []
    for i in range(9):
        b = dict(DEFAULT_BATTER)
        b["name"] = f"Batter #{i+1}"
        batters.append(b)

    print("\nRunning 100 sims (quick test)...")
    result = sim.simulate_game(
        avg_pitcher, avg_pitcher,
        batters, batters,
        n_sims=100,
    )

    print(f"\nAway SP: {result['away_pitcher']['k_mean']:.1f} K/game (σ={result['away_pitcher']['k_std']:.1f})")
    print(f"P(K >= 5): {result['away_pitcher']['k_probs'].get(5, 0):.0%}")
    print(f"P(K >= 7): {result['away_pitcher']['k_probs'].get(7, 0):.0%}")
    print(f"\nBatter #1 TB/game: {result['away_batters'][0]['tb_mean']:.2f}")
    print(f"Batter #1 P(TB >= 1): {result['away_batters'][0]['tb_probs'].get(1, 0):.0%}")
    print(f"Batter #1 P(HR >= 1): {result['away_batters'][0]['hr_probs'].get(1, 0):.0%}")

    return result


if __name__ == "__main__":
    demo()
