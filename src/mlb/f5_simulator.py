"""
F5 Monte Carlo Simulator — Markov Chain PA-level Simulation.

Uses the trained 8-class PA outcome model to simulate F5 games
play-by-play, aggregating 5,000 simulations to produce win/draw probabilities.

Architecture (per research):
- 24 base-out states (8 runner configs × 3 out states)
- Each PA: predict 8-class outcome from LightGBM model
- Resolve outcome → transition to new base-out state
- Run 5,000 full simulations per game
- Aggregate to F5 win/draw probabilities

For F5: only starting pitcher matters (no bullpen transitions).
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.settings import PROJECT_ROOT

MODEL_DIR = PROJECT_ROOT / "models" / "mlb"

# 8 outcome classes (must match f5_pa_outcome.py)
OUTCOME_NAMES = {0: "OUT", 1: "1B", 2: "2B", 3: "3B", 4: "HR", 5: "BB", 6: "HBP", 7: "K"}

# Run boundaries per base-out state configuration
# Base-out states: 24 states (8 runner configs × 3 out states)
# Runner configs: 0=none, 1=1st, 2=2nd, 3=3rd, 4=1st+2nd, 5=1st+3rd, 6=2nd+3rd, 7=loaded

def runners_to_config(on_1b, on_2b, on_3b):
    """Convert runner booleans to state index 0-7."""
    if on_1b and on_2b and on_3b: return 7
    if on_1b and on_2b: return 4
    if on_1b and on_3b: return 5
    if on_2b and on_3b: return 6
    if on_1b: return 1
    if on_2b: return 2
    if on_3b: return 3
    return 0


def resolve_outcome(outcome_code, state):
    """Resolve a PA outcome given current game state.
    
    state: dict with keys: outs, on_1b, on_2b, on_3b, home_score, away_score, is_top
    
    Returns updated state dict.
    """
    outs = state["outs"]
    on_1b = state.get("on_1b", False)
    on_2b = state.get("on_2b", False)
    on_3b = state.get("on_3b", False)
    
    if outcome_code == 7:  # K
        outs += 1
    
    elif outcome_code == 5:  # BB
        # Walk: check pre-walk state, then advance runners
        # Save pre-walk state to avoid bugs from in-place modification
        pre_on_1b, pre_on_2b, pre_on_3b = on_1b, on_2b, on_3b
        if not pre_on_1b and not pre_on_2b and not pre_on_3b:
            on_1b = True; on_2b = False; on_3b = False
        elif pre_on_1b and not pre_on_2b and not pre_on_3b:
            on_1b = True; on_2b = True; on_3b = False
        elif not pre_on_1b and pre_on_2b and not pre_on_3b:
            on_1b = True; on_2b = True; on_3b = False
        elif not pre_on_1b and not pre_on_2b and pre_on_3b:
            on_1b = True; on_2b = False; on_3b = True  # 3B stays, walk loads 1B
        elif pre_on_1b and pre_on_2b and not pre_on_3b:
            on_1b = True; on_2b = True; on_3b = True
        elif pre_on_1b and not pre_on_2b and pre_on_3b:
            on_1b = True; on_2b = True; on_3b = False  # 1B→2B, 3B stays
        elif not pre_on_1b and pre_on_2b and pre_on_3b:
            on_1b = True; on_2b = True; on_3b = True
        elif pre_on_1b and pre_on_2b and pre_on_3b:
            # Bases loaded: walk forces runner home
            _score(state)
            on_1b = True; on_2b = True; on_3b = True
    
    elif outcome_code == 6:  # HBP
        on_1b = True
        # Handle forced runners
        if on_1b and on_2b:
            on_2b = True
        if on_1b and on_2b and on_3b:
            _score(state)
    
    elif outcome_code == 1:  # 1B (single)
        # Runner on 2nd scores, runner on 1st to 2nd, runner on 3rd scores
        runs = 0
        if on_3b:
            runs += 1
        if on_2b:
            runs += 1
        new_1b = True
        new_2b = on_1b  # runner on 1st advances to 2nd
        new_3b = False
        on_1b, on_2b, on_3b = new_1b, new_2b, new_3b
        for _ in range(runs):
            _score(state)
    
    elif outcome_code == 2:  # 2B (double)
        runs = 0
        if on_3b:
            runs += 1
        if on_2b:
            runs += 1
        if on_1b:
            runs += 1
        on_1b, on_2b, on_3b = False, True, False
        for _ in range(runs):
            _score(state)
    
    elif outcome_code == 3:  # 3B (triple)
        runs = 0
        if on_3b: runs += 1
        if on_2b: runs += 1
        if on_1b: runs += 1
        on_1b, on_2b, on_3b = False, False, True
        for _ in range(runs):
            _score(state)
    
    elif outcome_code == 4:  # HR (home run)
        runs = 1  # batter
        if on_3b: runs += 1
        if on_2b: runs += 1
        if on_1b: runs += 1
        on_1b, on_2b, on_3b = False, False, False
        for _ in range(runs):
            _score(state)
    
    else:  # OUT (code 0)
        outs += 1
    
    # Handle inning over
    if outs >= 3:
        state["outs"] = 3
        state["on_1b"] = False
        state["on_2b"] = False
        state["on_3b"] = False
    else:
        state["outs"] = outs
        state["on_1b"] = on_1b
        state["on_2b"] = on_2b
        state["on_3b"] = on_3b
    
    return state


def _score(state):
    """Add a run to the current batting team."""
    if state.get("is_top", True):
        state["away_score"] = state.get("away_score", 0) + 1
    else:
        state["home_score"] = state.get("home_score", 0) + 1


class F5Simulator:
    """Monte Carlo F5 simulator using 8-class PA outcome model."""
    
    def __init__(self, model_path=None):
        import lightgbm as lgb
        
        if model_path is None:
            model_path = MODEL_DIR / "f5_pa_outcome.txt"
        
        if not model_path.exists():
            print(f"  PA outcome model not found at {model_path}")
            print("  Run: python -m src.mlb.f5_pa_outcome")
            self.model = None
            self.meta = {"feature_cols": []}
            return
        
        self.model = lgb.Booster(model_file=str(model_path))
        
        meta_path = model_path.with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path) as f:
                self.meta = json.load(f)
        else:
            self.meta = {"feature_cols": []}
        
        self.feature_cols = self.meta.get("feature_cols", [])
        print(f"  Loaded PA outcome model: {self.meta.get('n_samples', '?')} training samples, "
              f"{len(self.feature_cols)} features", flush=True)
    
    def predict_pa_outcome(self, features: dict) -> np.ndarray:
        """Predict 8-class outcome probabilities for a single PA."""
        if self.model is None:
            return np.ones(8) / 8
        
        vec = np.array([features.get(c, 0) for c in self.feature_cols], dtype=float).reshape(1, -1)
        return self.model.predict(vec)[0]
    
    def simulate_half_inning(self, pitcher_features: dict, lineup_probs: list, n_sims: int = 5000) -> dict:
        """Simulate one half-inning (1/2 of an F5 game) for a single lineup.
        
        Args:
            pitcher_features: dict of pitcher feature values
            lineup_probs: list of 9 dicts, one per batter, each with batter feature values
            n_sims: number of simulations to run
        
        Returns:
            dict with mean_runs, run_distribution, etc.
        """
        if self.model is None:
            return {"mean_runs": 0.5, "run_distribution": {0: 1.0}}
        
        run_totals = []
        
        for _ in range(n_sims):
            runs = 0
            state = {"outs": 0, "on_1b": False, "on_2b": False, "on_3b": False,
                     "home_score": 0, "away_score": 0, "is_top": True}
            batter_idx = 0
            
            while state["outs"] < 3:
                # Get batter features
                batter = lineup_probs[batter_idx % len(lineup_probs)]
                batter_idx += 1
                
                # Combine pitcher + batter + context features
                pa_features = {**pitcher_features, **batter}
                pa_features["outs_when_up"] = state["outs"]
                pa_features["runners_on"] = int(state["on_1b"]) + int(state["on_2b"]) + int(state["on_3b"])
                
                # Predict outcome
                probs = self.predict_pa_outcome(pa_features)
                outcome = np.random.choice(8, p=probs)
                
                # Resolve
                state = resolve_outcome(outcome, state)
            
            runs = state.get("away_score", 0)  # Top of inning runs
            run_totals.append(runs)
        
        run_totals = np.array(run_totals)
        result = {
            "mean_runs": float(np.mean(run_totals)),
            "std_runs": float(np.std(run_totals)),
            "median_runs": float(np.median(run_totals)),
            "run_distribution": {int(k): float(v) for k, v in 
                                 zip(*np.unique(run_totals, return_counts=True))},
            "n_sims": n_sims,
        }
        return result
    
    def simulate_f5_game(self, away_pitcher_feats: dict, home_pitcher_feats: dict,
                         away_lineup: list, home_lineup: list,
                         n_sims: int = 5000) -> dict:
        """Simulate a full F5 game (5 innings or 15 outs each).
        
        Returns dict with away_win_prob, home_win_prob, tie_prob.
        """
        if self.model is None:
            return {"away_win": 0.33, "home_win": 0.34, "tie": 0.33}
        
        away_wins = 0
        home_wins = 0
        ties = 0
        
        for sim in range(n_sims):
            game_state = {"outs": 0, "on_1b": False, "on_2b": False, "on_3b": False,
                          "home_score": 0, "away_score": 0}
            
            # Track batter order separately for each team
            away_batter_idx = 0
            home_batter_idx = 0
            
            # F5: 5 half-innings (away bats top 1-5, home bats bottom 1-5)
            # But if home team is winning after top 5, game ends at bottom 5
            for half_inning in range(9):  # Max 9 half-innings for F5
                if half_inning >= 5 and game_state["home_score"] != game_state["away_score"]:
                    break  # Game is over (home team doesn't need to bat)
                
                is_top = half_inning % 2 == 0
                inning = (half_inning // 2) + 1
                
                if inning > 5:
                    break
                
                pitcher = away_pitcher_feats if is_top else home_pitcher_feats
                lineup = away_lineup if is_top else home_lineup
                batter_idx = away_batter_idx if is_top else home_batter_idx
                
                state = {"outs": 0, "on_1b": False, "on_2b": False, "on_3b": False,
                         "home_score": game_state["home_score"],
                         "away_score": game_state["away_score"],
                         "is_top": is_top}
                
                local_batter_count = 0
                while state["outs"] < 3 and local_batter_count < 20:  # Safety limit
                    local_batter_count += 1
                    batter = lineup[batter_idx % len(lineup)]
                    batter_idx += 1
                    
                    pa_features = {**pitcher, **batter}
                    pa_features["outs_when_up"] = state["outs"]
                    pa_features["runners_on"] = int(state["on_1b"]) + int(state["on_2b"]) + int(state["on_3b"])
                    
                    probs = self.predict_pa_outcome(pa_features)
                    outcome = np.random.choice(8, p=probs)
                    state = resolve_outcome(outcome, state)
                
                # Update game state
                game_state["home_score"] = state["home_score"]
                game_state["away_score"] = state["away_score"]
                
                if is_top:
                    away_batter_idx = batter_idx
                else:
                    home_batter_idx = batter_idx
            
            # Tally result
            if game_state["away_score"] > game_state["home_score"]:
                away_wins += 1
            elif game_state["home_score"] > game_state["away_score"]:
                home_wins += 1
            else:
                ties += 1
        
        return {
            "away_win": away_wins / n_sims,
            "home_win": home_wins / n_sims,
            "tie": ties / n_sims,
            "n_sims": n_sims,
        }
    
    def scan_f5_markets(self, away_pitcher: dict, home_pitcher: dict,
                        away_code: str, home_code: str,
                        n_sims: int = 5000) -> dict:
        """Run F5 simulation for a specific matchup and return probabilities."""
        # Create lineup context (9 batters with average features)
        # In v1, we use the pitcher's opponent context (avg batter)
        # Future versions will use actual projected lineups
        
        # Average batter features (we use zeros as default, model handles them)
        avg_batter = {}
        for c in self.feature_cols:
            if c.startswith("b_"):
                avg_batter[c] = 0.0
        
        avg_lineup = [avg_batter.copy() for _ in range(9)]
        
        # Run simulation
        result = self.simulate_f5_game(
            away_pitcher, home_pitcher,
            avg_lineup, avg_lineup,
            n_sims=n_sims
        )
        
        # Map to outcome labels
        result["away_prob"] = result["away_win"]
        result["home_prob"] = result["home_win"]
        result["tie_prob"] = result["tie"]
        
        return result


def test_simulator():
    """Quick test of the simulator with random features."""
    sim = F5Simulator()
    if sim.model is None:
        print("No model found — run python -m src.mlb.f5_pa_outcome first")
        return
    
    # Synthetic test features
    test_feats = {c: 0.0 for c in sim.feature_cols}
    
    # Run a quick test
    result = sim.simulate_f5_game(
        test_feats, test_feats,
        [{c: 0.0 for c in sim.feature_cols} for _ in range(9)],
        [{c: 0.0 for c in sim.feature_cols} for _ in range(9)],
        n_sims=100  # Quick test
    )
    
    print(f"\nTest simulation result (100 sims):")
    print(f"  Away win: {result['away_win']:.1%}")
    print(f"  Home win: {result['home_win']:.1%}")
    print(f"  Tie:      {result['tie']:.1%}")
    
    return result


if __name__ == "__main__":
    test_simulator()
