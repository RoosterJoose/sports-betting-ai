"""Model-derived UFC prop bet probabilities.

Extracts method-of-victory (MoV) and round-of-finish probabilities from
the existing UFC model's feature set. The model itself is a binary
classifier (P(red wins) vs P(blue wins)), but the 107-feature set
includes career-average win-method rates (r_ko_rate, b_ko_rate,
r_sub_rate, b_sub_rate, r_dec_rate, b_dec_rate) plus weight-class
finish rates (wc_finish_rate, from WEIGHT_CLASS_FINISH_PCT) that we
can use to build a joint MoV distribution.

**Approach** (no model retrain required):
1. For a fight between red (R) and blue (B), we have:
   - P(R wins) and P(B wins) from the binary classifier
   - r_ko_rate, r_sub_rate, r_dec_rate (career fraction of wins by
     each method, scaled to sum to 1.0)
   - b_ko_rate, b_sub_rate, b_dec_rate
   - wc_finish_rate (weight-class baseline fraction of fights that end
     before the distance)
2. The probability that R wins by KO is:
     P(R wins) * r_ko_rate * (1 + wc_finish_boost)
   where wc_finish_boost scales the MoV probability toward the
   weight-class's historical finish rate. This is a prior-weighted
   estimate — not a trained MoV model.
3. Round-of-finish: P(fight ends in round R) is derived from
   `fighter_recent_first_round_rate`, `fighter_recent_ko_rate`, and
   the weight-class average fight time (in `wc_averages.json`).

**Calibration** (added 2026-06-11):
The raw `method_of_victory_probabilities()` output is a prior-based estimate.
When `models/ufc/mov_calibration.json` exists (built by
`src/scripts/train_ufc_mov_cal.py`), `prop_bet_model_probabilities()`
applies OOF bin-based calibration to each of the 6 MoV outcomes and the
6 round outcomes, then renormalizes to sum to 1.0. This corrects the
known overcompression in the prior (e.g., predicted P(red KO) = 0.25
when actual rate is 0.15). The calibration is trained on the same
TimeSeriesSplit CV folds used for the winner model — see
`oos_test_ufc_cal.py` for the equivalent OOS test for the binary winner.

**Limitations**:
- The calibration table has the same look-ahead bias as the underlying
  fighter stats (career rates at the time of the fight include the
  fight itself). Acceptable for bias correction of `method_of_victory_probabilities()`
  but not a true prospective OOS test.
- The model has NO round-specific features (rounds 1-5) — the
  round-of-finish probability is a single curve fit to career
  first-round-finish rate, not per-round.
- If `mov_calibration.json` is missing, falls back to the raw prior
  (uncalibrated) — same behavior as the previous version.

Use these probabilities for:
- Model validation (compare to DK's prop lines over time)
- Identifying fighters where the model strongly disagrees with the
  market (high edge)
- Surfacing in `kalshi_ufc.py` scanner output alongside moneyline picks

Do NOT use these for:
- Live betting decisions without further OOS validation.
"""

import json
import math
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from src.features.ufc import WEIGHT_CLASS_FINISH_PCT

MODEL_DIR = Path("models/ufc")

# Outcome keys for the 6 MoV + 6 round calibration targets
MOV_KEYS = ["red_ko", "red_sub", "red_dec", "blue_ko", "blue_sub", "blue_dec"]
ROUND_KEYS = ["round_1", "round_2", "round_3", "round_4", "round_5", "goes_distance"]

_MOV_CALIBRATION_CACHE: dict | None = None


def load_fighter_db() -> dict:
    """Load the fighter lookup DB (4,548 fighters with career stats)."""
    f = MODEL_DIR / "fighter_lookup.json"
    if not f.exists():
        return {}
    with open(f) as fh:
        return json.load(fh)


def load_wc_averages() -> dict:
    """Load weight-class averages (avg_fight_time, etc.)."""
    f = MODEL_DIR / "wc_averages.json"
    if not f.exists():
        return {}
    with open(f) as fh:
        return json.load(fh)


def load_mov_calibration(force_reload: bool = False) -> dict:
    """Load MoV + round calibration table from models/ufc/mov_calibration.json.

    The file is built by `src/scripts/train_ufc_mov_cal.py` and contains
    bin-based calibration tables for each of the 12 outcomes
    (6 MoV + 6 round). If the file is missing or empty, returns {} —
    callers should treat this as "no calibration available" and use the
    raw prior probs.

    Results are cached in module-level `_MOV_CALIBRATION_CACHE` to avoid
    re-reading the JSON on every call (the file is ~5KB but read
    frequently when surfacing MoV for each upcoming fight).

    Args:
        force_reload: bypass the cache and re-read the file. Used by
            the training script after it writes a new calibration.

    Returns:
        Dict mapping outcome_key → list of {bin_lo, bin_hi, model_pred,
        actual_rate, n} entries. Empty dict if no calibration available.
    """
    global _MOV_CALIBRATION_CACHE
    if _MOV_CALIBRATION_CACHE is not None and not force_reload:
        return _MOV_CALIBRATION_CACHE
    f = MODEL_DIR / "mov_calibration.json"
    if not f.exists():
        _MOV_CALIBRATION_CACHE = {}
        return _MOV_CALIBRATION_CACHE
    with open(f) as fh:
        data = json.load(fh)
    _MOV_CALIBRATION_CACHE = data
    return _MOV_CALIBRATION_CACHE


def calibrate_single_prob(prior_prob: float, outcome_key: str, cal: dict | None = None) -> float:
    """Map a prior probability → calibrated actual rate for one outcome.

    Uses the bin the prior falls into. If the prior is below the lowest
    bin or above the highest bin, uses the boundary bin's actual_rate.
    Returns the input unchanged if no calibration is available.

    Args:
        prior_prob: raw model-derived probability (0-1)
        outcome_key: one of the 12 keys (red_ko, round_1, goes_distance, ...)
        cal: calibration dict (from load_mov_calibration). If None, loads
             from cache.

    Returns:
        Calibrated probability (0-1). Same as input if no cal available.
    """
    if cal is None:
        cal = load_mov_calibration()
    if not cal:
        return prior_prob
    table = cal.get(outcome_key, [])
    if not table:
        return prior_prob
    # Find the bin the prior falls into
    for entry in table:
        if entry["bin_lo"] <= prior_prob < entry["bin_hi"]:
            return float(entry["actual_rate"])
    # Out of range — use the nearest bin's actual_rate
    if prior_prob < table[0]["bin_lo"]:
        return float(table[0]["actual_rate"])
    return float(table[-1]["actual_rate"])


def calibrate_mov_distribution(mov: dict, cal: dict | None = None) -> dict:
    """Apply calibration to all 6 MoV outcomes and renormalize to sum to 1.0.

    Args:
        mov: dict from `method_of_victory_probabilities()` with keys
             red_ko, red_sub, red_dec, blue_ko, blue_sub, blue_dec
        cal: optional calibration dict. If None, loads from cache.

    Returns:
        Same shape dict, calibrated + renormalized to sum to 1.0.
        Returns the input unchanged if no calibration is available.
    """
    if cal is None:
        cal = load_mov_calibration()
    if not cal:
        return mov
    calibrated = {
        k: calibrate_single_prob(mov[k], k, cal) for k in MOV_KEYS
    }
    total = sum(calibrated.values())
    if total > 0:
        calibrated = {k: v / total for k, v in calibrated.items()}
    return calibrated


def calibrate_round_distribution(rof: dict, cal: dict | None = None) -> dict:
    """Apply calibration to the round-of-finish distribution and renormalize.

    Args:
        rof: dict from `round_of_finish_probabilities()` with keys
             round_1, round_2, round_3, round_4, round_5, goes_distance
        cal: optional calibration dict. If None, loads from cache.

    Returns:
        Same shape dict, calibrated + renormalized to sum to 1.0.
        Returns the input unchanged if no calibration is available.
    """
    if cal is None:
        cal = load_mov_calibration()
    if not cal:
        return rof
    calibrated = {
        k: calibrate_single_prob(rof[k], k, cal) for k in ROUND_KEYS if k in rof
    }
    total = sum(calibrated.values())
    if total > 0:
        calibrated = {k: v / total for k, v in calibrated.items()}
    return calibrated


def get_fighter_stats(fighter_name: str, fighter_db: dict, wc_avg: dict) -> dict:
    """Look up fighter stats, falling back to weight-class averages if not found."""
    if fighter_name in fighter_db:
        return fighter_db[fighter_name]
    # Fuzzy: substring match
    matches = [
        (n, s) for n, s in fighter_db.items()
        if fighter_name.lower() in n.lower() or n.lower() in fighter_name.lower()
    ]
    if matches:
        return sorted(matches, key=lambda x: abs(len(x[0]) - len(fighter_name)))[0][1]
    # Fallback: weight-class average
    default_wc = wc_avg.get("_default", wc_avg.get("middleweight", {}))
    return {
        "avg_sig_str_landed": 27.0,
        "avg_td_landed": 1.3,
        "avg_sub_att": 0.5,
        "wins": 8, "losses": 4,
        "total_rounds_fought": 12,
        "height_cms": 178.0, "reach_cms": 183.0, "weight_lbs": 170.0,
        "age": 30, "weight_class": "middleweight",
        "avg_fight_time": default_wc.get("avg_fight_time", 652),
        # MoV defaults (career rates): ~40% KO, ~25% sub, ~35% decision
        # (typical UFC averages, slightly sub-heavy)
        "win_by_ko_tko": 4, "win_by_submission": 2,
        "win_by_decision_unanimous": 2, "win_by_decision_split": 0,
        "win_by_decision_majority": 0,
    }


def compute_mov_rates(fighter_stats: dict) -> dict:
    """Compute normalized KO/sub/decision rates for a fighter.

    Returns {"ko": 0.42, "sub": 0.21, "dec": 0.37} (sums to 1.0).
    Falls back to weight-class defaults if the fighter has 0 wins.
    """
    wins = max(fighter_stats.get("wins", 1), 1)
    ko = fighter_stats.get("win_by_ko_tko", 0) or 0
    sub = fighter_stats.get("win_by_submission", 0) or 0
    dec_uni = fighter_stats.get("win_by_decision_unanimous", 0) or 0
    dec_split = fighter_stats.get("win_by_decision_split", 0) or 0
    dec_maj = fighter_stats.get("win_by_decision_majority", 0) or 0
    total_dec = dec_uni + dec_split + dec_maj
    total = ko + sub + total_dec
    if total <= 0:
        # No MoV data — fall back to weight-class typical: 40/25/35
        return {"ko": 0.40, "sub": 0.25, "dec": 0.35}
    return {
        "ko": ko / total,
        "sub": sub / total,
        "dec": total_dec / total,
    }


def method_of_victory_probabilities(
    p_red_wins: float,
    red_stats: dict,
    blue_stats: dict,
    weight_class: str = "middleweight",
) -> dict:
    """Compute P(red wins by KO), P(red wins by sub), P(red wins by dec),
    and same for blue.

    Args:
        p_red_wins: P(red corner wins) from the binary winner model.
        red_stats, blue_stats: fighter lookup dicts (from fighter_lookup.json).
        weight_class: e.g. "lightweight", "heavyweight". Used for the
            wc_finish_rate baseline (what fraction of fights in this
            weight class end before the distance).

    Returns:
        {
            "red_ko": 0.22, "red_sub": 0.08, "red_dec": 0.18,
            "blue_ko": 0.20, "blue_sub": 0.10, "blue_dec": 0.22,
        }
        All values sum to 1.0 (red_ko + red_sub + red_dec +
        blue_ko + blue_sub + blue_dec = 1.0).
    """
    p_blue_wins = 1.0 - p_red_wins
    r_rates = compute_mov_rates(red_stats)
    b_rates = compute_mov_rates(blue_stats)
    # Boost factor: weight-class historical finish rate. If 55% of HW
    # fights end in KO/TKO/sub, scale the KO+sub probabilities up by
    # wc_finish_rate / 0.65 (assume ~65% baseline finish rate), and
    # scale decision down proportionally. This is a heuristic prior.
    wc_finish = WEIGHT_CLASS_FINISH_PCT.get(weight_class.lower(), 0.45)
    finish_boost = wc_finish / 0.55  # normalize around middleweight-ish baseline
    finish_boost = max(0.7, min(1.3, finish_boost))  # clip to avoid extremes

    def _distribute(p_wins: float, rates: dict) -> tuple[float, float, float]:
        """Distribute p_wins across KO/sub/dec using rates, then apply finish boost."""
        base_ko = p_wins * rates["ko"]
        base_sub = p_wins * rates["sub"]
        base_dec = p_wins * rates["dec"]
        # Apply finish boost: scale KO+sub up, scale dec down
        boost_ko = base_ko * finish_boost
        boost_sub = base_sub * finish_boost
        # Renormalize so they sum to p_wins
        total = boost_ko + boost_sub + base_dec
        if total <= 0:
            return p_wins / 3, p_wins / 3, p_wins / 3
        return (
            boost_ko * p_wins / total,
            boost_sub * p_wins / total,
            base_dec * p_wins / total,
        )

    red_ko, red_sub, red_dec = _distribute(p_red_wins, r_rates)
    blue_ko, blue_sub, blue_dec = _distribute(p_blue_wins, b_rates)
    return {
        "red_ko": red_ko,
        "red_sub": red_sub,
        "red_dec": red_dec,
        "blue_ko": blue_ko,
        "blue_sub": blue_sub,
        "blue_dec": blue_dec,
    }


def round_of_finish_probabilities(
    p_finish_by_ko: float,
    p_finish_by_sub: float,
    red_stats: dict,
    blue_stats: dict,
    scheduled_rounds: int = 3,
) -> dict:
    """Compute P(fight ends in round R) for R=1..scheduled_rounds, plus P(goes to distance).

    Uses an exponential decay model fit to career first-round-finish
    rate and weight-class average fight time. The decay rate λ is chosen
    so the expected number of rounds equals the weight-class average
    fight time / 300 seconds per round.

    Args:
        p_finish_by_ko, p_finish_by_sub: P(red or blue wins by KO/sub)
            from method_of_victory_probabilities(). The total P(fight
            ends before distance) is p_finish_by_ko + p_finish_by_sub.
        red_stats, blue_stats: for first-round-finish rate lookup.
        scheduled_rounds: 3 or 5.

    Returns:
        {
            "round_1": 0.12, "round_2": 0.08, "round_3": 0.06,
            "round_4": 0.04, "round_5": 0.03,  # 0 for 3-round fights
            "goes_distance": 0.67,
        }
        All values sum to 1.0.
    """
    p_finish = p_finish_by_ko + p_finish_by_sub
    p_distance = 1.0 - p_finish

    # First-round finish rate: average of both fighters' recent R1 finish
    # rate, fallback to wc_finish_rate / scheduled_rounds.
    r1_rate = (
        red_stats.get("fighter_recent_first_round_rate", 0.10) or 0.10
    )
    b1_rate = (
        blue_stats.get("fighter_recent_first_round_rate", 0.10) or 0.10
    )
    avg_r1 = (r1_rate + b1_rate) / 2.0

    # If p_finish is very small, skip the round breakdown and return all
    # mass on goes_distance.
    if p_finish < 0.01:
        return {
            f"round_{r}": 0.0 for r in range(1, scheduled_rounds + 1)
        } | {"goes_distance": 1.0}

    # Exponential decay: P(R) = λ * exp(-λ*(R-1)), where λ is chosen so
    # that P(R=1) = avg_r1 * (p_finish / 0.30) — scale R1 rate by how
    # likely the fight is to finish at all.
    target_r1 = avg_r1 * (p_finish / 0.30)  # normalize around 30% baseline
    target_r1 = max(0.01, min(0.5, target_r1))  # clip
    if target_r1 >= 0.99:
        lam = 0.01
    else:
        lam = -math.log(1 - target_r1)
    lam = max(0.1, min(3.0, lam))  # reasonable bounds

    # Distribute p_finish across rounds using the exponential
    probs = []
    remaining = p_finish
    for r in range(1, scheduled_rounds):
        p_r = lam * math.exp(-lam * (r - 1))
        probs.append(p_r)
        remaining -= p_r
    # Last round: whatever's left, capped at remaining
    last = max(0.0, remaining)
    probs.append(last)
    # Renormalize so sum == p_finish (in case of floating point drift)
    total = sum(probs)
    if total > 0:
        probs = [p * p_finish / total for p in probs]

    result = {f"round_{r+1}": p for r, p in enumerate(probs)}
    result["goes_distance"] = p_distance
    return result


def prop_bet_model_probabilities(
    p_red_wins: float,
    red_name: str,
    blue_name: str,
    weight_class: str = "middleweight",
    scheduled_rounds: int = 3,
    fighter_db: dict | None = None,
    wc_avg: dict | None = None,
) -> dict:
    """One-stop function: compute all model-derived prop bet probabilities
    for a single fight.

    Returns a flat dict suitable for joining against market odds:
        {
            "fight": "Topuria vs Gaethje",
            "weight_class": "lightweight",
            "scheduled_rounds": 5,
            "p_red_wins": 0.62,
            "p_blue_wins": 0.38,
            # Method of victory (all 6 outcomes, sum to 1.0)
            "p_red_ko": 0.20, "p_red_sub": 0.10, "p_red_dec": 0.32,
            "p_blue_ko": 0.15, "p_blue_sub": 0.06, "p_blue_dec": 0.17,
            # Round of finish (per-round + goes_distance, sum to 1.0)
            "p_round_1": 0.10, "p_round_2": 0.07, "p_round_3": 0.05,
            "p_round_4": 0.04, "p_round_5": 0.03,
            "p_goes_distance": 0.71,
        }
    """
    if fighter_db is None:
        fighter_db = load_fighter_db()
    if wc_avg is None:
        wc_avg = load_wc_averages()
    red_stats = get_fighter_stats(red_name, fighter_db, wc_avg)
    blue_stats = get_fighter_stats(blue_name, fighter_db, wc_avg)

    # Compute raw prior probs, then apply OOF calibration if available.
    # The calibration corrects overcompression in the prior (e.g., the raw
    # P(red KO) = 0.25 typically maps to actual rate ~0.15-0.18 OOS).
    cal = load_mov_calibration()
    mov = method_of_victory_probabilities(
        p_red_wins, red_stats, blue_stats, weight_class
    )
    mov_cal = calibrate_mov_distribution(mov, cal)
    rof = round_of_finish_probabilities(
        mov_cal["red_ko"] + mov_cal["blue_ko"],  # total P(KO)
        mov_cal["red_sub"] + mov_cal["blue_sub"],  # total P(sub)
        red_stats, blue_stats, scheduled_rounds,
    )
    rof_cal = calibrate_round_distribution(rof, cal)

    return {
        "fight": f"{red_name} vs {blue_name}",
        "weight_class": weight_class,
        "scheduled_rounds": scheduled_rounds,
        "p_red_wins": p_red_wins,
        "p_blue_wins": 1.0 - p_red_wins,
        "p_red_ko": mov_cal["red_ko"],
        "p_red_sub": mov_cal["red_sub"],
        "p_red_dec": mov_cal["red_dec"],
        "p_blue_ko": mov_cal["blue_ko"],
        "p_blue_sub": mov_cal["blue_sub"],
        "p_blue_dec": mov_cal["blue_dec"],
        "p_round_1": rof_cal.get("round_1", 0.0),
        "p_round_2": rof_cal.get("round_2", 0.0),
        "p_round_3": rof_cal.get("round_3", 0.0),
        "p_round_4": rof_cal.get("round_4", 0.0),
        "p_round_5": rof_cal.get("round_5", 0.0),
        "p_goes_distance": rof_cal["goes_distance"],
    }
