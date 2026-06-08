"""Multi-sport Stat Correlation Database — empirical pairwise correlations for parlay probability adjustment.

Supports MLB and NFL correlation databases.  Correlation lookup strategy:

MLB:
  1. Same player, different stat → empirical stat-to-stat ρ (from game logs)
  2. Same game, pitcher↔hitter  → cross-role ρ (e.g., p_SO ↔ h_HR)
  3. Same game, same role       → within-game team ρ
  4. Same stat, different games → 0.02  (near-zero)
  5. Default                    → 0.01  (effectively independent)

NFL:
  1. Same player, different stat → empirical ρ from position-group DB
  2. Same game                   → 0.05 (teammates mildly correlated)
  3. Same stat, different games → 0.02
  4. Default                    → 0.01

The joint probability correction uses the first-order covariance expansion:

    P(all ∩) ≈ ∏ p_i + Σ_{i<j} ρ_{ij} · σ_i · σ_j · ∏_{k≠i,j} p_k
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

# ── Database root path ──────────────────────────────────────────────────

_DB_ROOT = Path(__file__).resolve().parents[2] / "models"

# ── Correlation database cache ──────────────────────────────────────────

_CORR_DB: dict[str, dict] = {}  # sport -> db dict


def _load_db(sport: str) -> dict:
    """Load correlation database for a given sport ("mlb" or "nfl")."""
    global _CORR_DB
    if sport in _CORR_DB:
        return _CORR_DB[sport]

    db_path = _DB_ROOT / sport / "stat_correlations.json"
    if db_path.exists():
        with open(db_path) as f:
            _CORR_DB[sport] = json.load(f)
        return _CORR_DB[sport]

    # Return empty DB template
    if sport == "nfl":
        _CORR_DB[sport] = {"all_offense": {}, "qbs": {}, "rbs": {}, "wrs": {}, "tes": {}, "receivers": {}}
    else:
        _CORR_DB[sport] = {"all_players": {}, "pitchers": {}, "hitters": {}}
    return _CORR_DB[sport]


def _detect_sport(market_type: str) -> str:
    """Detect sport from market type name."""
    if market_type in MLB_STATS_MAP:
        return "mlb"
    if market_type in NFL_STATS_MAP:
        return "nfl"
    # Default: check if it looks like MLB (lowercase single stat) or NFL (uppercase)
    if market_type.isupper() and len(market_type) > 2:
        return "nfl"
    return "mlb"


# ── MLB stat-name mapping ───────────────────────────────────────────────

MLB_STATS_MAP = {
    "KS":   "so",
    "HR":   "hr",
    "TB":   "tb",
    "HRR":  "h_r_rbi",
    "BB":   "bb",
    "R":    "r",
    "RBI":  "rbi",
    "SB":   "sb",
    "ER":   "er",
    "IP":   "ip",
    "H":    "h",
    "SO":   "so",
}

MLB_PITCHER_STATS = {"KS", "SO", "ER", "H", "BB", "IP"}
MLB_HITTER_STATS = {"HR", "TB", "HRR", "R", "RBI", "SB"}

# backward-compatible aliases (used by kalshi_parlay.py)
STAT_NAME_MAP = MLB_STATS_MAP
PITCHER_STATS = MLB_PITCHER_STATS
HITTER_STATS = MLB_HITTER_STATS

# ── NFL stat-name mapping ───────────────────────────────────────────────

NFL_STATS_MAP = {
    "PASS_YDS":     "passing_yards",
    "PASS_TD":      "passing_tds",
    "PASS_ATT":     "pass_attempts",
    "INT":          "interceptions",
    "PASS_YDS+TD":  "pass_yds_td",
    "RUSH_YDS":     "rushing_yards",
    "RUSH_TD":      "rushing_tds",
    "RUSH+REC_YDS": "rush_rec_yds",
    "REC":          "receptions",
    "REC_YDS":      "receiving_yards",
    "REC_TD":       "receiving_tds",
    "TD":           "touchdowns",
    "CARRIES":      "carries",
    "TARGETS":      "targets",
}

# NFL position groups for correlation lookup
NFL_QB_STATS = {"PASS_YDS", "PASS_TD", "PASS_ATT", "INT", "PASS_YDS+TD"}
NFL_RB_STATS = {"RUSH_YDS", "RUSH_TD", "RUSH+REC_YDS", "CARRIES"}
NFL_WR_STATS = {"REC", "REC_YDS", "REC_TD", "TARGETS"}
NFL_TE_STATS = {"REC", "REC_YDS", "REC_TD"}
NFL_SKILL_STATS = {"TD", "RUSH+REC_YDS"}


# ── Core API ────────────────────────────────────────────────────────────

SAME_PLAYER_SAME_STAT = 1.0  # hard reject
DEFAULT_CORR = 0.01
SAME_STAT_DIFF_GAME = 0.02
SAME_GAME_MLB_HITTER_HITTER = 0.10
SAME_GAME_MLB_PITCHER_HITTER = -0.05
SAME_GAME_NFL = 0.05  # teammates are mildly correlated


def get_correlation(market_type_a: str, market_type_b: str,
                    same_player: bool = False,
                    same_game: bool = False,
                    player_role_a: str | None = None,
                    player_role_b: str | None = None) -> float:
    """Estimate correlation coefficient ρ between two prop markets.

    Auto-detects sport (MLB vs NFL) from market type names.

    Parameters
    ----------
    market_type_a, market_type_b : str
        Scanner market type names (e.g. "KS", "HR" or "PASS_YDS", "RUSH_YDS").
    same_player : bool
        True if both legs refer to the same player.
    same_game : bool
        True if both legs are from the same game.
    player_role_a, player_role_b : str or None
        For MLB: "pitcher" or "hitter". For NFL: player's position ("QB", "RB", "WR", "TE").

    Returns
    -------
    float
        Correlation coefficient ρ.
    """
    sport = _detect_sport(market_type_a)

    # ── Same player ─────────────────────────────────────────────────────
    if same_player:
        if market_type_a == market_type_b:
            return SAME_PLAYER_SAME_STAT
        if sport == "nfl":
            return _nfl_same_player_corr(market_type_a, market_type_b, player_role_a)
        return _mlb_same_player_corr(market_type_a, market_type_b, player_role_a)

    # ── Same game ───────────────────────────────────────────────────────
    if same_game:
        if sport == "nfl":
            return SAME_GAME_NFL
        role_a = _mlb_role(market_type_a, player_role_a)
        role_b = _mlb_role(market_type_b, player_role_b)
        if role_a == "pitcher" or role_b == "pitcher":
            return SAME_GAME_MLB_PITCHER_HITTER
        return SAME_GAME_MLB_HITTER_HITTER

    # ── Same stat, different games ──────────────────────────────────────
    if market_type_a == market_type_b:
        return SAME_STAT_DIFF_GAME

    # ── Default ─────────────────────────────────────────────────────────
    return DEFAULT_CORR


# ── Joint probability ───────────────────────────────────────────────────


def joint_probability(leg_probs: list[float], correlations: list[tuple[int, int, float]]) -> float:
    """Compute P(all legs win) for a multi-leg parlay with correlated legs.

    Uses the first-order covariance expansion.
    """
    probs = np.array(leg_probs, dtype=float)
    n = len(probs)
    if n == 0:
        return 0.0
    if n == 1:
        return float(np.clip(probs[0], 0.001, 0.999))

    p_ind = float(np.prod(probs))
    sigmas = np.sqrt(probs * (1.0 - probs))
    correction = 0.0
    for i, j, rho in correlations:
        if i < n and j < n and abs(rho) > 0:
            product_excluding_pair = p_ind / max(probs[i] * probs[j], 1e-10)
            correction += rho * sigmas[i] * sigmas[j] * product_excluding_pair

    joint = p_ind + correction
    upper = float(np.min(probs))
    lower = max(0.001, p_ind * 0.1)
    return float(np.clip(joint, lower, upper))


def market_joint_probability(leg_market_probs: list[float],
                             correlations: list[tuple[int, int, float]]) -> float:
    return joint_probability(leg_market_probs, correlations)


# ── Kelly sizing for multi-leg parlays ──────────────────────────────────


def parlay_kelly(joint_prob: float, payout_multiple: float,
                 fraction: float = 0.25) -> float:
    if payout_multiple <= 1:
        return 0.0
    q = 1.0 - joint_prob
    b = payout_multiple - 1.0
    if b <= 0:
        return 0.0
    kelly = (joint_prob * b - q) / b
    return max(0.0, kelly * fraction)


def compute_payout(leg_market_probs: list[float]) -> float:
    combined_market = float(np.prod(leg_market_probs))
    return 1.0 / combined_market if combined_market > 0 else 0.0


# ── Pairwise correlation builder ────────────────────────────────────────


def build_correlation_pairs(legs: list) -> list[tuple[int, int, float]]:
    """Build list of (i, j, ρ) pairs for all leg combinations.

    Auto-detects sport (MLB vs NFL) from the first leg's market type.
    """
    pairs = []
    n = len(legs)
    if n == 0:
        return pairs

    # Detect sport from first leg
    sport = _detect_sport(legs[0].market_type) if hasattr(legs[0], "market_type") else "mlb"

    for i in range(n):
        for j in range(i + 1, n):
            a, b = legs[i], legs[j]
            same_player = a.player_name.lower() == b.player_name.lower()
            same_game = a.game_id == b.game_id
            role_a = getattr(a, "position", None) or getattr(a, "player_role", None)
            role_b = getattr(b, "position", None) or getattr(b, "player_role", None)
            rho = get_correlation(
                a.market_type, b.market_type,
                same_player=same_player,
                same_game=same_game,
                player_role_a=role_a,
                player_role_b=role_b,
            )
            if abs(rho) > 0:
                pairs.append((i, j, rho))
    return pairs


# ── MLB internal helpers ────────────────────────────────────────────────


def _mlb_same_player_corr(stat_a: str, stat_b: str, role: str | None) -> float:
    db = _load_db("mlb")
    col_a = MLB_STATS_MAP.get(stat_a, stat_a.lower())
    col_b = MLB_STATS_MAP.get(stat_b, stat_b.lower())

    if role == "pitcher":
        key = "pitchers"
    elif role == "hitter":
        key = "hitters"
    else:
        key = "all_players"

    corr_dict = db.get(key, {})
    row = corr_dict.get(col_a, {})
    val = row.get(col_b)
    if val is not None:
        return val
    row = corr_dict.get(col_b, {})
    val = row.get(col_a)
    return abs(val) if val is not None else 0.77


def _mlb_role(market_type: str, explicit_role: str | None) -> str:
    if explicit_role:
        return explicit_role
    if market_type in MLB_PITCHER_STATS:
        return "pitcher"
    if market_type in MLB_HITTER_STATS:
        return "hitter"
    return "unknown"


# ── NFL internal helpers ────────────────────────────────────────────────

# Map player position → correlation DB group key
NFL_POSITION_GROUP_MAP = {
    "QB": "qbs",
    "RB": "rbs",
    "WR": "wrs",
    "TE": "tes",
    "FB": "rbs",  # fullbacks grouped with RBs
}

# Map stat type → preferred position group for same-player lookup
NFL_STAT_POSITION_MAP: dict[str, str] = {}
for s in NFL_QB_STATS:
    NFL_STAT_POSITION_MAP[s] = "qbs"
for s in NFL_RB_STATS:
    NFL_STAT_POSITION_MAP[s] = "rbs"
for s in NFL_WR_STATS:
    NFL_STAT_POSITION_MAP[s] = "wrs"
for s in NFL_TE_STATS:
    NFL_STAT_POSITION_MAP[s] = "tes"
for s in NFL_SKILL_STATS:
    NFL_STAT_POSITION_MAP[s] = "all_offense"


def _nfl_same_player_corr(stat_a: str, stat_b: str, position: str | None) -> float:
    """Look up empirical correlation between two stats for the same NFL player.

    Uses the position-group-specific correlation DB for more accurate estimates.
    Falls back to all_offense if the position group is not available.
    """
    db = _load_db("nfl")
    col_a = NFL_STATS_MAP.get(stat_a, stat_a.lower())
    col_b = NFL_STATS_MAP.get(stat_b, stat_b.lower())

    # Try position-specific lookup first
    if position and position.upper() in NFL_POSITION_GROUP_MAP:
        group_key = NFL_POSITION_GROUP_MAP[position.upper()]
    else:
        # Use preferred position group based on stat type
        group_key = NFL_STAT_POSITION_MAP.get(stat_a, "all_offense")

    corr_dict = db.get(group_key, {})
    row = corr_dict.get(col_a, {})
    val = row.get(col_b)
    if val is not None:
        return abs(float(val))

    # Try reverse lookup
    row = corr_dict.get(col_b, {})
    val = row.get(col_a)
    if val is not None:
        return abs(float(val))

    # Fallback: all_offense
    if group_key != "all_offense":
        corr_dict = db.get("all_offense", {})
        row = corr_dict.get(col_a, {})
        val = row.get(col_b)
        if val is not None:
            return abs(float(val))
        row = corr_dict.get(col_b, {})
        val = row.get(col_a)
        if val is not None:
            return abs(float(val))

    return 0.77  # fallback for same-player same-game
