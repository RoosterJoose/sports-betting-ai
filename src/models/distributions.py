"""Distribution mapping functions for player stat exceedance probabilities.

Replaces the normal CDF with distribution-appropriate mappings:
  - **Negative Binomial** for overdispersed volume stats
    (passing yards, rushing yards, receiving yards, receptions)
  - **Poisson** for rare-event count stats
    (touchdowns, interceptions, steals, blocks, home runs)
  - **Log-Normal** fallback for continuous stats where normal CDF is reasonable
    (mainly used when sigma² <= mu and NB degenerates)

Usage:
    from src.models.distributions import p_ge_stat, NB_STATS, POISSON_STATS

    p = p_ge_stat("PASS_YDS", mu=250.0, sigma=45.0, line_val=275)
"""

from typing import Optional

import numpy as np
from scipy.stats import nbinom, poisson

# ── Stat classification ──────────────────────────────────────────────────────

# Volume stats → Negative Binomial (overdispersed, right-skewed)
NB_STATS: set[str] = {
    "PASS_YDS", "RUSH_YDS", "REC_YDS", "REC", "PASS_ATT",
    "RUSH+REC_YDS", "PASS_YDS+TD",
    # NBA volume
    "PTS", "REB", "AST", "MIN", "FGA", "FGM", "FTA", "FTM", "FG3A", "FG3M",
    "PR", "PA", "RA", "PRA", "SB", "FPTS",
    # MLB count stats (volume counts, overdispersed from starter-pull logic)
    "SO", "K", "TB", "H", "ER", "BB", "R", "RBI", "H_R_RBI", "IP", "OUTS",
}

# Rare-event stats → Poisson (low count, zero-inflated)
POISSON_STATS: set[str] = {
    "PASS_TD", "TD", "INT",
    # MLB rare
    "HR", "SB",
    # NBA scarce
    "STL", "BLK", "TOV",
    # NHL
    "GOALS", "ASSISTS", "SHOTS", "PIM",
    # soccer
    "TOTAL_GOALS", "GOALS_FOR", "GOALS_AGAINST",
}

# ── Core functions ───────────────────────────────────────────────────────────

def p_ge_nbinom(mu: float | np.ndarray, sigma: float, line_val: float,
                continuity_correction: float = 0.5) -> float | np.ndarray:
    """P(X >= line_val) under Negative Binomial with mean=mu, std=sigma.

    NB is already discrete — the continuity correction is **not** applied
    (it is only needed when using the normal CDF to approximate a discrete
    stat).  The parameter is kept for a consistent API with ``p_ge_normal``.

    Parameters
    ----------
    mu : float | np.ndarray
        Predicted mean (from regressor). Accepts arrays for batched evaluation.
    sigma : float
        Predicted standard deviation (from residual_std of regressor).
    line_val : float
        Market line (e.g. 5 for SO >= 5).
    continuity_correction : float
        Ignored for NB (discrete).  Default 0.5.

    Returns
    -------
    float | np.ndarray
        P(X >= line_val) clipped to [0.001, 0.999].

    Notes
    -----
    NB parameterisation (Gamma-Poisson mixture):
        E[X] = mu
        Var[X] = mu + mu² / r  →  r = mu² / (sigma² - mu)
        p_success = r / (r + mu)   (scipy's "success" probability per trial)
    """
    if isinstance(line_val, (int, float)) and line_val <= 0:
        return np.where(np.asarray(mu) > 0, 0.999, 0.001) if isinstance(mu, np.ndarray) else max(0.001, min(0.999, 1.0))

    mu_arr = np.atleast_1d(np.asarray(mu, dtype=float))
    # Clip negative mu values to 0 (count stats can't be negative, but
    # LGBM can predict slightly negative for very low-volume players)
    mu_arr = np.maximum(mu_arr, 0)
    var = sigma * sigma

    # Underdispersed or degenerate → Poisson limit
    # Also handle zero-mu case (r = 0/var = 0 causes NaN in NB branch)
    poisson_mask = (var <= mu_arr) | (mu_arr == 0)
    r_vals = np.where(mu_arr > 0, mu_arr * mu_arr / (var - mu_arr), 0.0)
    finite_mask = np.isfinite(r_vals) & (r_vals > 0)
    use_poisson = poisson_mask | ~finite_mask

    result = np.full_like(mu_arr, 0.5)

    # Poisson branch — fully vectorized
    if use_poisson.any():
        mu_pois = mu_arr[use_poisson]
        lv = line_val
        if isinstance(line_val, np.ndarray):
            lv = line_val[use_poisson]
        result[use_poisson] = 1.0 - poisson.cdf(lv - 1, mu_pois)
        result[use_poisson] = np.clip(result[use_poisson], 0.001, 0.999)

    # NB branch — fully vectorized
    nb_mask = ~use_poisson
    if nb_mask.any():
        mu_nb = mu_arr[nb_mask]
        r_nb = r_vals[nb_mask]
        p_success = r_nb / (r_nb + mu_nb)
        if isinstance(line_val, np.ndarray):
            lv = line_val[nb_mask]
        else:
            lv = line_val
        prob = 1.0 - nbinom.cdf(lv - 1, r_nb, p_success)
        result[nb_mask] = np.clip(prob, 0.001, 0.999)

    # Return scalar if input was scalar (not a numpy array)
    if not isinstance(mu, np.ndarray):
        return float(result[0])
    return result


def p_ge_poisson(mu: float | np.ndarray, line_val: float,
                 continuity_correction: float = 0.5) -> float | np.ndarray:
    """P(X >= line_val) under Poisson(mean=mu).

    Poisson is already discrete — the continuity correction is **not** applied.
    The parameter is kept for a consistent API with ``p_ge_normal``.

    Parameters
    ----------
    mu : float | np.ndarray
        Predicted mean(s). Accepts arrays for batched evaluation.
    line_val : float
        Market line.
    continuity_correction : float
        Ignored for Poisson (discrete).  Default 0.5.

    Returns
    -------
    float | np.ndarray
        P(X >= line_val) clipped to [0.001, 0.999].
    """
    if isinstance(line_val, (int, float)) and line_val <= 0:
        return np.where(np.asarray(mu) > 0, 0.999, 0.001) if isinstance(mu, np.ndarray) else max(0.001, min(0.999, 1.0))

    mu_arr = np.atleast_1d(np.asarray(mu, dtype=float))
    # Zero-mean predictions -> P(>= x) = 0 for any x > 0
    zero_mask = mu_arr <= 0
    result = np.full_like(mu_arr, 0.5)
    result[zero_mask] = 0.001

    # Poisson is discrete: P(X >= k) = 1 - P(X <= k-1)
    calc_mask = ~zero_mask
    if calc_mask.any():
        result[calc_mask] = 1.0 - poisson.cdf(line_val - 1, mu_arr[calc_mask])

    result = np.clip(result, 0.001, 0.999)

    if not isinstance(mu, np.ndarray):
        return float(result[0])
    return result


def p_ge_normal(mu: float, sigma: float, line_val: float,
                continuity_correction: float = 0.5) -> float:
    """P(X >= line_val) under Normal(mu, sigma) — original fallback.

    Kept for backward compatibility and for stats whose distribution
    is reasonably approximated by the normal.
    """
    from scipy.stats import norm as _norm
    if sigma < 0.3:
        sigma = 0.3
    z = (line_val - continuity_correction - mu) / sigma
    prob = 1.0 - _norm.cdf(z)
    return max(0.001, min(0.999, float(prob)))


# ── Convenience ──────────────────────────────────────────────────────────────

_STAT_DIST_MAP = {
    s: "nbinom" for s in NB_STATS
}
_STAT_DIST_MAP.update({s: "poisson" for s in POISSON_STATS})


def p_ge_stat(stat_name: str, mu: float, sigma: float, line_val: float,
              continuity_correction: float = 0.5) -> float:
    """Compute P(X >= line_val) using the appropriate distribution for *stat_name*.

    Parameters
    ----------
    stat_name : str
        Upper-case stat name, e.g. ``\"PASS_YDS\"``, ``\"TD\"``, ``\"PTS\"``.
    mu : float
        Predicted mean from the regressor.
    sigma : float
        Residual standard deviation (or fitted std).
    line_val : float
        Market line.
    continuity_correction : float
        Continuity correction. Default 0.5.

    Returns
    -------
    float
        Probability clipped to [0.001, 0.999].
    """
    dist = _STAT_DIST_MAP.get(stat_name.upper(), "normal")
    if dist == "nbinom":
        return p_ge_nbinom(mu, sigma, line_val, continuity_correction)
    elif dist == "poisson":
        return p_ge_poisson(mu, line_val, continuity_correction)
    else:
        return p_ge_normal(mu, sigma, line_val, continuity_correction)
