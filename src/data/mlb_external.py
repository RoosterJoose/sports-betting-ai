"""MLB external data integrations: umpire strike zones and opener detection.

Per NotebookLM research:
- Umpire zone variance: 1-2 SO shift + 0.5-1.0 runs per game between large/small zone umps
- Opener detection: critical for IP model — openers pitch <3 IP, distorting IP props

**Status as of June 9, 2026: STUBS ONLY**
Both data sources require external scraping (UmpireScorecards, FanGraphs pitcher
role data) that has not been integrated yet. These stubs return league-average
defaults so downstream code can be wired today, and flipping the integration on
later is a one-line change.

Usage:
    from src.data.mlb_external import get_umpire_zone_size, detect_opener

    zone_size = get_umpire_zone_size(umpire_id="12345")
    is_opener = detect_opener(pitcher_id="54321", game_date="2026-06-10")
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# League-average defaults
LEAGUE_AVG_ZONE_SIZE = 0.0       # z-score, +/- from mean (0 = league average)
LEAGUE_OPENER_RATE = 0.05        # ~5% of starts use an opener (not yet sourced)


def get_umpire_zone_size(umpire_id: Optional[str] = None,
                          game_date: Optional[str] = None) -> float:
    """Return umpire strike-zone size as a z-score (0 = league average).

    **Stub:** Returns 0.0 (league average) until UmpireScorecards data is
    integrated. Per NotebookLM: 1-2 SO shift and 0.5-1.0 runs difference
    between extreme umpires.

    Integration TODO:
        1. Scrape UmpireScorecards.com weekly (CSV of umpire called-strike rates)
        2. Cache in data/cache/mlb/umpires/{umpire_id}.json
        3. Replace return below with cached value
    """
    return LEAGUE_AVG_ZONE_SIZE


def get_umpire_called_strike_rate(umpire_id: Optional[str] = None) -> float:
    """Return umpire's career called-strike rate (MLB average ~0.18).

    **Stub:** Returns 0.18 (league average) until data is integrated.
    """
    return 0.18


def detect_opener(pitcher_id: Optional[str] = None,
                  game_date: Optional[str] = None,
                  pitcher_history: Optional[pd.DataFrame] = None) -> bool:
    """Detect if a starting pitcher is actually an opener (<3 IP expected).

    **Stub:** Returns False (treats every starter as a traditional starter).
    Heuristic: a pitcher is an opener if their last 5 starts averaged <3 IP
    and they entered the game with a low pitch count in their prior outing.

    Integration TODO:
        1. Use FanGraphs pitcher role data (Starter vs Opener vs Bulk Reliever)
        2. Cross-reference pitch count from prior game
        3. Replace return below with detection logic
    """
    return False


def get_bullpen_quality(team_id: Optional[str] = None,
                         game_date: Optional[str] = None) -> float:
    """Return team bullpen quality as a z-score (0 = league average).

    **Stub:** Returns 0.0. Affects IP and ER predictions (good bullpen =
    starter can be pulled earlier without ER cost; bad bullpen = starter
    pushed deeper or ER inflation risk).
    """
    return 0.0
