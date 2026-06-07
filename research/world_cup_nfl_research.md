# World Cup 2026 — Betting Model Research Questions for NotebookLM

**Timeline:** World Cup starts June 11, 2026 (7 days away)
**Platforms:** Kalshi (match winner 1X2, total goals, player props), PrizePicks (player props)
**Goal:** Build profitable prediction models for World Cup betting markets

---

## Q1: Data Sources for World Cup 2026

"What free/open-source Python-accessible data sources provide historical World Cup and international soccer data for building prediction models? Specifically:

(a) **Match-level data:** What API or dataset provides historical international match results with team-level stats (possession, shots, goals, xG, fouls, cards) for the last 10+ years? Does `soccerdata` (pypi) work? What about the `worldfootball-R` / `understat` / `StatsBomb` free datasets?

(b) **Player-level data:** What source provides per-player statistics (goals, assists, shots, minutes played, key passes) for World Cup players and recent international matches? Is there a free API for this or must it be scraped?

(c) **Squad/roster data:** Where to get confirmed World Cup 2026 rosters, including expected lineups and formation information?

(d) **Elo/fifa rankings:** What library provides historical FIFA World Rankings or Elo ratings for national teams? Can these be computed from match results?

(e) **Kalshi market data:** How to access Kalshi World Cup markets programmatically? What ticker prefixes will Kalshi use for World Cup 2026 markets (e.g., KXWC)?

(f) **PrizePicks World Cup:** Does PrizePicks offer World Cup player props? What stat types are typically offered (goals, assists, shots, fouls)? What league ID does PrizePicks use for World Cup / soccer?"

## Q2: World Cup Match Prediction Model Architecture

"Document the most successful published architectures for international soccer / World Cup match prediction models:

(a) **Model type:** What model type achieves the highest accuracy for international soccer 1X2 (home/draw/away) prediction? Elo-based? Poisson regression? XGBoost? Neural network? What accuracy thresholds have been achieved?

(b) **Features:** What features drive international soccer match outcomes? Specifically quantify the predictive value of: FIFA ranking vs Elo rating, recent form (last 5 matches), goal differential, xG differential, shots on target differential, possession, home/neutral advantage. What is the minimum match history needed?

(c) **Tie/draw probability:** How should the draw outcome be modeled? Is it derived from the predicted goal distribution (Poisson/difference) or predicted directly as a 3rd class?

(d) **Player impact:** For World Cup specifically, how important are individual star players vs team-level metrics? Is there a published approach for incorporating player availability (injuries, suspensions)?

(e) **Kalshi markets specifically:** For Kalshi binary markets (e.g., 'Team A wins'), does the standard Poisson approach work, or should the model output be calibrated differently for exchange trading versus sportsbook betting?"

## Q3: World Cup Player Prop Models

"Document approaches for World Cup player prop betting (goals, assists, shots):

(a) **Goal scorer models:** What approach is used to predict which players will score in a specific match? Poisson regression based on historical goal rates? Minutes-weighted xG models? How many matches of data per player are needed?

(b) **PrizePicks stat types:** What stat types does PrizePicks offer for World Cup? Are there any pattern-based strategies (e.g., betting on star players against weak opponents, or fading overhyped players)?

(c) **Data requirements:** What's the minimum viable dataset for a World Cup player prop model? Nation team data only, or can domestic league performance be included?

(d) **Kalshi player props:** Does Kalshi offer individual player markets for the World Cup (goals, assists)? What ticker conventions would be used?"

## Q4: PrizePicks & Kalshi Market Calendar (June-August 2026)

"Document the complete sports calendar for June-August 2026 with focus on PrizePicks and Kalshi:

(a) **World Cup 2026** — exact dates (June 11 - July 19?), match schedule, group stage vs knockout format. What's total match count?

(b) **MLB** — regular season continues through October. All-star break dates?

(c) **NFL preseason** — starts August? When do player props become available on PrizePicks?

(d) **Summer sports:** Are there any other major sports events (Wimbledon, NBA offseason, etc.) with significant PrizePicks/Kalshi activity?

(e) **PrizePicks league IDs** — what are the PrizePicks league IDs for World Cup, Premier League, and other football/soccer competitions?"

## Q5: NFL 2026 Season — Model Preparation

"Document the approach for building NFL models across all bet types:

(a) **nfl_data_py:** What exact functions return weekly player data, team stats, and game outcomes for the 2021-2025 seasons? Are there play-by-play data for advanced features?

(b) **Kalshi NFL markets 2025 season:** What NFL market types does Kalshi typically offer? (team winner, spreads, totals? player props?) Is there seasonality to liquidity?

(c) **PrizePicks NFL:** What stat types does PrizePicks typically offer for NFL? (passing yards, rushing yards, receiving yards, receptions, touchdowns, interceptions, etc.) What league IDs?

(d) **Model requirements:** For each NFL bet type (player props, team totals, game winner), what specific features and historical data are needed?"
