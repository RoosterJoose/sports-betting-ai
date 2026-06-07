# F5 (First 5 Innings) MLB Betting Model — Research Questions for NotebookLM

## Question 1: F5 Model Architecture Comparison

"Document all published F5 (First 5 Innings) MLB betting model architectures from open-source projects and academic literature. Specifically compare three approaches:

**A) 3-way multiclass classifier** (predicts Away win / Home win / Tie directly using aggregated pitcher+team features). What accuracy thresholds have been achieved? What model types (XGBoost, LightGBM, neural net)?

**B) Monte Carlo simulation** (simulate every plate appearance thousands of times using discrete PA outcomes — single/double/triple/HR/walk/HBP/strikeout/out — advance game state dynamically through 5 innings). This is described as the 'most successful documented MLB architecture' — is this true? What specific accuracy or ROI has been published?

**C) Run distribution approach** (model each team's run scoring distribution separately, then compare to determine win/tie probability).

For each approach, provide:
- The exact published accuracy, ROI, or Brier score
- Number of training samples used
- Feature sets employed
- Whether results held out-of-sample (i.e., on seasons not used in training)
- Any GitHub repos or published papers with verified results

Also: does the 3-way multiclass approach have any known systematic weaknesses vs the simulation approach? Specifically, does treating the 3 outcomes (Away/Home/Tie) as independent classes vs letting them emerge from a generative process lead to different edge estimates?"

## Question 2: Kalshi KXMLBF5 Market Microstructure

"Document the specific market microstructure of Kalshi's KXMLBF5 (First 5 Innings) binary markets. Required details:

**A) Liquidity profile:**
- Typical bid-ask spreads on KXMLBF5 markets (in cents)
- Average daily volume per market
- Time to fill for maker orders placed 1¢ above the best bid

**B) Optimal pricing range:**
Published research says sportsbook F5 betting has a mathematical 'sweet spot' at 40-67¢ (decimal odds 1.50-2.50). Does this same range apply to Kalshi exchange markets, given:
- Kalshi charges fees of 0-7¢ per contract depending on VIP tier
- Kalshi markets settle at $1.00 (binary)
- Maker orders receive rebates, taker orders pay fees

What price range maximizes risk-adjusted returns specifically on Kalshi given its fee structure?

**C) F5 market efficiency:**
- Are Kalshi F5 markets efficiently priced, or do they exhibit the same favorite-longshot bias documented in Kalshi's broader markets?
- What is the documented edge threshold for Kalshi sports markets specifically (not sportsbook)? The sportsbook research says 5-6% edge minimum — does Kalshi's exchange need a different threshold given wider spreads?

**D) Liquidity constraints:**
- At $18 bankroll, are 1-contract positions (minimum $0.40-$0.67 per contract) viable on Kalshi F5 markets?
- What is the minimum practical bankroll for a multi-bet F5 strategy on Kalshi?
- Do F5 markets have enough volume to absorb quarter-Kelly sized positions?

**E) Order execution:**
- What percentage of maker orders (placed 1-2¢ above best bid) actually get filled before game start?
- Average time-to-fill for maker orders?
- Do games with no volume (0 volume_24h) ever have orders filled?"

## Question 3: F5 Feature Set Validation and Minimum Data Requirements

"Document the exact feature sets and minimum data requirements for building a profitable F5 betting model based on published work.

**A) Pitcher features:**
Research says 'xFIP and SIERA are essential' and 'raw ERA is useless.' Is this true? Specifically:
- Do published F5 models use xFIP/SIERA (derived from FanGraphs) or raw K%/BB%/HR9 (directly computed from box scores)?
- If xFIP/SIERA are essential, what library or API provides these for historical seasons? (pybaseball? FanGraphs API? manual computation?)
- Is there a published study quantifying how much accuracy improves with advanced metrics vs simple rolling averages of K/BB/HR/IP?

**B) Lineup features:**
Research says 'opposing lineup's wRC+ is essential' and 'handedness platoon splits provide documented edge.' Specifically:
- How should lineup quality be modeled for F5 only (starters typically face the lineup 1.5-2 times through)?
- Does each individual batter's quality matter, or is a team-level aggregate (wRC+ vs handedness) sufficient?
- What source provides daily lineup data for historical backtesting?

**C) Environmental features:**
Research says 'weather (wind, temperature, humidity) is vital for fly-ball starters.' Specifically:
- Which weather features matter for F5 specifically (vs full game)?
- What data source provides historical game-time weather at the ballpark?
- Has any published study quantified the edge gained from including weather vs ignoring it?
- Is umpire strike zone data (from Statcast) actually used in any published F5 model, or is it theoretical?

**D) Minimum data requirements:**
- How many seasons of data are needed to train a valid F5 model? (We have 2024-2026 = 2.5 seasons, ~5,700 games)
- Is 2.5 seasons sufficient, or does research suggest 5+ seasons minimum for 3-way outcomes?
- How many games (test set) are needed to validate that an F5 betting edge is statistically significant? (Research says 1,000 minimum — is this confirmed?)
- Do F5 bettors typically train on all available seasons or use a fixed rolling window (e.g., last 3 seasons)?
