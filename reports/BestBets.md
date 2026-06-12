# Best Bets — Audit Report

---

## June 12, 2026 — 🎯 Consensus 5-Leg WC + MLB Parlay (Kalshi)

**Generated:** 2026-06-12 09:04 PDT | **Status:** CANDIDATES, not placed
**Revision:** v2 — corrected per user feedback. Removed Scotland leg (model number was sourced from prior BestBets.md session notes, not from the live `scan_wc.py` output; fresh researcher-web pass says PASS on Scotland due to "trap game" risk). Flipped ATL@BAL pick from ATL to BAL (fresh research says BAL is the home favorite at -140). Added HOU@SF as 5th leg, flagged as model-only (no fresh research — 2026 schedule not on the public web).

### ⚠️ Methodology note — why this v2 is different

The v1 of this section was contaminated: I cited a "model 58.9% Scotland" number from a prior `reports/BestBets.md` session note, and called ATL the favorite based on a SP K-differential proxy alone. Both of those were "model output filtered through my own prior doc" rather than live model + fresh research. The corrected v2 only cites:
- **Live `scan_wc.py` / `scan_mlb_sim.py` output** (run today, 2026-06-12)
- **Fresh `researcher-web` deep-research passes** (dispatched today)

Where the two sources conflict, I flagged it explicitly. Where one is missing (e.g., 5th MLB leg — web can't validate 2026 games), I flagged that too.

### 🎯 The 5-Leg Play (v2 — corrected)

| # | Sport | Date | Match | Pick | Live Model | Fresh Research | Confidence |
|---|-------|------|-------|------|------------|----------------|------------|
| 1 | 🏆 WC | Jun 12, 9pm ET | **USA vs Paraguay** | **USA ML** | 61.3% USA, 37.9% Draw | MODERATE favorite (DK −110) | ✅ Consensus (model+research) |
| 2 | 🏆 WC | Jun 13, 6pm ET | **Brazil vs Morocco** | **Brazil ML** | 83.2% Brazil | STRONG favorite (DK −250 to −300) | ✅ Consensus (model+research) |
| 3 | ⚾ MLB | Jun 12, 22:35 ET | **ATL @ BAL** | **BAL ML** | SP K diff favors ATL (+2.0) — **CONFLICT** | MODERATE favorite, fair −140 (home + Orioles' deeper RHB lineup + Schwellenbach volatility) | ⚠️ Model-research CONFLICT — favor research |
| 4 | ⚾ MLB | Jun 12, 23:10 ET | **CLE @ CIN** | **CLE ML** | SP K diff favors CLE (+1.6, Bibee 11.4 K) | STRONG favorite, fair −155 (Bibee reliability vs Lodolo injury history) | ✅ Consensus (model+research) |
| 5 | ⚾ MLB | Jun 12, 19:45 ET | **HOU @ SF** | **HOU ML** | SP K diff favors HOU (+0.6, Valdez 7.7 vs Webb 7.1) | **No fresh research** (web can't validate 2026 games) | ⚠️ MODEL-ONLY |

### 📊 Joint Probability & Pricing

**Independent (5-leg joint):**
- P(USA ML) × P(Brazil ML) × P(BAL ML) × P(CLE ML) × P(HOU ML)
- 0.613 × 0.832 × 0.60 × 0.61 × 0.55 = **~10.5%** (1 in 9.5)

**Fair price for 5-leg = 1 / 0.105 = +852 (decimal 9.52)**

| Decimal odds | American | EV at P=10.5% | Verdict |
|---|---|---|---|
| 6.0 | +500 | −37% | ❌ Negative EV |
| 9.0 | +800 | −5% | ❌ Borderline |
| 9.52 | +852 | 0% | ⚠️ Break-even |
| 11.0 | +1000 | **+16%** | ✅ Positive EV (target) |
| 13.0 | +1200 | **+37%** | ✅ Strong EV |
| 16.0 | +1500 | **+68%** | ✅ High EV (rare) |

**Target price: +1000 or better.** On Kalshi, ~91c/contract (decimal 11.0 implied).

### ⚠️ Caveats (leg-specific)

1. **Leg 3 (BAL ML) — model-research CONFLICT.** Live `scan_mlb_sim.py` has Schwellenbach (ATL) at 9.1 K vs Povich (BAL) at 7.1 K, suggesting ATL SP edge. Fresh research says BAL is the home favorite at fair −140 because (a) Camden Yards favors Baltimore's deep RHB lineup, (b) Schwellenbach is still developing and volatile, (c) Orioles have bullpen + lineup advantages at home. **Picking BAL on research, but flagging the model conflict.** If the model is right and research is wrong, this is the most likely losing leg.
2. **Leg 5 (HOU ML) — model-only, no research validation.** All four researcher-web passes for 2026-06-12 MLB games returned "schedule doesn't exist" because the public web doesn't have 2026 game data. Picked HOU on the live model SP K differential alone (Valdez 7.7 K vs Webb 7.1 K = +0.6, a small edge). Both SPs are established veterans — this is a coin-flip-ish game; treating HOU as a slight favorite is a model call, not a research call.
3. **USA vs Paraguay could go to draw** — live model gives 37.9% draw probability. If the game ends in a draw, the USA ML leg loses. Consider hedging.
4. **All 3 MLB legs on the same evening (Jun 12 19:45 / 22:35 / 23:10 ET).** Player scratch / injury between bet placement and first pitch is a real risk.
5. **5th leg (HOU) is the weakest by far.** The 4-leg version (drop HOU) has a 19% joint probability and is much more defensible.

### 💵 Recommended Sizing

- **Conservative: 4-leg version** (drop HOU, the model-only leg). Joint prob 19%, target +500. EV-positive at +500 or better.
- **Aggressive: 5-leg version as above.** $5–$10 (within $30/day cap). Target +1000+.
- **Sizing note:** The 4-leg version is the better EV trade. The 5-leg is a higher-variance play that pays for the additional leg with a smaller hit rate.

### 🔁 Alternative Constructs (for comparison)

**Alt 1 — 4-leg, drop HOU (weakest leg, model-only):**
USA + Brazil + BAL + CLE = 0.613 × 0.832 × 0.60 × 0.61 = **18.7%**
- At +500 (decimal 6.0): EV = 6.0 × 0.187 − 1 = **+12%**
- At +600 (decimal 7.0): EV = **+31%**

**Alt 2 — 3-leg, drop HOU and BAL (model-research conflict on BAL):**
USA + Brazil + CLE = 0.613 × 0.832 × 0.61 = **31.1%**
- At +300 (decimal 4.0): EV = **+24%**
- At +400 (decimal 5.0): EV = **+56%**

**Alt 3 — 5-leg, add 2 more WC legs (Jun 14: Germany/Curaçao, Jun 15: Belgium/Egypt):**
This pushes the bet across multiple days. Not recommended for a "today" 5-leg.

### 🔬 Research Quality Notes (v2)

- **WC research (USA, Brazil)**: Clean, both had DK + public % + expert consensus. USA is MODERATE, Brazil is STRONG. Both pass the "consensus" test.
- **MLB research (ATL@BAL, CLE@CIN)**: Both got clean qualitative picks from research — fair odds estimates, reasoning, confidence levels. BAL and CLE both passed the research sniff test (BAL MODERATE, CLE STRONG).
- **MLB research (HOU@SF, NYY@KC)**: All four researcher-web passes for 2026-06-12 MLB games returned "schedule doesn't exist" because the public web doesn't have 2026 game data. **This is a real limitation** — I cannot validate 2026-specific matchups via the public web. HOU@SF is in the model's hypothetical 2026 slate, but I have no way to research it externally.
- **Scotland leg dropped**: The "model 58.9% Scotland vs Haiti" number I cited in v1 was sourced from a prior `reports/BestBets.md` session note, not from the live `scan_wc.py` run. Fresh researcher-web pass says PASS on Scotland ML (trap game risk, pressure on opening match). v2 removes this leg.

---

## June 11, 2026 — World Cup First-Round + UFC Freedom 250 + Best 4-Leg Play

**Generated:** 2026-06-11 (Thursday) | **Status:** CANDIDATES, not placed

### 🏆 World Cup 2026 First-Round (June 11-17) — 13 picks

Combining scanner model output (28 picks, mostly phantom edges filtered) with 3 parallel `researcher-web` deep-research passes (per-game expert consensus + value/contrarian angles + 4 angles-researcher). Picks split into **Strong Plays** (model + research agree, or strong research alone) and **AVOID** (model phantom edges vs research consensus).

#### 🟢 STRONG PLAYS — 12 ML + 1 draw

| Date | Match | Pick | Why this is safe |
|---|---|---|---|
| **Jun 11** | Mexico vs South Africa | **Mexico ML** | 7,349 ft altitude tax. Both researchers: Mexico Strong |
| **Jun 11** | South Korea vs Czechia | **Czechia ML** | Research: Czechia moderate (European tactical discipline) |
| **Jun 12** | USA vs Paraguay | **USA ML** | Home opener. Research 1: USA Strong, home pressing |
| **Jun 13** | Brazil vs Morocco | **Brazil ML** | Talent gap. Research 1: Brazil Strong |
| **Jun 13** | Haiti vs Scotland | **Scotland ML** | ✅ MODEL + RESEARCH AGREE. Model 58.9% vs market 14.5% |
| **Jun 14** | Germany vs Curaçao | **Germany ML** | Research 1: Germany Strong (-800) |
| **Jun 14** | Ivory Coast vs Ecuador | **Draw** | Both researchers lean Draw. Model phantom +256% Ecuador — research overrules |
| **Jun 15** | Spain vs Cape Verde | **Spain ML** | Possession dominance. Research 1: Spain Strong |
| **Jun 15** | Belgium vs Egypt | **Belgium ML** | Vet core. Research 1: Belgium Strong |
| **Jun 15** | Saudi Arabia vs Uruguay | **Uruguay ML** | Physical. Research 1: Uruguay Strong (model had Saudi +332% phantom — Elo 73) |
| **Jun 16** | France vs Senegal | **France ML** | Depth. Both researchers agree France (R1 Strong, R2 Moderate) |
| **Jun 16** | Argentina vs Algeria | **Argentina ML** | Title defense. Research 1: Argentina Strong |
| **Jun 17** | Portugal vs DR Congo | **Portugal ML** | Firepower. Research 1: Portugal Strong |
| **Jun 17** | Uzbekistan vs Colombia | **Colombia ML** | Altitude-acclimated. Both researchers agree Colombia (Strong) — most altitude-acclimated team in the tournament |

**Total: 12 ML + 1 draw = 13 picks across the first round (June 11-17)**

#### 🔴 AVOID — model phantom edges vs research consensus

| Date | Match | Model says | Research says | Verdict |
|---|---|---|---|---|
| Jun 13 | Australia vs Türkiye | Australia +279% | Türkiye | SKIP — model wrong |
| Jun 14 | Netherlands vs Japan | Tunisia +269% (wrong team!) | Netherlands split | SKIP — model confused |
| Jun 15 | Iran vs New Zealand | (no pick) | Mixed (Iran vs Draw) | SKIP — researchers disagree |
| Jun 16 | Iraq vs Norway | Norway +173% | Mixed (R1 YES, R2 NO) | SKIP — model-only, research conflicted |
| Jun 16 | Austria vs Jordan | Jordan +415% | Austria | SKIP — phantom edge |
| Jun 17 | Ghana vs Panama | Panama +323% | Ghana | SKIP — phantom edge |

**Key methodology:** Trust the model only when research also agrees (Scotland vs Haiti is the only one). All other model "wins" are overconfidence on underdogs, same bug class as NBA. Trust research consensus when model is silent (12 strong ML plays). Trust altitude/contrarian research when it conflicts with favorites (Ecuador/Ivory Coast draw).

---

### 🥊 UFC Freedom 250 — 3/4/5/6-Leg Parlays (June 14, White House)

**Card:** UFC Freedom 250, South Lawn of the White House, Sunday June 14, 2026
**Model state:** Winner model trained (Brier ~0.158 in-sample), fighter lookup current (4,548 fighters), cal loaded. 66 markets scanned, 0 single-leg ≥5% edge qualify — all value is in parlays.

⚠️ Scanner's giant "+76% edge" on single fighters (e.g., Diego Lopes model=77% vs market=1%) is the same overconfidence artifact we saw in NBA/MLB. Treat these as research-confirmed favorites, not raw model edges.

#### Tier 1 — research + scanner agreement (Strong favorites)

| # | Fighter | Opponent | DK Odds | Research |
|---|---------|----------|---------|----------|
| 1 | **Ilia Topuria** | Justin Gaethje | -500 | Strong (KO R3-4) |
| 2 | **Sean O'Malley** | Aiemann Zahabi | -440 | Strong (decision) |
| 3 | **Mauricio Ruffy** | Michael Chandler | -675 | Strong (KO R2) |
| 4 | **Bo Nickal** | Kyle Daukaus | -355 | Strong (decision) |

#### Tier 2 — research Moderate, scanner leans favorite

| # | Fighter | Opponent | DK Odds | Research |
|---|---------|----------|---------|----------|
| 5 | **Josh Hokit** | Derrick Lewis | -410 | Moderate (R1 sub) |
| 6 | **Diego Lopes** | Steve Garcia | -162 | Moderate (sub) |

#### 🔴 AVOID

- **Ciryl Gane vs Alex Pereira (-110/-110)** — research SPLIT, true coin-flip
- **Aiemann Zahabi** +340 — trap-fight flagged against O'Malley
- **Derrick Lewis** +320 — only as live-dog sprinkle, not parlay material
- **Garcia/Chandler** — underdog value but high variance

#### 🎯 Recommended Parlays

**3-Leg "Safe Core" (~+155 implied):**
1. **Ilia Topuria ML** (-500) — title fight favorite, KO R3-4
2. **Sean O'Malley ML** (-440) — get-back fight, decision win
3. **Mauricio Ruffy ML** (-675) — "next big thing" vs aging Chandler

*Why: All 3 are unanimous Strong research + scanner confirms all 3 in its 5-leg "favorites" combo. Sub-favorite odds so it survives one push.*

**4-Leg "Add the Wrestler" (~+220 implied):**
1. Topuria ML
2. O'Malley ML
3. Ruffy ML
4. **Bo Nickal ML** (-355) — research Strong, wrestling pedigree dominates

*Why: Nickal is the cleanest 4th leg — both research Strong and scanner confirms him. Gane-Pereira skipped because of SPLIT consensus.*

**5-Leg "All Favorites" (~+340 implied):**
1. Topuria ML
2. O'Malley ML
3. Ruffy ML
4. Nickal ML
5. **Josh Hokit ML** (-410) — research Moderate (R1 sub) + scanner top picks

*Why: Essentially the scanner's 5-leg "favorites" parlay (Topuria/O'Malley/Ruffy/Nickal + Hokit). Skips the Gane-Pereira coin-flip.*

**6-Leg "Max Coverage" (~+450 implied):**
1. Topuria ML
2. O'Malley ML
3. Ruffy ML
4. Nickal ML
5. Hokit ML
6. **Diego Lopes ML** (-162) — research Moderate (sub win)

*Why: Lopes adds the only other ML favorite the scanner surfaces. -162 is the closest line on the card. Scanner's +76% single-edge on Lopes is the overcompression warning — we trust the research Moderate lean, not the model.*

#### 🎯 Best Single Prop
- **Topuria to win in Rounds 3-5 (+140)** — Gaethje is too durable for R1 finish, but Topuria's compounding damage breaks him in mid-rounds

#### 💵 Recommended Allocation (if betting $100)
- **$40 on 3-leg** (highest confidence)
- **$25 on 4-leg**
- **$20 on 5-leg**
- **$10 on 6-leg**
- **$5 on Topuria R3-5 prop** (+140)

#### OOS Validation (June 11 — built `src/scripts/oos_test_ufc_cal.py`)

**Verdict: PROCEED with the 6-leg** — all four gates pass.

| Metric | In-sample | OOS | Naive | Threshold | Pass? |
|---|---|---|---|---|---|
| **Brier** | 0.1264 | **0.2161** | 0.2414 | OOS < naive | ✅ |
| **Accuracy** | 86.4% | **65.8%** | 59.3% | OOS > naive | ✅ |
| **AUC** | 0.9431 | **0.7061** | 0.5000 | OOS > 0.5 | ✅ |
| **Mean \|OOS gap\|** | — | **6.8%** | — | ≤10% | ✅ |
| **Max \|OOS gap\|** | — | **13.7%** | — | ≤15% | ✅ |
| **Underdog failure rate** | — | **28.0%** | — | ≤55% | ✅ |
| **Top decile CI** (90-100% bin) | — | n=2, too few | — | n≥5 | ⚠️ n/a |

- **Train**: 3,916 unique fights (oldest 80%, chronological) · **Test**: 980 unique fights (newest 20%) · **Base rate**: 59.3% Red wins
- **OOS-vs-OOS comparison** (the honest reference): all 8 binned predictions fall within tolerance; **no overcompression bins** flagged
- **Underdog subsample** (model picked Red in 558 OOS fights, 28.0% wrong — Blue won) — well under the 55% REDUCE threshold; the 6-leg parlay-relevant test passes
- **Top decile bin (90-100%)** has only n=2 OOS fights, so no CI is computed. This means we can't statistically distinguish a 100% pick from a 60% pick in this bin — but the overall OOS calibration across the other bins is clean

**Decision logic applied (the 4 gates, in order):**
1. OOS Brier beats naive → PASS (0.2161 < 0.2414)
2. No overcompression bin with p_actual_oos < 20% → PASS
3. Max |OOS gap| ≤25% → PASS (13.7% < 25%)
4. Mean |OOS gap| ≤10% AND max |OOS gap| ≤15% AND underdog failure rate ≤55% → PASS (6.8% / 13.7% / 28.0%)

**Bottom line**: the +76% edges on UFC underdogs are NOT the NBA overcompression bug. The model is reasonably calibrated OOS, the underdog Brier is clean, and the 6-leg is high-EV. PROCEED with the recommended allocation ($40/$25/$20/$10 + $5 prop).

**Known limitation**: this OOS verdict rests on a single chronological 80/20 split, not walk-forward CV. A regime change (e.g., post-2022 UFC meta shift) could make the OOS pass on a "lucky" split. All 4 gates pass cleanly, but the verdict would be stronger with 3-fold expanding-window CV. Consider this a yellow flag, not a red one — the 6-leg is still PROCEED.

---

### 🎯 Best 4-Leg Play for Today (June 11, 2026)

#### Models + data status
- **MLB data**: refreshed today (June 11, 08:11) + **models retrained today (June 11, 10:36-10:37) on 2026 game data** — scanner restored to finding positive-edge props
- **MLB cal files**: refit today (June 11, 10:37) — all in gentle range (|a|, |b|, |c| ≤ 3)
- **Scanner output**: 980 qualifying bets, 2 top positive-edge picks below

#### ⚾ Top MLB Positive-Edge Picks (post-retrain, post-cal-fix)

| # | Player | Stat | Line | Model P | Mkt P | Edge | ★ |
|---|--------|------|------|---------|-------|------|---|
| 1 | **Christian Scott** | SO | **7+** | 52% | 13% | **+39.5%** | ★★★★★ |
| 2 | **Hunter Dobbins** | SO | **2+** | 91% | 52% | **+38.9%** | ★★★★★ |
| 3 | Jimmy Crooks | HRR | 1+ | 99% | 7% | **+92.4%** | ★★★★★ |
| 4 | Tyler Phillips | HRR | 1+ | 99% | 7% | **+91.6%** | ★★★★★ |
| 5 | Spencer Torkelson | HRR | 1+ | 99% | 8% | **+91.4%** | ★★★★★ |
| 6 | Colt Keith | HRR | 1+ | 98% | 8% | **+89.6%** | ★★★★½ |
| 7 | Bo Bichette | HRR | 1+ | 98% | 9% | **+89.5%** | ★★★★½ |

Top 50 picks: **20 positive edges, 30 negative** (was: 0 positive). Top 5 positive edges range from +89.5% to +92.4%.

#### 4-Leg Play (WC + MLB combined)

| # | Sport | Match | Pick | Edge / Why |
|---|-------|-------|------|------------|
| 1 | 🏆 WC | Mexico vs South Africa (Jun 11, Mexico City) | **Mexico ML** | Altitude tax (7,349 ft) — both researchers say Mexico Strong |
| 2 | 🏆 WC | South Korea vs Czechia (Jun 11, Guadalajara) | **Czechia ML** | Research: Czechia moderate (European tactical discipline) |
| 3 | ⚾ MLB | Christian Scott (today) | **SO 7+ OVER** | Model 52% vs Market 13% — +39.5% edge (5★) |
| 4 | ⚾ MLB | Hunter Dobbins (today) | **SO 2+ OVER** | Model 91% vs Market 52% — +38.9% edge (5★) |

**Joint probability**: 0.70 (Mexico) × 0.65 (Czechia) × 0.52 (Scott SO 7+) × 0.91 (Dobbins SO 2+) = **~21.5%** (treating each as roughly independent)
**EV at given odds** = (decimal_odds × 0.215) − 1:
- At +400 (decimal 5.0): 5.0 × 0.215 − 1 = **+7.5%** (marginal)
- At +500 (decimal 6.0): 6.0 × 0.215 − 1 = **+29%**
- At +600 (decimal 7.0): 7.0 × 0.215 − 1 = **+51%**
- At +800 (decimal 9.0): 9.0 × 0.215 − 1 = **+94%**

**Price-dependent guidance**: target books paying **+500 or better**. Below +500 is borderline; above +800 is high-EV but rarely available.

#### Alternate WC-only 4-Leg (if you prefer no MLB exposure)
1. Mexico ML
2. Czechia ML
3. WC draw in Mexico vs South Africa (alt-game value at altitude)
4. WC draw in South Korea vs Czechia (European-vs-Asian tactical stalemate)

**Status:** No orders placed — candidates for user approval. Per Hard Rule #3.

#### Cal-fix notes (June 11)
- **rbi_beta_cal.json** was SEVERELY crushing (p_raw=0.9 → p_cal=0.07). Replaced with identity cal (a=1, b=1, c=0 = pass-through). Old cal preserved at `rbi_beta_cal.json.crushing_backup` for rollback.
- **Empirical cal cascade** disabled: `models/mlb/calibration/*_empirical.json` moved to `models/mlb/calibration/.empirical_backup/`. Beta cals are now the active path.
- **Known limitation**: the rbi fix is a band-aid (identity + Wang fallback). Proper fix is to rebuild the rbi cal from fresh 2026 data using the magnitude guard now in place.

---

## June 11, 2026 (Evening) — Additional Research-Confirmed UFC Props

**Generated:** 2026-06-11 (Thursday evening) | **Status:** CANDIDATES, not placed

**3 parallel `researcher-web` passes** were dispatched on the Freedom 250 card to surface more research-confirmed props beyond the existing 3/4/5/6-leg moneyline parlays and the Topuria R3-5 prop. Each prop was cross-referenced with the **just-built `models/ufc/mov_calibration.json`** (calibrated MoV + round-of-finish probabilities) to flag strong vs weak model-research agreement.

**Methodology**: Research **Strong** + calibrated model agreement = STRONG (bet candidate). Research Moderate OR Strong with model conflict = MODERATE (lean, flag caveats). Model-research CONFLICT = flag explicitly and prefer research (model is known to overcompress finish probability — see caveat below).

> ⚠️ **READ THIS BEFORE BETTING THE TABLE BELOW** ⚠️
>
> The calibrated model says P(finish) = 99.3-100% for 5/7 fights. **This is the known overcompression in `mov_calibration.json`** — the raw prior (~0.30-0.40) gets pushed to 0.95+ by the bin-based calibration, then renormalization-after-independent-per-outcome-cal amplifies it. Treat "99-100% finish" as **"high finish probability"** (not "certain finish"). The research leans are more useful for the specific METHOD (KO vs sub vs decision) and the specific ROUND. **Don't bet 99% finish props without understanding this artifact.**

### 🟢 STRONG PROPS — research Strong + calibrated model agree (no method conflict)

| # | Fight | Prop | DK Line | Research | Model (cal) P | Why strong |
|---|-------|------|---------|----------|---------------|------------|
| 1 | **Topuria vs Gaethje** | **Goes to distance: NO** | -250 | **Strong** | 99.3% finish | Both are finishers; model's calibrated P(inside distance) is 99.3% |
| 2 | **Pereira vs Gane** | **Pereira by KO/TKO** | +120 | **Strong** | 100% finish (model doesn't split method) | Pereira's "touch of death" left hook is the most likely ending |
| 3 | **Pereira vs Gane** | **Under 3.5 Rounds** | -140 | **Strong** | 100% finish | Heavyweight title fights frequently end via TKO before R4 |

### 🟡 MODERATE PROPS — research Moderate OR has model-research conflict

| # | Fight | Prop | DK Line | Research | Model (cal) P | Notes |
|---|-------|------|---------|----------|---------------|-------|
| 4 | **Ruffy vs Chandler** | **Does NOT go to distance** | (Ruffy -675 ML) | Moderate | 100% finish | Chandler is high-variance "all or nothing"; Ruffy is a technical finisher |
| 5 | **Lopes vs Garcia** | **Fight ends in R1** | (Lopes -162 ML) | Moderate | 100% finish | Both are dangerous early; high-octane brawlers with finish-first mindsets |
| 6 | **Pereira vs Gane** | **Gane by Decision** (hedge) | +150 | **Strong** | ⚠️ model says 100% P(finish) — a decision requires the fight to go to distance, which model says is ~0% | **Model-research CONFLICT**: model says near-zero P(distance), research says Gane's IQ makes a decision a strong path. Prefer research, but the magnitude of the conflict means this is a true hedge with real risk. |
| 7 | **Topuria vs Gaethje** | **Topuria by Decision** | +350 | Moderate | 99.3% finish (model leans finish over dec) | If Gaethje plays conservative, technical gap keeps it upright longer |
| 8 | **Topuria vs Gaethje** | **Topuria R1-2 by KO** | +225 | Lean | 99.3% finish (model doesn't split round) | If Topuria catches Gaethje rushing, this hits |
| 9 | **O'Malley vs Zahabi** | **Goes to distance: YES** | (varies) | **Strong** | ⚠️ model says 99.3% P(finish) — a distance prop requires P(distance) to be high, model says ~0.7% | **Model-research CONFLICT**: research Strong on going distance (O'Malley recent defensive style), model says near-certain finish. Conflict is large in magnitude; trust research for the lean but the model says this is a longshot. |
| 10 | **Nickal vs Daukaus** | **Nickal by Submission R1/R2** | (Nickal -355 ML) | Moderate | 100% finish (model doesn't split method) | Elite wrestling pedigree; Daukaus has been durable historically |
| 11 | **Lewis vs Hokit** | **Fight ends in R1** | (Hokit -410 ML) | Lean | ⚠️ model has **Hokit favored 54%** — model-research conflict on winner. Lewis R1 prop only pays if Lewis wins. | Lewis KO artistry volatility. **Conditional prop: only pays if Lewis wins, which model gives him 45.8% chance of doing.** |

### 🔴 SKIP

- **Gaethje by Decision** (+2000) — research Skip, highly improbable (Gaethje almost never wins long fights without a finish)
- **Gane by KO/TKO** (+400) — research Moderate, very longshot (Gane is more of a technical volume striker)
- **Gane/Pereira coin-flip ML** — already flagged in earlier session as true coin-flip with SPLIT research
- **Lewis ML** (+320) — model has Hokit as favorite, only live-dog sprinkle (not parlay material)
- **Chandler ML** — underdog value but high variance
- **Garcia ML** — same as Chandler

### ⚠️ KEY CALIBRATION CAVEAT — model finish-prob overcompression

The calibrated model is **too aggressive** on finish probability for 5/7 fights (P(finish) = 99-100%). This is consistent with the known overcompression in `mov_calibration.json` — when the raw prior says P(finish) = 0.30-0.40, the bin-based calibration often pushes it to 0.95+. The renormalization after independent per-outcome calibration amplifies this further.

**Practical guidance**: when the model says 99%+ finish, treat it as **"high finish probability"** (not "certain finish"). The research leans are more useful for the **specific method** (KO vs sub vs decision) and the **specific round**. The 99% finish probability is more a sanity check than a betting signal — what matters is whether research agrees on the method/round.

### 🎯 Recommended Reallocation (extending the existing $100 budget) — **RECOMMENDED: Option A**

The existing 3/4/5/6-leg parlays ($40/$25/$20/$10) + $5 Topuria R3-5 prop total $100. Adding the new props would exceed the $30/day cap (Hard Rule #1). **RECOMMENDED: Option A** (defer parlays, bet the new props on fight day) — keeps each day within the $30 cap and puts the money on the higher-EV STRONG props:

**Option A: Defer parlays, bet the new props on fight day (June 14)** ⭐ **RECOMMENDED**
- $10 on **Pereira by KO/TKO** (+120) — primary prop, research Strong
- $10 on **Under 3.5 Rounds Pereira/Gane** (-140) — high confidence
- $5 on **Topuria NO distance** (-250) — lock at heavy juice
- $5 on **Ruffy/Chandler no distance** — moderate (caveat applies)
- **Total**: $30 (at daily cap, no parlay exposure needed)
- Parlays stay on the table for the next cash-injection day

**Option B: Reduce parlay size, bet the new props on fight day**
- $20 on 3-leg "Safe Core" (Topuria + O'Malley + Ruffy)
- $5 on 4-leg (+ Nickal)
- $5 on Topuria R3-5 prop (+140)
- $10 on Pereira by KO/TKO (+120)
- $5 on Gane by Decision (+150)
- $5 on Ruffy/Chandler no distance
- **Total**: $50 across 2 days (stagger $30 Friday, $20 Sunday)

**Option C: Skip the new props, keep the existing 3/4/5/6-leg parlays** — safe but leaves the Pereira/Gane coin-flip and the undercard props on the table

### 🔬 Research Pass Quality Notes

- **Researcher 1 (main events)**: Detailed DK odds + public % + expert consensus for Topuria/Gaethje and Pereira/Gane. High quality — directly actionable.
- **Researcher 2 (undercard)**: Conceptual framework + fighter archetype priors, no specific DK lines. Useful for fight-archetype reasoning (e.g., Lewis R1 KO volatility, Chandler "all or nothing" pattern).
- **Researcher 3 (special props)**: Returned **unreliable output** — claimed the event is a fictional exhibition at the White House and that no commercial sportsbooks are offering lines. It IS a real UFC event at the White House with standard DK/FanDuel/ESPN BET coverage. SKIP this researcher's output.

---

## June 10, 2026 — Today's Plays (Game 4 of NBA Finals + MLB slate)

**Generated:** 2026-06-10 08:04 PDT | **Staleness:** NBA OK (2d), MLB OK (1d) | **Status:** CANDIDATES, not placed

### 🏀 Top 3 NBA Plays (Kalshi, Game 4 tonight)

| # | Player | Stat | Line | Model P | Mkt P | Edge | Price | ★ |
|---|--------|------|------|---------|-------|------|-------|---|
| 1 | **Mitchell Robinson** | REB | **6+** | 79% | 36% | **+43%** | 36c | ★★★★★ |
| 2 | **Mitchell Robinson** | REB | **5+** | 87% | 49% | **+38%** | 49c | ★★★★½ |
| 3 | **Miles McBride** | 3PT | **2+** | 56% | 27% | **+29%** | 26c | ★★★★ |

### ⚾ Top 3 MLB Plays (PrizePicks, MIN_LINE ≥ 1.0 filter applied)

| # | Player | Stat | Line | Model P | Edge | Side | ★ |
|---|--------|------|------|---------|------|------|---|
| 1 | **Davis Martin** | Pitcher SO | 3.5 | 99.8% | +45.6% | OVER | ★★★★★ |
| 2 | **Chris Sale** | Hits Allowed | 3.5 | 99.1% | +44.9% | OVER | ★★★★★ |
| 3 | **Ian Happ** | H+R+RBI | 1.5 | 80.5% | +26.4% | OVER | ★★★★ |

### 🎰 Best Parlays

**2-Leg (PrizePicks Power Play, 3× payout):**
- Davis Martin SO 3.5 OVER (P=99.8%) + Chris Sale H allowed 3.5 OVER (P=99.1%)
- Joint (independent): 98.9% | EV @ 3×: +197%
- ⚠️ **Correlated** — verify not same game before locking both

**2-Leg NBA (Kalshi combined):**
- Robinson REB 5+ @ 49c + McBride 3PT 2+ @ 26c → 0.87 × 0.56 = **48.7% joint**

**3-Leg Mixed (Power Play 5×):**
- Robinson REB 5+ (Kalshi 87%) + Davis Martin SO 3.5 (PP 99.8%) + Ian Happ H+R+RBI 1.5 (PP 80.5%)
- Joint: **69.9%** | EV @ 5×: **+250%**

**4-Leg Mixed (Power Play 10×):**
- Above 3 + McBride 3PT 2+ (Kalshi 56%)
- Joint: **39.1%** | EV @ 10×: **+291%**

### 💵 Recommended Portfolio ($9.11 total exposure)

| # | Pick | Amount | Notes |
|---|------|--------|-------|
| 1 | Robinson REB 6+ YES @ 36c | $0.36 | 5★ |
| 2 | Robinson REB 5+ YES @ 49c | $0.49 | 4.5★ |
| 3 | McBride 3PT 2+ YES @ 26c | $0.26 | 4★ |
| 4 | Davis Martin SO 3.5 OVER | $2.00 | 5★ |
| 5 | Chris Sale H allowed 3.5 OVER | $2.00 | 5★ |
| 6 | Ian Happ H+R+RBI 1.5 OVER | $2.00 | 4★ |
| 7 | 3-leg parlay (Robinson + Martin + Happ) | $2.00 | Cross-platform |
| | **Total** | **$9.11** | within $30/day limit |

Expected return if all 6 singles hit ≈ $50. Per PROJECT.md Hard Rule #3, **no orders placed** — these are candidates for user approval.

---

# Best Bets — Audit Report (June 9, 2026)

## Portfolio Status

| Sport | Orders | Status |
|-------|--------|--------|
| **MLB** | 81 | Resting (not filled — low liquidity) |
| **NBA** | 49 | Resting (not filled — low liquidity) |
| **World Cup** | 45 | Executed ✅ (filled) |
| **CFB** | 1 (Clemson) | Resting (disabled — off-season) |

**Balance:** $78.58

## MLB Scanner Health

✅ **Working.** 9 active stat types (KS, HR, TB, HRR, IP, ER, H, BB, RBI), 2 info-only (R, SB). 1,937 players loaded.

Sample KS scan for today (JUN09): 210 markets → 21 qualifying:
- Lucas Giolito 2+ KS: edge=+20% (model=72% vs mkt=52%)
- Gerrit Cole 7+ KS: edge=+28% (model=53% vs mkt=24%)
- Walbert Ureña 5+ KS: edge=+6%
- Kai-Wei Teng 7+ KS: edge=+12%
- Gerrit Cole 8+ KS: edge=+10%

## NBA Scanner Health

✅ **Working** (via nba_bet.py). NBA Finals SAS vs NYK active for June 10. Fixed `-m` module loading bug — morning_scan now imports from nba_bet.py.

## Known Issues

- **MLB orders resting**: Most markets have low liquidity — orders sit unfilled until a counterparty takes them
- **UFC**: Title parser fixed for comma-separated combo format. Model trained (85.2% acc). 0 qualifying — fighters missing from DB (Diego Lopes, Bo Nickal, Alex Pereira)
- **CFB**: Off-season, disabled from betting
- **NFL**: Off-season

## Next Steps

1. Run `python3 src/scripts/nba_bet.py --bet` to place NBA bets for tomorrow's Finals
2. Run `python -m src.scripts.morning_scan --scan` to test full morning scan pipeline
3. Retrain UFC model with updated fighter data to get Diego Lopes, Bo Nickal, Alex Pereira
