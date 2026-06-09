# Best Bets — Morning Scan June 9, 2026

## Summary
- **Balance**: $89.88 → $79.11
- **Orders placed**: 19 total (5 singles + 14 parlay legs)
- **Total exposure**: $22.92
- **Active models**: 9 MLB + 11 NBA + 6 WNBA + 1 WC = 27 live models

---

## Singles Placed

| # | Sport | Player/Match | Bet | Edge | Price | Status |
|---|-------|-------------|-----|------|-------|--------|
| 1 | WC | Jordan vs Algeria | Jordan wins | +52% | 15¢ | ✅ Filled |
| 2 | WC | Turkiye vs USA | Turkiye wins | +141% | 36¢ | ✅ Filled |
| 3 | WC | Ecuador vs Germany | Ecuador wins | +258% | 19¢ | ✅ Filled |
| 4 | WC | Algeria vs Austria | Algeria wins | +175% | 25¢ | ✅ Filled |
| 5 | CFB | Clemson (LSU) | Clemson wins | +52% | 27¢ | ⚠️ Resting (disabled now) |

---

## Parlay Legs Placed

| # | Legs | Ticker | Bet | Edge | Price | Status |
|---|------|--------|-----|------|-------|--------|
| 1 | 2-leg | KXMLBHR-WSHCABRAMS5-1 | HR 1+ | +23% | 13¢ | Resting |
| 2 | 2-leg | KXMLBHRR-WSHCABRAMS5-5 | HRR 5+ | +22% | 13¢ | Resting |
| 3 | 2-leg | KXMLBHRR-WSHCABRAMS5-5 | HRR 5+ | +22% | 13¢ | Resting |
| 4 | 2-leg | KXMLBTB-WSHCABRAMS5-4 | TB 4+ | +27% | 19¢ | Resting |
| 5 | 3-leg | KXMLBHR-WSHCABRAMS5-1 | HR 1+ | +23% | 13¢ | Resting |
| 6 | 3-leg | KXMLBHRR-WSHCABRAMS5-5 | HRR 5+ | +22% | 13¢ | Resting |
| 7 | 3-leg | KXMLBTB-WSHCABRAMS5-4 | TB 4+ | +27% | 19¢ | Resting |
| 8 | 3-leg | KXMLBHR-WSHCABRAMS5-1 | HR 1+ | +23% | 13¢ | Resting |
| 9 | 3-leg | KXMLBHRR-WSHCABRAMS5-5 | HRR 5+ | +22% | 13¢ | Resting |
| 10 | 3-leg | KXMLBTB-WSHCABRAMS5-3 | TB 3+ | +33% | 26¢ | Resting |
| 11 | 4-leg | KXMLBKS-NYYGCOLE45-7 | KS 7+ | +53% | 25¢ | Resting |
| 12 | 4-leg | KXMLBHR-WSHCABRAMS5-1 | HR 1+ | +23% | 13¢ | Resting |
| 13 | 4-leg | KXMLBHRR-WSHCABRAMS5-5 | HRR 5+ | +22% | 13¢ | Resting |
| 14 | 4-leg | KXMLBTB-WSHCABRAMS5-4 | TB 4+ | +27% | 19¢ | Resting |

---

## Model Inventory

| Sport | Active Models | Status |
|-------|--------------|--------|
| **MLB** | KS, HR, TB, HRR, IP, ER, H, BB, RBI (9) | ✅ Live |
| **NBA** | PTS, REB, AST, BLK, STL, 3PT, FTM, PRA, PA, PR, RA (11) | ✅ Live |
| **WNBA** | PTS, REB, AST, FG3M, BLK, PRA (6) | ✅ Live |
| **WC** | Match winner (1) | ✅ Live |
| **CFB** | Game winner (1) | ❌ Disabled (off-season) |
| **NFL** | PASS_YDS, PASS_TD, RUSH_YDS, REC, REC_YDS, TD (6) | ❌ Off-season |
| **NHL** | GOALS, ASSISTS, POINTS, SHOTS, PIM (5) | ❌ Off-season |
| **UFC** | Fight winner (1) | ⚠️ Needs title parser fix |

---

## Notes
- NBA Finals JUN 10 markets are up (SAS vs NYK)
- WC model fixed: removed ISNS weighting → Brier 0.32 vs naive 0.39 (+16%)
- CFB disabled until season starts (~Aug 2026)
- UFC needs title parser fix for Kalshi multi-outcome format
- Morning scan folder: `reports/BestBets.md`
