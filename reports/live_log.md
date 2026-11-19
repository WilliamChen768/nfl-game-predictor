# 2026 Live Log

Notes on each week's live picks, newest first. Picks are committed to
[`predictions/`](../predictions/) before kickoff; results are summarized in the
[README](../README.md#live-2026-record).

Gap = model MAE minus Vegas MAE (lower is better; backtest average 0.21).

---

## Week 10

**Record:** 6-6 ATS · MAE 10.84 (Vegas 10.57, gap 0.27) · winners 9/14 (Vegas 9/14)

**QB overrides:** ARI (Carson Beck), BAL (Jacoby Brissett), CLE (Deshaun Watson), LV (Fernando Mendoza), MIA (Kyle McCord), MIN (Kirk Cousins)

**Best vs Vegas**
- CAR@NO: model CAR 0.4, Vegas NO 2.5, result CAR 3 (beat Vegas by 2.9)
- BUF@NYJ: model BUF 10.5, Vegas BUF 8.5, result BUF 19 (beat Vegas by 2.0)
- SF@DAL: model SF 4.1, Vegas SF 3.5, result SF 6 (beat Vegas by 0.6)

**Worst vs Vegas**
- MIN@GB: model GB 6.5, Vegas GB 3, result MIN 20 (Vegas better by 3.5)
- NE@DET: model NE 0.6, Vegas DET 2.5, result DET 25 (Vegas better by 3.1)
- LA@ARI: model LA 8.3, Vegas LA 9.5, result LA 28 (Vegas better by 1.2)

**Notes:** Six QB overrides, mostly neutral. MIN@GB was the worst miss: Cousins (traded
from LV) started and MIN won by 20, but the model moved toward GB when he was approved
(6.2 → 6.5) because he had zero MIN starts; familiarity treats a traded veteran like a
backup. Second time this season the model misjudged a QB change direction (see ARI@SEA,
week 9). Strongest case so far for a per-QB rating. My own call, SF by 15 at DAL: SF won by 6.

**Running total:** gap 0.12 over 71 games. ATS 32-36-1.

---

## Week 9

**Record:** 7-8 ATS · MAE 8.52 (Vegas 8.43, gap 0.09) · winners 14/15 (Vegas 14/15)

**QB overrides:** ARI (Gardner Minshew), BAL (Joe Fagnano), CLE (Shedeur Sanders), PHI (Joe Milton), SEA (Drew Lock)

**Best vs Vegas**
- CIN@ATL: model CIN 6.5, Vegas CIN 3, result CIN 17 (beat Vegas by 3.5)
- HOU@LAC: model HOU 6.2, Vegas HOU 4.5, result HOU 22 (beat Vegas by 1.7)
- LV@SF: model SF 11.3, Vegas SF 10, result SF 21 (beat Vegas by 1.3)

**Worst vs Vegas**
- JAX@BAL: model JAX 6.4, Vegas JAX 10, result JAX 23 (Vegas better by 3.6)
- BUF@MIN: model BUF 1.3, Vegas BUF 3.5, result BUF 17 (Vegas better by 2.2)
- NYJ@KC: model KC 8.1, Vegas KC 10, result KC 23 (Vegas better by 1.9)

**Notes:** Five QB overrides, all set midweek from news and Sleeper. Sleeper's calls held up:
PHI with Milton won by 8 (model PHI 3.9, Vegas 3.5) and CLE with Sanders lost by 1 at NO.
JAX@BAL was the worst miss: with Lamar and Huntley out, BAL started third-stringer Fagnano;
the override moved the model only 1.6 points (JAX 4.8 → 6.4) while Vegas had JAX 10, and
JAX won by 23. The model undervalues a starter → third-stringer drop. ARI@SEA with backups
on both sides went better than feared: model SEA 9.3, Vegas 8.5, SEA won by 14.

**Running total:** gap 0.09 over 57 games. ATS 26-30-1.

---

## Week 8

**Record:** 9-5 ATS · MAE 10.11 (Vegas 10.96, gap −0.85) · winners 11/14 (Vegas 10/14)

**What changed**
- QB starters now come from play-by-play (each team's first passer). The schedule's
  QB field was wrong for ~6% of 2026 team-games (e.g. SEA weeks 3–6, PHI week 7).
- Added a Sleeper check that flags injured or IR starters and suggests overrides.
- Picks for games that have kicked off are now locked; tested on CAR@GB (Thursday).

**What went right**
- The two biggest edges won: KC@DEN (model DEN 2.9, result DEN 4) and BAL@BUF
  (model BUF 7.2 vs Vegas 3, result BUF 26).
- TEN@CIN: the line moved 10 → 12.5 before the final run, flipping the side to
  TEN, which covered.

**What went wrong**
- PHI@WAS: model WAS 2.5 with McKee, Vegas 6, result WAS 17. The correct QB helped
  (the Hurts projection had WAS 0.9), but the model still undervalues a starter →
  backup downgrade.
- BAL@BUF: the Lamar override was correct before kickoff (8.4 → 7.2), but he was
  hurt on the first play and Huntley played the game. BUF won by 26.

**Is this unusual?** No, it's the mirror image of week 6 (weekly std 0.90). The
week 6 worry about over-confidence didn't repeat: this week the largest edges were right.

**Running total:** gap 0.09 over 42 games, consistent with the backtest (0.21).
ATS 19-22-1.

---

## Week 7

**Record:** 5-8-1 ATS · MAE 10.61 (Vegas 10.54, gap 0.08) · winners 10/14 (Vegas 9/14)

**What went right**
- Gap to Vegas of 0.08, better than the backtest average (0.21).
- SEA override (Darnold) held up: SEA won by 17.

**What went wrong**
- Two blowouts nobody predicted: TB won at CAR by 30 (Vegas missed by 33.5) and
  CIN won at BAL by 22 (Vegas missed by 25).
- The BAL and TB QB flags weren't resolved before kickoff (no Saturday rerun).

**What changed**
- Picks are now locked once a game kicks off, so later reruns can't change them.

**Running total:** gap 0.56 over 28 games. ATS 10-17-1, within normal variance for 27 bets.

---

## Week 6

**Record:** 5-9 ATS · MAE 7.64 (Vegas 6.61, gap 1.04) · winners 7/14 (Vegas 9/14)

**What went wrong**
- PHI and CHI: the projected QB didn't start. Without these two games the gap
  is 0.76, so they explain about a quarter of it.
- The rest was spread out. In most misses the model was more confident than
  Vegas (NE 11.8 vs 7.5, LV −8.4 vs −5.5, SF 10.0 vs 7.5) and the result landed
  near the line.
- Best calls: LA (model 9.2 vs Vegas 12.5, result −2) and DEN (0.6 vs −2.5,
  result 3).

**Is this unusual?** No. In the backtest, 13% of weeks had a gap of 1.04 or
worse, and the weekly gap has a standard deviation of 0.90. This is normal
variance, not a model problem, so no model changes.

**What changed**
- Added an injury-report QB check to `predict.py`. Replayed on week 6, it
  flags both PHI and CHI.

**Watch:** if "more confident than Vegas and wrong" keeps showing up over
several weeks, the live predictions may be too spread out.
