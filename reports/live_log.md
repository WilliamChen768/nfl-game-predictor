# 2026 Live Log

Notes on each week's live picks, newest first. Picks are committed to
[`predictions/`](../predictions/) before kickoff; results are summarized in the
[README](../README.md#live-2026-record).

Gap = model MAE minus Vegas MAE (lower is better; backtest average 0.21).

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