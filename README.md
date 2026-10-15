# NFL Game Predictor

Predicts the point margin of every NFL game, then measures those predictions
against the Vegas closing line, the toughest benchmark there is.

Every week's picks are committed to this repo before kickoff.

## Results

Walk-forward backtest over 2,400+ games (2015–2023): each season is predicted
using only games from earlier seasons.

| Model | MAE (points) | Gap to Vegas | Brier score |
|---|---|---|---|
| Home-field baseline | 10.94 | 1.10 | 0.246 |
| Elo | 10.35 | 0.51 | 0.229 |
| + margin of victory and rest | 10.17 | 0.33 | 0.222 |
| Ridge (EPA + Elo) | 10.13 | 0.30 | 0.221 |
| **+ QB familiarity (current)** | **10.09** | **0.26** | **0.220** |
| *Vegas closing line* | *9.84* | — | *0.213* |

- **MAE:** average number of points a prediction misses the final margin by.
- **Brier score:** accuracy of the win probabilities (0.25 = always guessing 50%).

The current model closes about **75% of the gap** between a naive baseline
and the betting market.

**Holdout (2024–2025, never used for tuning):** ridge MAE 9.99 vs Vegas 9.69,
a gap of 0.30 (Elo: 0.44).

**Live 2026 record:** starts week 6. See [`predictions/`](predictions/).

## How it works

1. **Ingest:** downloads schedules, results, Vegas lines, and play-by-play
   for 2006–present from [nflverse](https://github.com/nflverse).
2. **Features:** each team's offensive and defensive EPA per play over its
   previous 9 games, an Elo rating difference, rest days, home field, and how
   familiar the starting QB is (starts in the team's last 15 games). All are
   computed only from games before kickoff.
3. **Model:** ridge regression on those features.
4. **Backtest:** walk-forward by season, so the model never sees the future.
   2024–2025 are held out for a single final check.
5. **Predict:** each week, retrains on all completed games and saves
   timestamped picks. Projected starting QBs are cross-checked against depth
   charts, with manual overrides for confirmed changes.
6. **Grade:** scores live picks against results, Vegas, and the spread.

## Usage

```bash
pip install -e .
python -m nflmodel.data.ingest          # download / refresh data
python -m nflmodel.evaluation.backtest  # historical evaluation
python -m nflmodel.predict              # next week's picks
python -m nflmodel.evaluation.grade     # live record so far
```

## Project structure

```
src/nflmodel/
├── config.py          # shared paths and settings
├── data/              # download and clean data
├── features/          # EPA, QB, and the per-game feature table
├── models/            # baseline, Elo, and ridge
├── evaluation/        # metrics, backtest, live grading
└── predict.py         # weekly predictions
predictions/           # committed weekly picks (the public record)
```

*For educational purposes only. Not betting advice.*