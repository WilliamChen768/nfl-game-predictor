"""
weekly.py — the weekly routine in one command: ingest, grade, predict.
Any extra arguments go to predict, e.g.  python -m nflmodel.weekly --approve PHI CLE
"""

import subprocess
import sys
import pandas as pd
from nflmodel.config import PREDICTIONS_DIR


# run one step as its own process, using this same Python; stop everything if it fails
def run(title, module, extra=()):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}", flush=True)
    subprocess.run([sys.executable, "-m", module, *extra], check=True)


# when is the next unplayed game in the latest picks file?
def next_kickoff():
    latest = sorted(PREDICTIONS_DIR.glob("*_week??.csv"))[-1]   # e.g. 2026_week09.csv
    picks = pd.read_csv(latest)
    today = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d")
    upcoming = picks[picks["gameday"] >= today].sort_values("gameday")
    if upcoming.empty:
        return latest.stem, None, None
    first = upcoming.iloc[0]
    return latest.stem, first["gameday"], f"{first['away_team']}@{first['home_team']}"


if __name__ == "__main__":
    predict_args = sys.argv[1:]          # passed straight through to predict

    run("1/3  Ingest", "nflmodel.data.ingest")
    run("2/3  Grade", "nflmodel.evaluation.grade")
    run("3/3  Predict", "nflmodel.predict", predict_args)

    key, day, game = next_kickoff()
    week = int(key.split("week")[1])          # "2026_week09" -> 9
    print(f"\n{'=' * 60}\nDone")
    if day:
        weekday = pd.Timestamp(day).day_name()
        print(f"Next kickoff: {weekday} {day} ({game}). Commit picks before then.")
    print("\nTo commit:")
    print(f"  git add predictions/{key}*.csv predictions/qb_overrides.json reports/graded.csv")
    print(f'  git commit -m "Week {week} picks"')
    print("  git push")