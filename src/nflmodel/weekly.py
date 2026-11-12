"""
weekly.py — the weekly routine in one command: ingest, grade, predict.
Extra arguments go to predict, e.g.  python -m nflmodel.weekly --approve PHI CLE
--summary FILE also writes a markdown summary (the weekly PR's description).
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
import pandas as pd
from nflmodel.config import PREDICTIONS_DIR, QB_OVERRIDES_FILE, ROOT
from nflmodel.evaluation.grade import summarize
from nflmodel.evaluation.report import ats_record

GRADED = ROOT / "reports" / "graded.csv"


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


# this week's picks as a markdown table (margins are home-team margins)
def picks_table(picks):
    rows = ["| Date | Game | Vegas | Model | Pick | Edge |", "|---|---|---|---|---|---|"]
    for r in picks.itertuples():
        rows.append(f"| {r.gameday} | {r.away_team}@{r.home_team} | {r.spread_line:g} | "
                    f"{r.pred_margin:g} | {r.pick} | {r.edge:+g} |")
    return "\n".join(rows)


# the weekly PR's description: deadline, season record, QB checkboxes, picks
def summary(key, day, game):
    week = int(key.split("week")[1])
    lines = [f"## Week {week} picks", ""]
    if day:
        lines.append(f"**Merge before {pd.Timestamp(day).day_name()} {day}** ({game} kicks off).")
    if GRADED.exists():
        t = summarize(pd.read_csv(GRADED))
        lines.append(f"Season to date: gap to Vegas {t['mae_model'] - t['mae_vegas']:.2f} "
                     f"over {int(t['games'])} games, ATS {ats_record(t)}.")

    all_entries = json.loads(QB_OVERRIDES_FILE.read_text()) if QB_OVERRIDES_FILE.exists() else {}
    entries = all_entries.get(key, {})
    pending = sorted((t, e) for t, e in entries.items() if not e.get("approved"))
    done = sorted((t, e) for t, e in entries.items() if e.get("approved"))
    lines += ["", "### QB suggestions"]
    if pending:
        lines.append("Tick to approve; the picks update automatically.")
        lines += [f"- [ ] {t}: {e.get('name', e['id'])} ({e['id']})" for t, e in pending]
    else:
        lines.append("None pending.")
    if done:
        lines.append("")
        lines.append("Approved: " + ", ".join(f"{t} ({e.get('name', e['id'])})" for t, e in done))

    picks = pd.read_csv(PREDICTIONS_DIR / f"{key}.csv")
    lines += ["", "### Picks", picks_table(picks)]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--summary")
    args, predict_args = parser.parse_known_args()   # everything else goes to predict

    run("1/3  Ingest", "nflmodel.data.ingest")
    run("2/3  Grade", "nflmodel.evaluation.grade")
    run("3/3  Predict", "nflmodel.predict", predict_args)

    key, day, game = next_kickoff()
    week = int(key.split("week")[1])
    print(f"\n{'=' * 60}\nDone")
    if day:
        print(f"Next kickoff: {pd.Timestamp(day).day_name()} {day} ({game}). Commit picks before then.")
    print("\nTo commit:")
    print(f"  git add predictions/{key}*.csv predictions/qb_overrides.json "
          "reports/graded.csv README.md reports/live_log.md")
    print(f'  git commit -m "Week {week} picks"')
    print("  git push")

    if args.summary:
        Path(args.summary).write_text(summary(key, day, game), encoding="utf-8")
        print(f"\nWrote PR summary to {args.summary}")