"""
report.py — writes graded results into the README's live table and the live log
"""

import json
from nflmodel.config import ROOT, QB_OVERRIDES_FILE
from nflmodel.data.games import load_games

README = ROOT / "README.md"
START, END = "<!-- LIVE_START -->", "<!-- LIVE_END -->"
LOG = ROOT / "reports" / "live_log.md"


# "9-5" or "5-8-1" when there was a push
def ats_record(r):
    s = f"{int(r['ats_w'])}-{int(r['ats_l'])}"
    return s + f"-{int(r['ats_p'])}" if r["ats_p"] else s


# ---------- README live table ----------

# the README's live table as markdown, one row per week plus a total
def live_table(by_week, total):
    rows = ["| Week | Games | MAE | Vegas MAE | Gap | ATS | Winners (model / Vegas) |",
            "|---|---|---|---|---|---|---|"]
    for (season, week), r in by_week.iterrows():
        rows.append(f"| {week} | {int(r['games'])} | {r['mae_model']:.2f} | {r['mae_vegas']:.2f} | "
                    f"{r['mae_model'] - r['mae_vegas']:.2f} | {ats_record(r)} | "
                    f"{int(r['model_correct'])} / {int(r['vegas_correct'])} |")
    t = total
    rows.append(f"| **Total** | **{int(t['games'])}** | **{t['mae_model']:.2f}** | "
                f"**{t['mae_vegas']:.2f}** | **{t['mae_model'] - t['mae_vegas']:.2f}** | "
                f"**{ats_record(t)}** | **{int(t['model_correct'])} / {int(t['vegas_correct'])}** |")
    return "\n".join(rows)


# replace everything between the README markers with the new table
def update_readme(table):
    text = README.read_text(encoding="utf-8")
    if START not in text or END not in text:
        print("README markers not found; live table not updated.")
        return
    before, rest = text.split(START, 1)
    after = rest.split(END, 1)[1]
    README.write_text(f"{before}{START}\n{table}\n{END}{after}", encoding="utf-8")
    print("Updated the live table in README.md.")


# ---------- live log draft ----------

# a home-positive margin as text, e.g. "DEN 2.9"
def margin_text(m, home, away):
    return "even" if m == 0 else f"{home if m > 0 else away} {abs(m):g}"


# one game as a log bullet: model, Vegas, result, and who was closer
def game_line(r):
    h, a = r["home_team"], r["away_team"]
    extra = r["model_miss"] - r["vegas_miss"]
    verdict = ("same as Vegas" if extra == 0 else
               f"beat Vegas by {-extra:.1f}" if extra < 0 else f"Vegas better by {extra:.1f}")
    return (f"- {a}@{h}: model {margin_text(r['pred_margin'], h, a)}, "
            f"Vegas {margin_text(r['spread_line'], h, a)}, "
            f"result {margin_text(r['result'], h, a)} ({verdict})")


# the numbers part of a week's log entry; you fill in the Notes
def draft_entry(season, week, week_df, row, total):
    key = f"{season}_week{week:02d}"
    ov = json.loads(QB_OVERRIDES_FILE.read_text()).get(key, {}) if QB_OVERRIDES_FILE.exists() else {}
    used = [f"{t} ({e.get('name', e['id'])})" for t, e in sorted(ov.items()) if e.get("approved")]
    w = week_df.assign(extra=week_df["model_miss"] - week_df["vegas_miss"]).sort_values("extra")
    games = int(row["games"])
    lines = [
        f"## Week {week}", "",
        f"**Record:** {ats_record(row)} ATS · MAE {row['mae_model']:.2f} "
        f"(Vegas {row['mae_vegas']:.2f}, gap {row['mae_model'] - row['mae_vegas']:.2f}) · "
        f"winners {int(row['model_correct'])}/{games} (Vegas {int(row['vegas_correct'])}/{games})", "",
        f"**QB overrides:** {', '.join(used) if used else 'none'}", "",
        "**Best vs Vegas**", *[game_line(r) for _, r in w.head(3).iterrows()], "",
        "**Worst vs Vegas**", *[game_line(r) for _, r in w.tail(3).iloc[::-1].iterrows()], "",
        "**Notes:** TODO", "",
        f"**Running total:** gap {total['mae_model'] - total['mae_vegas']:.2f} over "
        f"{int(total['games'])} games. ATS {ats_record(total)}.",
    ]
    return "\n".join(lines)


# put a new entry at the top of the log (under the intro), unless that week is already there
def add_log_entry(week, entry):
    text = LOG.read_text(encoding="utf-8")
    if f"## Week {week}\n" in text:
        print(f"Log already has week {week}; no draft added.")
        return
    head, rest = text.split("\n---\n", 1)
    LOG.write_text(f"{head}\n---\n\n{entry}\n\n---\n{rest}", encoding="utf-8")
    print(f"Added a week {week} draft to {LOG.name}. Fill in the Notes.")


# draft the latest week's log entry, but only once every game that week is graded
def draft_latest_week(df, by_week, total):
    season, week = by_week.index[-1]
    schedule = load_games(include_upcoming=True)
    scheduled = ((schedule["season"] == season) & (schedule["week"] == week)).sum()
    week_df = df[(df["season"] == season) & (df["week"] == week)]
    if len(week_df) < scheduled:
        print(f"Week {week}: {len(week_df)}/{scheduled} games graded; log draft waits until it's done.")
        return
    add_log_entry(week, draft_entry(season, week, week_df, by_week.loc[(season, week)], total))
