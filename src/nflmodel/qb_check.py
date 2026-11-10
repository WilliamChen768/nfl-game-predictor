"""
qb_check.py — weekly QB overrides and the checks that suggest them
(depth chart, injury report, Sleeper)
"""

import json
import pandas as pd
import nflreadpy as nfl
import requests
from nflmodel.config import PREDICTIONS_DIR, QB_OVERRIDES_FILE

SLEEPER_TEAMS = {"LAR": "LA"}            # Sleeper team codes that differ from nflverse
UNAVAILABLE = ("Out", "Doubtful", "IR", "PUP", "Sus")


# ---------- overrides file ----------

# all weeks' overrides
def load_all_overrides():
    return json.loads(QB_OVERRIDES_FILE.read_text()) if QB_OVERRIDES_FILE.exists() else {}


def save_all_overrides(all_entries):
    QB_OVERRIDES_FILE.write_text(json.dumps(all_entries, indent=2) + "\n")


# team -> QB id, only for the entries you've approved
def approved(entries):
    return {team: e["id"] for team, e in entries.items() if e.get("approved")}


# apply --approve and --set from the command line to this week's entries; True if anything changed
def apply_cli_overrides(entries, key, approve, set_pairs):
    changed = False
    for team in approve:
        team = team.upper()
        if team in entries:
            entries[team]["approved"] = True
            changed = True
        else:
            print(f"  No suggestion for {team} in {key}; use --set {team} <qb_id> instead.")
    for team, qb_id in set_pairs:
        entries[team.upper()] = {"id": qb_id, "approved": True, "source": "manual"}
        changed = True
    return changed


# ---------- data sources ----------

# each team's most recent depth-chart QB1, plus a name lookup for every QB listed
def depth_chart_qbs(season):
    depth = nfl.load_depth_charts([season]).to_pandas()
    qbs = depth[depth["pos_abb"] == "QB"]
    names = dict(zip(qbs["gsis_id"], qbs["player_name"]))
    qb1 = qbs[qbs["pos_rank"] == 1].sort_values("dt").groupby("team").tail(1)
    return dict(zip(qb1["team"], qb1["gsis_id"])), names


# each QB's injury-report game status for this week, e.g. {"00-00XXXXX": "Out"}
def qb_injury_statuses(season, week):
    inj = nfl.load_injuries([season]).to_pandas()
    inj = inj[(inj["week"] == week) & (inj["position"] == "QB") & inj["report_status"].notna()]
    return dict(zip(inj["gsis_id"], inj["report_status"]))


# every QB's status and depth order from Sleeper, with gsis ids from nflverse rosters
def sleeper_qbs(season):
    r = requests.get("https://api.sleeper.app/v1/players/nfl", timeout=30)
    r.raise_for_status()
    p = pd.DataFrame(r.json().values())
    p = p[(p["position"] == "QB") & p["team"].notna()].copy()
    p["team"] = p["team"].replace(SLEEPER_TEAMS)
    ids = (nfl.load_rosters([season]).to_pandas()[["sleeper_id", "gsis_id"]]
           .dropna().astype(str).drop_duplicates("sleeper_id"))
    p = p.merge(ids, left_on="player_id", right_on="sleeper_id", how="left", suffixes=("_sleeper", ""))
    p["gsis_id"] = p["gsis_id"].fillna(p["gsis_id_sleeper"].str.strip())
    return p[["team", "full_name", "status", "injury_status", "depth_chart_order", "gsis_id"]]


# ---------- checks ----------

# flag projected starters who are injured, or who differ from the depth chart's QB1
def check_qbs(this_week, qb1, names, injuries, overrides):
    flagged = False
    for side in ["home", "away"]:
        for team, qb in zip(this_week[f"{side}_team"], this_week[f"{side}_qb_id"]):
            if team in overrides:
                continue
            status = injuries.get(qb)
            if status:
                print(f"  CHECK {team}: projected {names.get(qb, qb)} is listed {status.upper()}")
                flagged = True
            listed = qb1.get(team)
            if listed and listed != qb:
                # if the depth chart's QB1 is out, the projected backup is expected; no need to flag
                if injuries.get(listed) not in ("Out", "Doubtful"):
                    print(f"  CHECK {team}: projected {names.get(qb, qb)}, "
                          f"depth chart QB1 {names.get(listed, listed)}")
                    flagged = True
    if not flagged:
        print("  No QB issues found.")


# flag projected starters Sleeper disagrees with; return suggested override entries
def sleeper_check(this_week, sq, overrides):
    suggestions = {}
    for side in ["home", "away"]:
        for team, qb in zip(this_week[f"{side}_team"], this_week[f"{side}_qb_id"]):
            if team in overrides:
                continue
            team_qbs = sq[sq["team"] == team]
            healthy = team_qbs[(team_qbs["status"] == "Active")
                               & ~team_qbs["injury_status"].isin(UNAVAILABLE)
                               & team_qbs["gsis_id"].notna()].sort_values("depth_chart_order")
            if healthy.empty:
                continue
            top = healthy.iloc[0]
            if top["gsis_id"] != qb:
                print(f"  SUGGEST {team}: Sleeper QB1 is {top['full_name']} ({top['gsis_id']})")
                suggestions[team] = {"id": top["gsis_id"], "name": top["full_name"],
                                     "approved": False, "source": "sleeper"}
    if not suggestions:
        print("  Sleeper agrees with all projected starters.")
    return suggestions


# run both checks for this week, and save any new Sleeper suggestions to the overrides file
def run_qb_checks(this_week, season, week, key, overrides, all_entries):
    entries = all_entries.setdefault(key, {})

    print("\nQB check:")
    try:
        qb1, names = depth_chart_qbs(season)
        injuries = qb_injury_statuses(season, week)
        check_qbs(this_week, qb1, names, injuries, overrides)
    except Exception as e:
        print(f"  Couldn't load QB data ({e}). Check starters manually.")

    print("\nSleeper check:")
    try:
        sq = sleeper_qbs(season)
        sq.to_csv(PREDICTIONS_DIR / f"{key}_qb_news.csv", index=False)
        suggestions = sleeper_check(this_week, sq, overrides)
        new = {t: s for t, s in suggestions.items()
               if t not in entries
               or (not entries[t].get("approved") and entries[t]["id"] != s["id"])}
        if new:
            entries.update(new)
            save_all_overrides(all_entries)
            print(f"\n  Added {len(new)} suggestion(s) to {QB_OVERRIDES_FILE.name} under {key}. "
                  f"Approve with: python -m nflmodel.predict --approve {' '.join(new)}")
    except Exception as e:
        print(f"  Couldn't reach Sleeper ({e}). Check starters manually.")
