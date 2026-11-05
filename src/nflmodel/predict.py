"""
predict.py — predicts the next unplayed week using the hybrid model and saves a CSV
"""

import datetime as dt
import json
import numpy as np
import pandas as pd
import nflreadpy as nfl
import requests
import argparse
from nflmodel.config import PREDICTIONS_DIR
from nflmodel.data.games import load_games
from nflmodel.features.build import build_features
from nflmodel.models.hybrid import HybridModel
from nflmodel.models.elo import EloModel
from nflmodel.evaluation.metrics import win_prob

# QB overrides for every week, keyed like "2026_week09". Only "approved": true entries are used.
OVERRIDES_FILE = PREDICTIONS_DIR / "qb_overrides.json"
SLEEPER_TEAMS = {"LAR": "LA"}            # Sleeper team codes that differ from nflverse
UNAVAILABLE = ("Out", "Doubtful", "IR", "PUP", "Sus")


# all weeks' overrides
def load_all_overrides():
    return json.loads(OVERRIDES_FILE.read_text()) if OVERRIDES_FILE.exists() else {}


def save_all_overrides(all_entries):
    OVERRIDES_FILE.write_text(json.dumps(all_entries, indent=2) + "\n")


# team -> QB id, only for the entries you've approved
def approved(entries):
    return {team: e["id"] for team, e in entries.items() if e.get("approved")}


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the next unplayed week.")
    parser.add_argument("--approve", nargs="+", default=[], metavar="TEAM",
                        help="approve this week's suggested QB overrides for these teams")
    parser.add_argument("--set", nargs=2, action="append", default=[], metavar=("TEAM", "QB_ID"),
                        help="set a team's QB by id, e.g. --set ARI 00-0035289")
    args = parser.parse_args()

    # find next week from the schedule first, so its overrides can be loaded before building features
    schedule = load_games(include_upcoming=True)
    upcoming = schedule[schedule["result"].isna()]
    season = int(upcoming["season"].min())
    week = int(upcoming[upcoming["season"] == season]["week"].min())

    key = f"{season}_week{week:02d}"
    all_entries = load_all_overrides()
    entries = all_entries.setdefault(key, {})

    changed = False
    for team in args.approve:
        team = team.upper()
        if team in entries:
            entries[team]["approved"] = True
            changed = True
        else:
            print(f"  No suggestion for {team} in {key}; use --set {team} <qb_id> instead.")
    for team, qb_id in args.set:
        entries[team.upper()] = {"id": qb_id, "approved": True, "source": "manual"}
        changed = True
    if changed:
        save_all_overrides(all_entries)

    overrides = approved(entries)
    if overrides:
        print(f"Using {len(overrides)} QB override(s) for {key}: {', '.join(overrides)}\n")

    games = build_features(include_upcoming=True, qb_overrides=overrides)
    played = games[games["result"].notna()]
    this_week = games[(games["season"] == season) & (games["week"] == week) & games["result"].isna()]

    # train on every completed game, then predict this week
    model = HybridModel().fit(played)
    pred = model.predict(this_week)

    out = this_week[["game_id", "gameday", "away_team", "home_team", "spread_line"]].copy()
    out["gameday"] = pd.to_datetime(out["gameday"]).dt.strftime("%Y-%m-%d")
    out["pred_margin"] = pred.round(1)                      # + means home team wins
    out["pick"] = np.where(pred > 0, out["home_team"], out["away_team"])
    home_prob = win_prob(pred)
    out["home_win_prob"] = home_prob.round(3)
    out["pick_win_prob"] = np.where(pred > 0, home_prob, 1 - home_prob).round(3)
    out["edge"] = (out["pred_margin"] - out["spread_line"]).round(1)   # model vs Vegas
    out["elo_margin"] = this_week["elo_margin"].round(1)               # for comparison
    out["home_qb_id"] = this_week["home_qb_id"]
    out["away_qb_id"] = this_week["away_qb_id"]
    out["model"] = "elo" if week <= 3 else "ridge"
    out["predicted_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    PREDICTIONS_DIR.mkdir(exist_ok=True)
    path = PREDICTIONS_DIR / f"{key}.csv"

    # Picks for games that have kicked off are final: keep them from the saved file
    if path.exists():
        old = pd.read_csv(path)
        today = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d")
        already_played = ~old["game_id"].isin(this_week["game_id"]) | (old["gameday"] < today)
        locked = old[already_played]
        if not locked.empty:
            out = out[~out["game_id"].isin(locked["game_id"])]
            out = pd.concat([locked, out], ignore_index=True).sort_values(["gameday", "game_id"])
            print(f"Kept {len(locked)} locked pick(s): {', '.join(locked['game_id'])}\n")

    out.to_csv(path, index=False)

    # show team rankings using elo ratings
    elo = EloModel().fit(played)
    ratings = pd.DataFrame(
        sorted(elo.ratings.items(), key=lambda item: item[1], reverse=True),
        columns=["team", "elo"],
    )
    ratings["elo"] = ratings["elo"].round(1)
    ratings_path = PREDICTIONS_DIR / f"{key}_ratings.csv"
    ratings.to_csv(ratings_path, index=False)

    print(out[["gameday", "away_team", "home_team", "spread_line",
               "pred_margin", "pick", "pick_win_prob", "edge"]].to_string(index=False))
    print(f"\nSaved {path}")
    print(f"Saved {ratings_path}")

    # make sure no issues with qb
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
            print(f"\n  Added {len(new)} suggestion(s) to {OVERRIDES_FILE.name} under {key}. "
                  f"Approve with: python -m nflmodel.predict --approve {' '.join(new)}")
    except Exception as e:
        print(f"  Couldn't reach Sleeper ({e}). Check starters manually.")