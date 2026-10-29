"""
predict.py — predicts the next unplayed week using ridge and saves a CSV
"""

import datetime as dt
import numpy as np
import pandas as pd
import nflreadpy as nfl
from nflmodel.config import PREDICTIONS_DIR
from nflmodel.features.build import build_features
from nflmodel.models.hybrid import HybridModel
from nflmodel.models.elo import EloModel
from nflmodel.evaluation.metrics import win_prob

# Set a team's starter by hand when the news says the data is wrong,
# e.g. {"TB": "00-0041251"}. Clear it out each week.
QB_OVERRIDES = {"BAL": "00-0034796"}


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


# flag projected starters who are injured, or who differ from the depth chart's QB1
def check_qbs(this_week, qb1, names, injuries):
    flagged = False
    for side in ["home", "away"]:
        for team, qb in zip(this_week[f"{side}_team"], this_week[f"{side}_qb_id"]):
            if team in QB_OVERRIDES:
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
    games = build_features(include_upcoming=True, qb_overrides=QB_OVERRIDES)
    played = games[games["result"].notna()]
    upcoming = games[games["result"].isna()]

    # next week is the earliest week that still has unplayed games
    season = int(upcoming["season"].min())
    week = int(upcoming[upcoming["season"] == season]["week"].min())
    this_week = upcoming[(upcoming["season"] == season) & (upcoming["week"] == week)]

    # train on every completed game, then predict this week
    model = HybridModel().fit(played)
    pred = model.predict(this_week)

    out = this_week[["game_id", "gameday", "away_team", "home_team", "spread_line"]].copy()
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
    path = PREDICTIONS_DIR / f"{season}_week{week:02d}.csv"

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
    ratings_path = PREDICTIONS_DIR / f"{season}_week{week:02d}_ratings.csv"
    ratings.to_csv(ratings_path, index=False)

    print(out[["gameday", "away_team", "home_team", "spread_line",
               "pred_margin", "pick", "pick_win_prob", "edge"]].to_string(index=False))
    print(f"\nSaved {path}")
    print(f"Saved {ratings_path}")

    # make sure no issues qith qb
    print("\nQB check:")
    try:
        qb1, names = depth_chart_qbs(season)
        injuries = qb_injury_statuses(season, week)
        check_qbs(this_week, qb1, names, injuries)
    except Exception as e:
        print(f"  Couldn't load QB data ({e}). Check starters manually.")