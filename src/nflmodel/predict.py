"""
predict.py — predicts the next unplayed week using the hybrid model and saves a CSV
"""

import argparse
import datetime as dt
import numpy as np
import pandas as pd
from nflmodel.config import PREDICTIONS_DIR
from nflmodel.data.games import load_games
from nflmodel.features.build import build_features
from nflmodel.models.hybrid import HybridModel, EARLY_WEEKS
from nflmodel.models.elo import EloModel
from nflmodel.evaluation.metrics import win_prob
from nflmodel.qb_check import load_all_overrides, save_all_overrides, approved, \
    apply_cli_overrides, run_qb_checks


# the earliest season and week that still has unplayed games
def next_week():
    schedule = load_games(include_upcoming=True)
    upcoming = schedule[schedule["result"].isna()]
    season = int(upcoming["season"].min())
    week = int(upcoming[upcoming["season"] == season]["week"].min())
    return season, week


# one row per game: the model's margin, pick, win probability, and edge vs Vegas
def make_picks(this_week, pred, week):
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
    out["model"] = "elo" if week <= EARLY_WEEKS else "ridge"
    out["predicted_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    return out


# picks for games that have kicked off are final: keep them from the saved file
def keep_locked(out, path, this_week):
    if not path.exists():
        return out
    old = pd.read_csv(path)
    today = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d")
    already_played = ~old["game_id"].isin(this_week["game_id"]) | (old["gameday"] < today)
    locked = old[already_played]
    if locked.empty:
        return out
    out = out[~out["game_id"].isin(locked["game_id"])]
    print(f"Kept {len(locked)} locked pick(s): {', '.join(locked['game_id'])}\n")
    return pd.concat([locked, out], ignore_index=True).sort_values(["gameday", "game_id"])


# team rankings by Elo rating
def save_ratings(played, path):
    elo = EloModel().fit(played)
    ratings = pd.DataFrame(
        sorted(elo.ratings.items(), key=lambda item: item[1], reverse=True),
        columns=["team", "elo"],
    )
    ratings["elo"] = ratings["elo"].round(1)
    ratings.to_csv(path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the next unplayed week.")
    parser.add_argument("--approve", nargs="+", default=[], metavar="TEAM",
                        help="approve this week's suggested QB overrides for these teams")
    parser.add_argument("--set", nargs=2, action="append", default=[], metavar=("TEAM", "QB_ID"),
                        help="set a team's QB by id, e.g. --set ARI 00-0035289")
    args = parser.parse_args()

    # find next week first, so its overrides can be loaded before building features
    season, week = next_week()
    key = f"{season}_week{week:02d}"

    all_entries = load_all_overrides()
    entries = all_entries.setdefault(key, {})
    if apply_cli_overrides(entries, key, args.approve, args.set):
        save_all_overrides(all_entries)
    overrides = approved(entries)
    if overrides:
        print(f"Using {len(overrides)} QB override(s) for {key}: {', '.join(overrides)}\n")

    games = build_features(include_upcoming=True, qb_overrides=overrides)
    played = games[games["result"].notna()]
    this_week = games[(games["season"] == season) & (games["week"] == week) & games["result"].isna()]

    # train on every completed game, then predict this week
    model = HybridModel().fit(played)
    out = make_picks(this_week, model.predict(this_week), week)

    PREDICTIONS_DIR.mkdir(exist_ok=True)
    path = PREDICTIONS_DIR / f"{key}.csv"
    out = keep_locked(out, path, this_week)
    out.to_csv(path, index=False)

    ratings_path = PREDICTIONS_DIR / f"{key}_ratings.csv"
    save_ratings(played, ratings_path)

    print(out[["gameday", "away_team", "home_team", "spread_line",
               "pred_margin", "pick", "pick_win_prob", "edge"]].to_string(index=False))
    print(f"\nSaved {path}")
    print(f"Saved {ratings_path}")

    run_qb_checks(this_week, season, week, key, overrides, all_entries)
