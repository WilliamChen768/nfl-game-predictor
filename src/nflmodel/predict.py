"""
predict.py — predicts the next unplayed week using elo and saves a CSV
"""

import datetime as dt
import pandas as pd
import numpy as np
from nflmodel.config import PREDICTIONS_DIR
from nflmodel.data.games import load_games
from nflmodel.models.elo import EloModel
from nflmodel.evaluation.metrics import win_prob


if __name__ == "__main__":
    games = load_games(include_upcoming=True)
    played = games[games["result"].notna()]
    upcoming = games[games["result"].isna()]

    # next week is the earliest week that still has unplayed games
    season = upcoming["season"].min()
    week = upcoming[upcoming["season"] == season]["week"].min()
    this_week = upcoming[(upcoming["season"] == season) & (upcoming["week"] == week)]

    # train on every completed game, then predict this week
    model = EloModel().fit(played)
    pred = model.predict(this_week)

    out = this_week[["game_id", "gameday", "away_team", "home_team", "spread_line"]].copy()
    out["pred_margin"] = pred.round(1)                      # + means home team wins
    out["pick"] = np.where(pred > 0, out["home_team"], out["away_team"])
    home_prob = win_prob(pred)
    out["home_win_prob"] = home_prob.round(3)
    out["pick_win_prob"] = np.where(pred > 0, home_prob, 1 - home_prob).round(3)
    out["edge"] = (out["pred_margin"] - out["spread_line"]).round(1)   # model vs Vegas
    out["predicted_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    PREDICTIONS_DIR.mkdir(exist_ok=True)
    path = PREDICTIONS_DIR / f"{season}_week{week:02d}.csv"
    out.to_csv(path, index=False)

    # Save each team's current Elo rating, best to worst
    ratings = pd.DataFrame(
        sorted(model.ratings.items(), key=lambda item: item[1], reverse=True),
        columns=["team", "elo"],
    )
    ratings["elo"] = ratings["elo"].round(1)
    ratings_path = PREDICTIONS_DIR / f"{season}_week{week:02d}_ratings.csv"
    ratings.to_csv(ratings_path, index=False)

    print(out.drop(columns=["game_id", "predicted_at"]).to_string(index=False))
    print(f"\nSaved {path}")
    print(f"Saved {ratings_path}")