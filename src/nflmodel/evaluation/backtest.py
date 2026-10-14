"""backtest.py — train on earlier seasons, predict and score the next"""

import pandas as pd
from nflmodel.config import BACKTEST_SEASONS, HOLDOUT_SEASONS
from nflmodel.data.games import load_games
from nflmodel.evaluation.metrics import score
from nflmodel.models.baseline import HomeFieldBaseline
from nflmodel.models.elo import EloModel


# for each season, train on earlier ones and predict/score next
def backtest(model, games, seasons=BACKTEST_SEASONS):
    rows = []
    for season in seasons:
        train = games[games["season"] < season]
        test = games[games["season"] == season]

        model.fit(train)
        pred = model.predict(test)
        result = score(test, pred)
        result["season"] = season
        rows.append(result)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    games = load_games()
    for model in [HomeFieldBaseline(), EloModel()]:
        results = backtest(model, games)
        print(type(model).__name__)
        print(results[["season", "mae_model", "mae_vegas", "brier_model", "brier_vegas"]].round(3).to_string(index=False))
        print("Average:", results[["mae_model", "mae_vegas", "brier_model", "brier_vegas"]].mean().round(3).to_dict())
        print()
    
    print("=== HOLDOUT (2024-2025) ===")
    holdout = backtest(EloModel(), games, seasons=HOLDOUT_SEASONS)
    print(holdout[["season", "mae_model", "mae_vegas", "brier_model", "brier_vegas"]].round(3).to_string(index=False))
    print("Average:", holdout[["mae_model", "mae_vegas", "brier_model", "brier_vegas"]].mean().round(3).to_dict())