"""backtest.py — train on earlier seasons, predict and score the next"""

import pandas as pd
from nflmodel.config import BACKTEST_SEASONS
from nflmodel.features.build import build_features
from nflmodel.evaluation.metrics import score
from nflmodel.models.baseline import HomeFieldBaseline
from nflmodel.models.elo import EloModel
from nflmodel.models.ridge import RidgeModel
from nflmodel.models.hybrid import HybridModel
from nflmodel.models.lgbm import LGBMModel
from nflmodel.models.massey import MasseyModel


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
    games = build_features()
    
    for model in [HomeFieldBaseline(), EloModel(), RidgeModel(), HybridModel(), LGBMModel(), MasseyModel()]:
        results = backtest(model, games)
        print(type(model).__name__)
        print(results[["season", "mae_model", "mae_vegas", "brier_model", "brier_vegas"]].round(3).to_string(index=False))
        print("Average:", results[["mae_model", "mae_vegas", "brier_model", "brier_vegas"]].mean().round(3).to_dict())
        print()