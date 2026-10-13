import numpy as np
import pandas as pd
from scipy.stats import norm
from .config import BACKTEST_SEASONS, MARGIN_SD
from .games import load_games


class HomeFieldBaseline:
    # finds the games that arent neutral and takes the avg of the results
    # (home field advantage)
    def fit(self, train):
        games = train[train["neutral"] == 0]
        self.hfa = games["result"].mean()
        return self

    # predicts self.hfa for normal games or 0 for neutral
    def predict(self, test):
        return np.where(test["neutral"] == 0, self.hfa, 0)

# return the win prob based on the margin on a bell curve
def win_prob(margin):
    return norm.cdf(margin / MARGIN_SD)

# scores the model and vegas against actual (MAE and Brier)
def score(test, pred):
    actual = test["result"].to_numpy()
    line = test["spread_line"].to_numpy()
    
    outcome = np.where(actual >= 0, np.where(actual > 0, 1, 0.5), 0)

    # MAE for the model and Vegas
    model_diff = np.abs(pred - actual)
    mae_model = np.mean(model_diff)

    vegas_diff = np.abs(line - actual)
    mae_vegas = np.mean(vegas_diff)

    # Brier for model and Vegas
    brier_model = np.mean((win_prob(pred) - outcome)**2)
    brier_vegas = np.mean((win_prob(line) - outcome)**2)

    return {
        "games": len(test),
        "mae_model": mae_model,
        "mae_vegas": mae_vegas,
        "brier_model": brier_model,
        "brier_vegas": brier_vegas,
    }

# for each season, train on earlier ones and predict/score next
def backtest(model, games):
    rows = []
    for season in BACKTEST_SEASONS:
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
    print(backtest(HomeFieldBaseline(), games))