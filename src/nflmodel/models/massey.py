"""massey.py — least-squares team ratings: the ratings that best explain recent margins."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

HALF_LIFE_DAYS = 120   # a game this many days old counts half as much
ALPHA = 1.0            # small penalty: keeps ratings stable for teams with few games
CAP = 21


class MasseyModel:
    def __init__(self, half_life_days=HALF_LIFE_DAYS, alpha=ALPHA):
        self.half_life_days = half_life_days
        self.alpha = alpha

    # solve for each team's rating (and home field) using games played before `as_of`
    def _ratings(self, games, as_of):        
        past = games[games["result"].notna() & (games["gameday"] < as_of)]
        past = past[past["gameday"] >= as_of - pd.Timedelta(days=730)]   # last ~2 seasons
        if past.empty:
            return {}, 0.0

        teams = sorted(set(past["home_team"]) | set(past["away_team"]))
        idx = {t: i for i, t in enumerate(teams)}
        X = np.zeros((len(past), len(teams) + 1))
        rows = np.arange(len(past))
        X[rows, past["home_team"].map(idx)] = 1      # +1 for the home team
        X[rows, past["away_team"].map(idx)] = -1     # -1 for the away team
        X[:, -1] = (~past["neutral"]).astype(float)  # last column: home field

        age = (as_of - past["gameday"]).dt.days
        weight = 0.5 ** (age / self.half_life_days)  # recent games count more

        model = Ridge(alpha=self.alpha, fit_intercept=False)
        model.fit(X, past["result"].clip(-CAP, CAP), sample_weight=weight)
        return dict(zip(teams, model.coef_[:-1])), model.coef_[-1]

    # pre-game Massey margin for every game, each using only games before its week
    def pregame_margins(self, games):
        preds = pd.Series(index=games.index, dtype=float)
        for _, week_games in games.groupby(["season", "week"]):
            ratings, hfa = self._ratings(games, week_games["gameday"].min())
            home = week_games["home_team"].map(ratings).fillna(0)
            away = week_games["away_team"].map(ratings).fillna(0)
            preds[week_games.index] = home - away + hfa * (~week_games["neutral"])
        return preds.loc[games.index].to_numpy()

    # remember the history; ratings are solved fresh each week in predict
    def fit(self, train):
        self.history = train
        return self

    # re-solve ratings before each week, using every game played before it
    def predict(self, test):
        all_games = pd.concat([self.history, test])
        preds = pd.Series(index=test.index, dtype=float)
        for week, week_games in test.groupby("week"):
            ratings, hfa = self._ratings(all_games, week_games["gameday"].min())
            home = week_games["home_team"].map(ratings).fillna(0)
            away = week_games["away_team"].map(ratings).fillna(0)
            preds[week_games.index] = home - away + hfa * (~week_games["neutral"])
        return preds.loc[test.index].to_numpy()