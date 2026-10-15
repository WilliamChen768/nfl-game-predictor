"""hybrid.py — Elo early in the season (when features are stale), ridge after."""

import numpy as np
from nflmodel.models.elo import EloModel
from nflmodel.models.ridge import RidgeModel

EARLY_WEEKS = 3  # Elo beat ridge in weeks 1-3 of the 2015-2025 backtest


class HybridModel:
    def __init__(self, early_weeks=EARLY_WEEKS):
        self.early_weeks = early_weeks

    # train both models on the same games
    def fit(self, train):
        self.elo = EloModel().fit(train)
        self.ridge = RidgeModel().fit(train)
        return self

    # Elo's prediction for early-season games, ridge's for the rest
    def predict(self, test):
        elo_pred = self.elo.predict(test)
        ridge_pred = self.ridge.predict(test)
        return np.where(test["week"] <= self.early_weeks, elo_pred, ridge_pred)