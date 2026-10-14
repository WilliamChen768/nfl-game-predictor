"""elo.py — every team has a rating that updates after each game"""

import math
import numpy as np
import pandas as pd

K = 20
HOME_BONUS = 40
REGRESS = 1/3
REST_BONUS = 5


# return the probability that the home team wins based on ratings
def expected_home_win(diff):
    return 1 / (1 + 10 ** (-diff / 400))

# turn the actual result into 1 for home win, 0 for loss, 0.5 for tie
def outcome_from_margin(result):
    if result > 0:
        return 1
    elif result == 0:
        return 0.5
    else:
        return 0

# bigger wins move ratings more
def mov_multiplier(result, diff):
    winner_diff = diff if result > 0 else -diff
    size = math.log(max(abs(result), 1) + 1)
    return size * 2.2 / (winner_diff * 0.001 + 2.2)

class EloModel:
    # run through every game and build ratings
    def fit(self, train):
        self.ratings = {}
        self.season = None
        for game in train.itertuples():
            self.process_game(game)
        return self

    # predict each game and return array of predictions
    def predict(self, test):
        preds = []
        for game in test.itertuples():
            preds.append(self.process_game(game))
        return np.array(preds)

    # predicts the outcome for a game (before seeing actual) and learns from the actual result
    def process_game(self, game):
        # new season: pull every rating partway back toward 1500
        if game.season != self.season:
            for team in self.ratings:
                self.ratings[team] += REGRESS * (1500 - self.ratings[team])
            self.season = game.season
        
        home = game.home_team
        away = game.away_team
        result = game.result
        neutral = game.neutral
        home_rating = self.ratings.get(home, 1500)
        away_rating = self.ratings.get(away, 1500)

        diff = home_rating - away_rating
        if not neutral:
            diff += HOME_BONUS
        diff += REST_BONUS * (game.home_rest - game.away_rest)
        predicted_margin = diff / 25

        expected = expected_home_win(diff)

        # if game hasnt been played, dont update ratings
        if pd.isna(result):
            return predicted_margin

        outcome = outcome_from_margin(result)
        mult = mov_multiplier(result, diff)
        self.ratings[home] = home_rating + K * mult * (outcome - expected)
        self.ratings[away] = away_rating - K * mult * (outcome - expected)
        return predicted_margin