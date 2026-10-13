"""elo.py — every team has a rating that updates after each game"""

import numpy as np

K = 35
HOME_BONUS = 40
REGRESS = 1/4


# return the probability that the home team wins based on ratings
def expected_home_win(home_rating, away_rating, neutral):
    diff = home_rating - away_rating
    if not neutral:
        diff += HOME_BONUS
    return 1 / (1 + 10 ** (-diff / 400))

# return the predicted outcome based on margin (1 for win, 0 for loss, 0.5 for tie)
def outcome_from_margin(result):
    if result > 0:
        return 1
    elif result == 0:
        return 0.5
    else:
        return 0

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

        expected = expected_home_win(home_rating, away_rating, neutral)

        diff = home_rating - away_rating
        if not neutral:
            diff += HOME_BONUS
        predicted_margin = diff / 25

        outcome = outcome_from_margin(result)
        self.ratings[home] = home_rating + K * (outcome - expected)
        self.ratings[away] = away_rating - K * (outcome - expected)
        return predicted_margin