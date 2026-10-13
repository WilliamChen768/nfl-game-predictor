"""baseline.py — every home team wins by the average home margin"""

import numpy as np

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