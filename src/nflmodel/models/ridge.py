"""ridge.py — linear model on pre-game features"""

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "home_off_epa_prior", "home_def_epa_prior",
    "away_off_epa_prior", "away_def_epa_prior",
    "elo_margin", "massey_margin", "rest_diff", "home_field",
    "home_qb_starts", "away_qb_starts",
    "home_qb_off_epa_prior", "away_qb_off_epa_prior",
]
ALPHA = 10


class RidgeModel:
    def __init__(self, features=FEATURES, alpha=ALPHA):
        self.features = features
        self.alpha = alpha

    def fit(self, train):
        train = train.dropna(subset=self.features)
        self.pipe = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
        self.pipe.fit(train[self.features], train["result"])
        return self

    def predict(self, test):
        return self.pipe.predict(test[self.features])