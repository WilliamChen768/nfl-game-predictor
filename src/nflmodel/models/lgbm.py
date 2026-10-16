"""lgbm.py — gradient-boosted trees on the same pre-game features as ridge."""

from lightgbm import LGBMRegressor
from nflmodel.models.ridge import FEATURES

# conservative: shallow trees, slow learning, big leaves
PARAMS = dict(
    n_estimators=100,
    learning_rate=0.03,
    num_leaves=4,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    verbose=-1,
)


class LGBMModel:
    def __init__(self, features=FEATURES, **params):
        self.features = features
        self.params = {**PARAMS, **params}

    # learn an ensemble of small trees from past games
    def fit(self, train):
        train = train.dropna(subset=self.features)
        self.model = LGBMRegressor(**self.params)
        self.model.fit(train[self.features], train["result"])
        return self

    # predicted home margin for each game
    def predict(self, test):
        return self.model.predict(test[self.features])