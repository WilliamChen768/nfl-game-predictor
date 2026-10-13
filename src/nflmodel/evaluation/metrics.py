"""metrics.py — converts margins to win probabilities and scores predictions against Vegas"""

import numpy as np
from scipy.stats import norm
from nflmodel.config import MARGIN_SD

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