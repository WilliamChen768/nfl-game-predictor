"""Leakage tests: a game's pre-game features must not depend on that game itself."""

import pandas as pd
from nflmodel.features.epa import add_priors
from nflmodel.features.qb import qb_recent_starts


# one team's five games in order, with made-up EPA values
def team_games(off_epa):
    return pd.DataFrame({
        "team": ["AAA"] * 5,
        "season": [2025] * 5,
        "week": [1, 2, 3, 4, 5],
        "off_epa": off_epa,
        "def_epa": [0.0] * 5,
        "off_pass_epa": [0.0] * 5,
        "def_pass_epa": [0.0] * 5,
    })


def test_epa_prior_ignores_its_own_game():
    base = add_priors(team_games([0.1, 0.2, 0.3, 0.4, 0.5]))
    changed = add_priors(team_games([0.1, 0.2, 9.9, 0.4, 0.5]))   # change week 3 only

    # weeks 1-3 have the same priors: week 3's own EPA isn't part of its feature
    assert base["off_epa_prior"][:3].equals(changed["off_epa_prior"][:3])
    # later weeks do see the change
    assert base["off_epa_prior"][3] != changed["off_epa_prior"][3]


def test_first_game_has_no_prior():
    df = add_priors(team_games([0.1, 0.2, 0.3, 0.4, 0.5]))
    assert pd.isna(df["off_epa_prior"][0])


# five games between two teams, with the home QB given for each game
def qb_games(home_qbs):
    return pd.DataFrame({
        "home_team": ["AAA"] * 5,
        "away_team": ["BBB"] * 5,
        "home_qb_id": home_qbs,
        "away_qb_id": ["B1"] * 5,
    })


def test_qb_starts_count_only_earlier_games():
    base = qb_recent_starts(qb_games(["A1", "A1", "A1", "A1", "A1"]))
    changed = qb_recent_starts(qb_games(["A1", "A1", "A2", "A1", "A1"]))   # backup starts game 3

    # games 1-2 are unaffected, and A1's count before each game is how many he started earlier
    assert list(base["home_qb_starts"]) == [0, 1, 2, 3, 4]
    assert list(changed["home_qb_starts"][:2]) == [0, 1]
    # the backup has 0 earlier starts in his own game
    assert changed["home_qb_starts"][2] == 0