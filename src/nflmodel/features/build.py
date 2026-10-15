"""build.py — one row per game with every pre-game feature the models use"""

import pandas as pd
from nflmodel.data.games import load_games
from nflmodel.features.epa import build_team_epa
from nflmodel.features.qb import qb_recent_starts
from nflmodel.models.elo import EloModel


PRIOR_COLS = ["off_epa_prior", "def_epa_prior", "off_pass_epa_prior", "def_pass_epa_prior"]


# attach one team's pre-game EPA to each game, with a home_ or away_ prefix
def add_team_features(games, team_epa, side):
    cols = team_epa[["game_id", "team"] + PRIOR_COLS]
    renames = {"team": f"{side}_team"}
    for c in PRIOR_COLS:
        renames[c] = f"{side}_{c}"
    cols = cols.rename(columns=renames)
    return games.merge(cols, on=["game_id", f"{side}_team"], how="left")


# every feature for every game, in date order
def build_features(include_upcoming=False):
    games = load_games(include_upcoming)
    team_epa = build_team_epa()

    df = add_team_features(games, team_epa, "home")
    df = add_team_features(df, team_epa, "away")

    df["elo_margin"] = EloModel().pregame_margins(df)
    df["rest_diff"] = df["home_rest"] - df["away_rest"]
    df["home_field"] = (~df["neutral"]).astype(int)
    df = df.join(qb_recent_starts(df))
    return df


if __name__ == "__main__":
    df = build_features()
    print(df.shape)
    print(df.isna().sum()[lambda s: s > 0])     # which columns have missing values
    cols = ["game_id", "home_off_epa_prior", "away_off_epa_prior", "elo_margin", "result"]
    print(df[cols].tail(5).to_string(index=False))