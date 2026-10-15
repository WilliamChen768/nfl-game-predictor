"""build.py — one row per game with every pre-game feature the models use"""

import pandas as pd
from nflmodel.data.games import load_games
from nflmodel.features.epa import build_team_epa
from nflmodel.models.elo import EloModel


# attach one team's pre-game EPA to each game, with a home_ or away_ prefix
def add_team_features(games, team_epa, side):
    cols = team_epa[["game_id", "team", "off_epa_prior", "def_epa_prior"]]
    cols = cols.rename(columns={
        "team": f"{side}_team",
        "off_epa_prior": f"{side}_off_epa_prior",
        "def_epa_prior": f"{side}_def_epa_prior",
    })
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
    return df


if __name__ == "__main__":
    df = build_features()
    print(df.shape)
    print(df.isna().sum()[lambda s: s > 0])     # which columns have missing values
    cols = ["game_id", "home_off_epa_prior", "away_off_epa_prior", "elo_margin", "result"]
    print(df[cols].tail(5).to_string(index=False))