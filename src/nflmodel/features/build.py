"""build.py — one row per game with every pre-game feature the models use"""

import pandas as pd
from nflmodel.data.games import load_games
from nflmodel.features.epa import load_plays, team_game_epa, add_priors, current_priors, add_qb_epa_prior
from nflmodel.features.qb import qb_recent_starts, fill_upcoming_qbs
from nflmodel.models.elo import EloModel
from nflmodel.models.massey import MasseyModel


PRIOR_COLS = ["off_epa_prior", "def_epa_prior", "off_pass_epa_prior",
              "def_pass_epa_prior", "qb_off_epa_prior"]


# attach one team's pre-game EPA to each game, with a home_ or away_ prefix
def add_team_features(games, team_epa, side):
    cols = team_epa[["game_id", "team"] + PRIOR_COLS]
    renames = {"team": f"{side}_team"}
    for c in PRIOR_COLS:
        renames[c] = f"{side}_{c}"
    cols = cols.rename(columns=renames)
    return games.merge(cols, on=["game_id", f"{side}_team"], how="left")


# every feature for every game
def build_features(include_upcoming=False, qb_overrides=None):
    games = load_games(include_upcoming)
    team_games = team_game_epa(load_plays())
    team_epa = add_priors(team_games)
    team_epa = add_qb_epa_prior(team_epa, games)

    df = add_team_features(games, team_epa, "home")
    df = add_team_features(df, team_epa, "away")

    for side in ["home", "away"]:
        df[f"{side}_qb_epa_diff"] = df[f"{side}_qb_off_epa_prior"] - df[f"{side}_off_epa_prior"]

    # upcoming games have no play-by-play: use each team's current average instead
    upcoming = df["result"].isna()
    if upcoming.any():
        current = current_priors(team_games)
        for side in ["home", "away"]:
            for c in current.columns:
                df.loc[upcoming, f"{side}_{c}"] = df.loc[upcoming, f"{side}_team"].map(current[c])
        df = fill_upcoming_qbs(df, qb_overrides)

    df["elo_margin"] = EloModel().pregame_margins(df)
    df["massey_margin"] = MasseyModel().pregame_margins(df)
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