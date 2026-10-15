"""build.py — one row per game with every pre-game feature the models use"""

import pandas as pd
from nflmodel.data.games import load_games
from nflmodel.features.epa import load_plays, team_game_epa, add_priors, current_priors
from nflmodel.features.qb import qb_recent_starts, fill_upcoming_qbs
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


# every feature for every game
def build_features(include_upcoming=False, qb_overrides=None):
    games = load_games(include_upcoming)
    team_games = team_game_epa(load_plays())
    team_epa = add_priors(team_games)

    df = add_team_features(games, team_epa, "home")
    df = add_team_features(df, team_epa, "away")

    # upcoming games have no play-by-play: use each team's current average instead
    upcoming = df["result"].isna()
    if upcoming.any():
        current = current_priors(team_games)
        for side in ["home", "away"]:
            for c in PRIOR_COLS:
                df.loc[upcoming, f"{side}_{c}"] = df.loc[upcoming, f"{side}_team"].map(current[c])
        df = fill_upcoming_qbs(df, qb_overrides)

    df["elo_margin"] = EloModel().pregame_margins(df)
    df["rest_diff"] = df["home_rest"] - df["away_rest"]
    df["home_field"] = (~df["neutral"]).astype(int)
    df["season_progress"] = (df["week"].clip(upper=4) - 1) / 3 # less epa trust early in season
    for side in ["home", "away"]:
        for c in ["off_epa_prior", "def_epa_prior"]:
            df[f"{side}_{c}_x_prog"] = df[f"{side}_{c}"] * df["season_progress"]
    df = df.join(qb_recent_starts(df))
    return df


if __name__ == "__main__":
    df = build_features()
    print(df.shape)
    print(df.isna().sum()[lambda s: s > 0])     # which columns have missing values
    cols = ["game_id", "home_off_epa_prior", "away_off_epa_prior", "elo_margin", "result"]
    print(df[cols].tail(5).to_string(index=False))