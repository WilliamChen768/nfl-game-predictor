"""epa.py — per-team epa per play with pre-game avgs"""

import pandas as pd
from nflmodel.config import RAW_DIR, FIRST_SEASON

WINDOW = 9  # how many past games the rolling average uses


# load every season's play-by-play and keep only pass/run plays
def load_plays(first=FIRST_SEASON, last=2026):
    pbp = pd.concat(
        [pd.read_parquet(RAW_DIR / f"pbp_{s}.parquet") for s in range(first, last + 1)],
        ignore_index=True,
    )
    is_play = (pbp["pass"] == 1) | (pbp["rush"] == 1)
    return pbp[is_play & pbp["epa"].notna()].copy()


# one row per team per game: overall and pass-only EPA per play, offense and defense
def team_game_epa(plays):
    offense = plays.groupby(["game_id", "season", "week", "posteam"]).agg(
        off_epa=("epa", "mean"),
        plays=("epa", "count"),
    ).reset_index().rename(columns={"posteam": "team"})

    defense = plays.groupby(["game_id", "defteam"]).agg(
        def_epa=("epa", "mean"),
    ).reset_index().rename(columns={"defteam": "team"})

    # same thing, but only pass plays
    passes = plays[plays["pass"] == 1]
    pass_off = passes.groupby(["game_id", "posteam"]).agg(
        off_pass_epa=("epa", "mean"),
    ).reset_index().rename(columns={"posteam": "team"})

    pass_def = passes.groupby(["game_id", "defteam"]).agg(
        def_pass_epa=("epa", "mean"),
    ).reset_index().rename(columns={"defteam": "team"})

    combined = offense
    for table in [defense, pass_off, pass_def]:
        combined = combined.merge(table, on=["game_id", "team"], how="left")
    return combined.sort_values(["team", "season", "week"]).reset_index(drop=True)


# add each team's average over its previous WINDOW games
def add_priors(team_games, window=WINDOW):
    df = team_games.copy()
    for col in ["off_epa", "def_epa", "off_pass_epa", "def_pass_epa"]:
        df[f"{col}_prior"] = (
            df.groupby("team")[col]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )
    return df


# plays -> team-games -> pre-game averages
def build_team_epa(window=WINDOW):
    return add_priors(team_game_epa(load_plays()), window)


if __name__ == "__main__":
    team_epa = build_team_epa()
    print(team_epa.shape)
    print(team_epa[team_epa["team"] == "SF"].tail(6).to_string(index=False))