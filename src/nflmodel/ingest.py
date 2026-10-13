"""
ingest.py — downloads nfl data from nflverse and saves it
"""

import datetime as dt
import nflreadpy as nfl
from .config import RAW_DIR, FIRST_SEASON, TEAM_ALIASES

# return current nfl season as a year
def current_season():
    today = dt.date.today()
    return today.year if today.month >= 3 else today.year - 1

# download every nfl game
def pull_schedules():
    df = nfl.load_schedules().to_pandas()
    for col in ["home_team", "away_team"]:
        df[col] = df[col].replace(TEAM_ALIASES)
    df.to_parquet(RAW_DIR / "schedules.parquet", index=False)
    print(f"schedules: {len(df)} games")

# set the play-by-play columns needed
PBP_COLUMNS = [
    "game_id", "season", "week", "season_type",
    "posteam", "defteam", "pass", "rush",
    "epa", "success", "passer_player_id",
]

# download every play from a season
def pull_pbp(season):
    path = RAW_DIR / f"pbp_{season}.parquet"
    if path.exists() and season != current_season():
        return  # already have it, and old seasons never change
    df = nfl.load_pbp([season]).select(PBP_COLUMNS).to_pandas()
    for col in ["posteam", "defteam"]:
        df[col] = df[col].replace(TEAM_ALIASES)
    df.to_parquet(path, index=False)
    print(f"pbp {season}: {len(df)} plays")


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    pull_schedules()
    for season in range(FIRST_SEASON, current_season() + 1):
        pull_pbp(season)