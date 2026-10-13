"""
games.py — turns raw schedule into clean table
"""

import pandas as pd
from nflmodel.config import RAW_DIR, FIRST_SEASON

# return every game since FIRST_SEASON (include unplayed if include_upcoming=True)
def load_games(include_upcoming=False):
    df = pd.read_parquet(RAW_DIR / "schedules.parquet")

    recent = df["season"] >= FIRST_SEASON
    played = df["result"].notna()
    if include_upcoming:
        df = df[recent].copy()
    else:
        df = df[recent & played].copy()

    df["neutral"] = df["location"] == "Neutral"
    df["playoff"] = df["game_type"] != "REG"
    df["gameday"] = pd.to_datetime(df["gameday"])

    columns = [
        "game_id", "season", "week", "gameday",
        "home_team", "away_team",
        "result",        # home score minus away score (predict)
        "spread_line",   # expected home margin (benchmark)
        "home_rest", "away_rest", "div_game", "neutral", "playoff",
    ]

    return df[columns].sort_values(["gameday", "game_id"]).reset_index(drop=True)