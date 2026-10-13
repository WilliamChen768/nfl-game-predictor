"""
games.py — turns raw schedule into clean table
"""

import pandas as pd
from .config import RAW_DIR, FIRST_SEASON

# return every game since FIRST_SEASON
def load_games():
    df = pd.read_parquet(RAW_DIR / "schedules.parquet")

    df = df[(df["season"] >= FIRST_SEASON) & df["result"].notna()].copy()

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