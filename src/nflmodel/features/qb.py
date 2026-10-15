"""qb.py — qb starts in the team's previous n games"""

import pandas as pd

LOOKBACK = 15  # how many previous games to check


# count, for each team-game, how many of the team's previous LOOKBACK games this QB started
def qb_recent_starts(games, lookback=LOOKBACK):
    history = {}            # team -> list of QB ids, in game order
    home_starts, away_starts = [], []

    for game in games.itertuples():      # games are already in date order
        for team, qb, out in [(game.home_team, game.home_qb_id, home_starts),
                              (game.away_team, game.away_qb_id, away_starts)]:
            past = history.setdefault(team, [])
            out.append(past[-lookback:].count(qb))   # count BEFORE adding this game
            past.append(qb)

    return pd.DataFrame({
        "home_qb_starts": home_starts,
        "away_qb_starts": away_starts,
    }, index=games.index)