"""qb.py — qb starts in the team's previous n games"""

import pandas as pd
from nflmodel.config import TEAM_ALIASES

LOOKBACK = 15  # how many previous games to check


# the starting QB for each team in each game: whoever threw the team's first pass
def pbp_starters(plays):
    passes = plays[plays["passer_player_id"].notna()].copy()
    passes["team"] = passes["posteam"].replace(TEAM_ALIASES)
    if "play_id" in passes.columns:
        passes = passes.sort_values(["game_id", "play_id"])
    first = passes.drop_duplicates(["game_id", "team"], keep="first")
    return dict(zip(zip(first["game_id"], first["team"]), first["passer_player_id"]))


# replace the schedule's QB ids with play-by-play starters wherever they disagree
def apply_pbp_starters(games, plays):
    starters = pbp_starters(plays)
    games = games.copy()
    names = {}
    for side in ["home", "away"]:
        names.update(zip(games[f"{side}_qb_id"], games[f"{side}_qb_name"]))
    for side in ["home", "away"]:
        keys = zip(games["game_id"], games[f"{side}_team"])
        from_pbp = pd.Series([starters.get(k) for k in keys], index=games.index)
        changed = from_pbp.notna() & (from_pbp != games[f"{side}_qb_id"])
        games.loc[changed, f"{side}_qb_id"] = from_pbp[changed]
        games.loc[changed, f"{side}_qb_name"] = from_pbp[changed].map(names)
        print(f"  pbp starters: corrected {changed.sum()} {side} QBs")
    return games


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


# for unplayed games, assume each team's most recent starter starts again (unless overridden)
def fill_upcoming_qbs(games, overrides=None):
    df = games.copy()
    last = {}
    for g in df[df["result"].notna()].itertuples():   # date order: later games overwrite earlier
        last[g.home_team] = g.home_qb_id
        last[g.away_team] = g.away_qb_id
    last.update(overrides or {})                       # manual overrides win

    upcoming = df["result"].isna()
    for side in ["home", "away"]:
        df.loc[upcoming, f"{side}_qb_id"] = df.loc[upcoming, f"{side}_team"].map(last)
    return df