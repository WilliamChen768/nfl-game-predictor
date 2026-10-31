"""
grade.py — scores saved prediction files against actual results
"""

import numpy as np
import pandas as pd
from nflmodel.config import PREDICTIONS_DIR, ROOT
from nflmodel.data.games import load_games


# turn all weekly predictions CSV into a table
def load_predictions():
    files = sorted(p for p in PREDICTIONS_DIR.glob("*_week*.csv")
                if not p.stem.endswith(("_ratings", "_qb_news")))
    return pd.concat([pd.read_csv(p) for p in files], ignore_index=True)

# combine stats of graded games into a row
def summarize(df):
    bets = df[df["ats_bet"]]
    return pd.Series({
        "games": len(df),
        "model_correct": int(df["model_correct"].sum()),
        "vegas_correct": int(df["vegas_correct"].sum()),
        "mae_model": df["model_miss"].mean(),
        "mae_vegas": df["vegas_miss"].mean(),
        "ats_w": int(bets["ats_win"].sum()),
        "ats_l": int((~bets["ats_win"]).sum()),
    })


if __name__ == "__main__":
    preds = load_predictions()
    played = load_games()  # completed games only

    # Keep only predicted games that have been played since
    df = preds.merge(played[["game_id", "season", "week", "result"]], on="game_id")
    if df.empty:
        print("No graded games yet. Check back after this week's games.")
        raise SystemExit

    winner = np.sign(df["result"])
    df["model_correct"] = np.sign(df["pred_margin"]) == winner
    df["vegas_correct"] = np.sign(df["spread_line"]) == winner
    df["model_miss"] = (df["pred_margin"] - df["result"]).abs()
    df["vegas_miss"] = (df["spread_line"] - df["result"]).abs()

    # Against the spread: our side is wherever we disagree with Vegas.
    # Skip games with no disagreement or that land exactly on the line (push).
    cover = df["result"] - df["spread_line"]
    df["ats_bet"] = (df["edge"] != 0) & (cover != 0)
    df["ats_win"] = np.sign(df["edge"]) == np.sign(cover)
    df["ats_side"] = np.where(df["edge"] > 0, df["home_team"], df["away_team"])

    counts = ["games", "model_correct", "vegas_correct", "ats_w", "ats_l"]
    by_week = df.groupby(["season", "week"]).apply(summarize)
    by_week[counts] = by_week[counts].astype(int)
    by_week["gap"] = by_week["mae_model"] - by_week["mae_vegas"]
    total = summarize(df)
    total[counts] = total[counts].astype(int)

    pd.set_option("display.float_format", "{:.2f}".format)
    print(by_week.to_string())
    print("\nSeason to date:")
    print(f"  Winners:  model {int(total['model_correct'])}/{int(total['games'])}, "
          f"Vegas {int(total['vegas_correct'])}/{int(total['games'])}")
    print(f"  MAE:      model {total['mae_model']:.2f}, Vegas {total['mae_vegas']:.2f}")
    print(f"  ATS:      {int(total['ats_w'])}-{int(total['ats_l'])}  (break-even is 52.4%)")
    print(f"  MAE:      model {total['mae_model']:.2f}, Vegas {total['mae_vegas']:.2f}, "
        f"gap {total['mae_model'] - total['mae_vegas']:.2f}")

    df = df.round({"model_miss": 1, "vegas_miss": 1})
    df.to_csv(ROOT / "reports" / "graded.csv", index=False)