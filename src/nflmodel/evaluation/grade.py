"""
grade.py — scores saved prediction files against actual results
"""

import numpy as np
import pandas as pd
from nflmodel.config import PREDICTIONS_DIR
from nflmodel.data.games import load_games


# turn all weekly predictions CSV into a table
def load_predictions():
    files = sorted(p for p in PREDICTIONS_DIR.glob("*_week*.csv")
                   if not p.stem.endswith("_ratings"))
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

    by_week = df.groupby(["season", "week"]).apply(summarize)
    total = summarize(df)

    pd.set_option("display.float_format", "{:.2f}".format)
    print(by_week.to_string())
    print("\nSeason to date:")
    print(f"  Winners:  model {total['model_correct']}/{total['games']}, "
          f"Vegas {total['vegas_correct']}/{total['games']}")
    print(f"  MAE:      model {total['mae_model']:.2f}, Vegas {total['mae_vegas']:.2f}")
    print(f"  ATS:      {total['ats_w']}-{total['ats_l']}  (break-even is 52.4%)")

    df.to_csv(PREDICTIONS_DIR / "graded.csv", index=False)