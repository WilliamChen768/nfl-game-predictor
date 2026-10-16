"""
calibration.py — groups games by predicted home win probability and compares with how often the home team actually won
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nflmodel.config import BACKTEST_SEASONS, ROOT
from nflmodel.evaluation.metrics import win_prob
from nflmodel.features.build import build_features
from nflmodel.models.hybrid import HybridModel

MIN_GAMES = 25                  # skip bins with too few games to trust
BINS = np.linspace(0, 1, 11)    # 0-10%, 10-20%, ..., 90-100%

# chart colors
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
MODEL_COLOR, VEGAS_COLOR = "#2a78d6", "#eb6834"


# for each probability bin: average predicted probability, actual win rate, and game count
def calibration_table(prob, outcome):
    df = pd.DataFrame({"prob": prob, "outcome": outcome})
    df["bin"] = pd.cut(df["prob"], BINS, include_lowest=True)
    table = df.groupby("bin", observed=True).agg(
        predicted=("prob", "mean"), actual=("outcome", "mean"), games=("prob", "size"))
    return table[table["games"] >= MIN_GAMES]


if __name__ == "__main__":
    games = build_features()
    rows = []
    for season in BACKTEST_SEASONS:
        train = games[games["season"] < season]
        test = games[games["season"] == season].copy()
        test["pred"] = HybridModel().fit(train).predict(test)
        rows.append(test)
    preds = pd.concat(rows)

    errors = preds["result"] - preds["pred"]
    print(errors.std().round(2))

    # home win = 1, loss = 0, tie = 0.5
    outcome = np.where(preds["result"] > 0, 1.0, np.where(preds["result"] < 0, 0.0, 0.5))
    model = calibration_table(win_prob(preds["pred"]), outcome)
    vegas = calibration_table(win_prob(preds["spread_line"]), outcome)

    print("Model:\n", model.round(3).to_string())
    print("\nVegas:\n", vegas.round(3).to_string())

    # --- chart ---
    fig, ax = plt.subplots(figsize=(6.5, 6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.5, ls="--", zorder=1)
    ax.text(0.93, 0.87, "perfect", color=MUTED, fontsize=9, rotation=45, ha="center")

    for table, color, label in [(model, MODEL_COLOR, "Model"), (vegas, VEGAS_COLOR, "Vegas")]:
        ax.plot(table["predicted"], table["actual"], color=color, lw=2,
                marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
                label=label, zorder=3)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    ax.set_xlabel("Predicted home win probability", color=INK)
    ax.set_ylabel("Actual home win rate", color=INK)
    ax.set_title(f"Calibration, {BACKTEST_SEASONS[0]}–{BACKTEST_SEASONS[-1]} "
                 f"({len(preds):,} games)", color=INK, loc="left")
    ax.grid(color=GRID, lw=1)
    ax.tick_params(colors=MUTED)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.legend(frameon=False, labelcolor=INK, loc="upper left")

    out = ROOT / "reports" / "calibration.png"
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")