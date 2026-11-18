"""Plot the active return interval for each momentum allocation."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def draw(output):
    output = Path(output)
    results = pd.read_csv(output / "allocation_summary.csv")
    fig, ax = plt.subplots(figsize=(8.1, 2.85), layout="constrained")
    for index, row in results.iterrows():
        color = "#985433" if row.sleeve_weight == 0.25 else "#344a59"
        lower = (row.active_arithmetic - row.block_low) * 100
        upper = (row.block_high - row.active_arithmetic) * 100
        ax.errorbar(
            row.active_arithmetic * 100,
            index,
            xerr=[[lower], [upper]],
            color=color,
            marker="o",
            capsize=4.5,
            lw=2,
        )
    ax.set(
        yticks=range(len(results)),
        yticklabels=[f"{w:.0%} momentum" for w in results.sleeve_weight],
        xlabel="Annual active return, percentage points (95% block interval)",
        xlim=(-1.6, 5.05),
        ylim=(-0.6, 3.55),
    )
    ax.axvline(0, color="#b5a99f", linestyle="--", lw=1)
    ax.grid(axis="x", color="#dfd9d2", alpha=0.65)
    ax.spines[["right", "top"]].set_visible(False)
    fig.savefig(output / "allocation_risk.png", dpi=180)
    plt.close(fig)
