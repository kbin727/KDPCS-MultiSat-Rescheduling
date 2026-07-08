import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


ROOT = Path(r"E:\论文材料\DynamicReschedule_MultiSat")
WORK = ROOT / "results" / "_inspect_KDPCS_polished_v1"
FIG = WORK / "figures"
PREVIEW = FIG / "preview_v14"
CSV = ROOT / "results" / "final_kdpcs_all18_merged_20260705" / "complexity_validation" / "kdpcs_complexity_points.csv"

BLUE = "#163a76"
RED = "#bf3b36"
LIGHT_GRID = "#e3e3e3"

SCALE_LABELS = {
    "s3_n100": "S3-T100", "s3_n150": "S3-T150", "s3_n200": "S3-T200",
    "s5_n100": "S5-T100", "s5_n150": "S5-T150", "s5_n200": "S5-T200",
    "s8_n200": "S8-T200", "s8_n300": "S8-T300", "s8_n400": "S8-T400",
    "s10_n200": "S10-T200", "s10_n300": "S10-T300", "s10_n400": "S10-T400",
    "s15_n400": "S15-T400", "s15_n600": "S15-T600", "s15_n800": "S15-T800",
    "s20_n400": "S20-T400", "s20_n600": "S20-T600", "s20_n800": "S20-T800",
}


def setup():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    FIG.mkdir(parents=True, exist_ok=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)


def polish(ax):
    ax.grid(True, axis="y", color=LIGHT_GRID, linewidth=0.55, zorder=0)
    ax.tick_params(axis="both", labelsize=7.3, width=0.6, length=2.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)
        spine.set_color("#222222")


def save(fig, name):
    pdf = FIG / f"{name}.pdf"
    png = PREVIEW / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.006)
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.006)
    plt.close(fig)
    print(pdf)


def main():
    setup()
    df = pd.read_csv(CSV)
    df["C_million"] = df["candidate_pairs"] / 1e6
    agg = df.groupby("scale").agg(
        C_mean=("C_million", "mean"),
        t_mean=("seconds", "mean"),
        t_std=("seconds", "std"),
        group=("group", "first"),
    ).reset_index().sort_values("C_mean").reset_index(drop=True)

    group_colors = {"Small": "#6aa57a", "Medium": "#496795", "Large": "#cf6861"}
    x = np.arange(len(agg))

    fig, ax = plt.subplots(figsize=(3.65, 2.15))
    for i, row in agg.iterrows():
        ax.bar(i, row["t_mean"], width=0.58, color=group_colors[row["group"]],
               edgecolor="white", linewidth=0.4, alpha=0.82, zorder=3)
        ax.errorbar(i, row["t_mean"], yerr=row["t_std"], fmt="none",
                    ecolor="#333333", elinewidth=0.7, capsize=2, zorder=4)

    coeff = np.polyfit(agg["C_mean"].values, agg["t_mean"].values, 1)
    fit = np.polyval(coeff, agg["C_mean"].values)
    ax.plot(x, fit, color=RED, linewidth=1.25, linestyle="--",
            marker="D", markersize=3.3, markerfacecolor="white",
            markeredgewidth=0.8, label="Linear trend", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([SCALE_LABELS.get(s, s) for s in agg["scale"]],
                       rotation=55, ha="right", fontsize=6.2)
    ax.set_xlabel(r"Benchmark setting, ordered by $C$", fontsize=8.8)
    ax.set_ylabel("Runtime (s)", fontsize=8.8)
    ax.set_xlim(-0.7, len(agg) - 0.3)
    polish(ax)

    handles = [
        Patch(facecolor=group_colors["Small"], label="Small"),
        Patch(facecolor=group_colors["Medium"], label="Medium"),
        Patch(facecolor=group_colors["Large"], label="Large"),
        Line2D([0], [0], color=RED, lw=1.25, linestyle="--", marker="D",
               markerfacecolor="white", markersize=3.3, label="Linear trend"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=6.8, frameon=True,
              framealpha=0.92, borderpad=0.25, handlelength=1.1, labelspacing=0.22)
    fig.subplots_adjust(left=0.15, right=0.995, top=0.98, bottom=0.34)
    save(fig, "kdpcs_complexity_validation_20260705")


if __name__ == "__main__":
    main()
