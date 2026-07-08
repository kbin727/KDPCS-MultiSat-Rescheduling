"""
Redraw Fig.8(b) Runtime preference-count sensitivity and Fig.10 S8-T300 gantt.

Fig.8(b): boxplot + mean line, remove K=50 text, keep vertical dashed line,
          y-axis from 0 to 1.08 * max whisker cap, legend outside top.
Fig.10 :  S8-T300 inst=1, pref_id=0,2,4 -> omega=(1,0),(0.5,0.5),(0,1).
          Red = Inserted task (original_scheduled=False),
          Blue = Original-plan task (original_scheduled=True).
          Preference labels above each subplot, legend below the figure.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(r"E:\论文材料\DynamicReschedule_MultiSat")
WORK = ROOT / "results" / "KDPCS_overleaf_clean_20260709_final_fig8_fig10_polished_work"
FIG = WORK / "figures"
PREVIEW = FIG / "preview_v14"

PREF_ROWS = (
    ROOT
    / "results"
    / "server_preference_sensitivity_20260708"
    / "preference_count_sensitivity"
    / "preference_count_sensitivity_rows.csv"
)
GANTT = (
    ROOT
    / "results"
    / "_tmp_gantt_candidate_search_20260709"
    / "s8_n300_inst1"
    / "preference_tradeoff_schedules.csv"
)

BLUE = "#1d3f79"
RED = "#c83a35"
BOX_FACE = "#dbe5f2"
LIGHT_GRID = "#e3e3e3"
K_VALUES = [20, 50, 100, 150, 200]


def setup():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    FIG.mkdir(parents=True, exist_ok=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)


def polish(ax):
    ax.grid(True, axis="y", color=LIGHT_GRID, linewidth=0.55, zorder=0)
    ax.tick_params(axis="both", labelsize=7.6, width=0.6, length=2.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)
        spine.set_color("#222222")


def save(fig, name):
    pdf = FIG / f"{name}.pdf"
    png = PREVIEW / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.006)
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.006)
    plt.close(fig)
    print(f"  PDF: {pdf}")
    print(f"  PNG: {png}")


def plot_runtime():
    """Fig.8(b): Runtime preference-count sensitivity."""
    print("\n=== Fig.8(b): Runtime preference-count sensitivity ===")
    rows = pd.read_csv(PREF_ROWS)
    rows["scale_group"] = rows["scale_group"].astype(str).str.lower().str.strip()
    rows = rows[rows["scale_group"] == "all"].copy()
    if rows.empty:
        rows = pd.read_csv(PREF_ROWS)

    x = np.arange(len(K_VALUES))
    data = [rows[rows["num_prefs"] == K]["seconds"].values for K in K_VALUES]

    fig, ax = plt.subplots(figsize=(3.55, 1.78))
    bp = ax.boxplot(
        data, positions=x, widths=0.48, patch_artist=True, showfliers=False,
        medianprops={"color": "#1a1a1a", "linewidth": 0.8},
        boxprops={"linewidth": 0.75, "color": BLUE},
        whiskerprops={"linewidth": 0.65, "color": BLUE},
        capprops={"linewidth": 0.65, "color": BLUE},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(BOX_FACE)
        patch.set_alpha(0.82)

    # mean line
    means = [np.mean(d) for d in data]
    ax.plot(x, means, color=BLUE, linewidth=1.35, marker="o", markersize=3.8,
            markerfacecolor="white", markeredgewidth=1.0, zorder=5)

    # keep K=50 vertical dashed line, NO K=50 text label
    ax.axvline(1, color="#999999", linewidth=0.7, linestyle=(0, (3, 3)), zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.set_xlabel(r"Preference count $K$", fontsize=8.8)
    ax.set_ylabel("Runtime (s)", fontsize=8.8)

    # y-axis: 0 to 1.08 * max whisker cap (no truncation of whiskers)
    whisker_caps = [w.get_ydata()[1] for w in bp["whiskers"][1::2]]  # upper caps
    y_top = max(whisker_caps) * 1.08
    ax.set_ylim(0, y_top)

    polish(ax)
    ax.legend(
        handles=[
            Patch(facecolor=BOX_FACE, edgecolor=BLUE, label="90 instances"),
            Line2D([0], [0], color=BLUE, marker="o", markerfacecolor="white",
                   linewidth=1.35, markersize=3.8, label="Mean"),
        ],
        loc="upper left", bbox_to_anchor=(0.0, 1.22), ncol=2,
        fontsize=7.0, frameon=False, handlelength=1.5, columnspacing=1.2,
    )
    save(fig, "preference_runtime_sensitivity")


def plot_gantt():
    """Fig.10: S8-T300 triptych gantt under three preferences."""
    print("\n=== Fig.10: S8-T300 gantt triptych ===")
    df = pd.read_csv(GANTT)
    prefs = [
        (0, r"$\omega=(1,0)$"),
        (2, r"$\omega=(0.5,0.5)$"),
        (4, r"$\omega=(0,1)$"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.35), sharex=True, sharey=True)
    for ax, (pid, label) in zip(axes, prefs):
        sub = df[df["pref_id"] == pid].copy()
        for _, row in sub.iterrows():
            sat = int(row["satellite_id"])
            is_original = bool(row["original_scheduled"])
            color = BLUE if is_original else RED
            ax.broken_barh(
                [(float(row["start"]), float(row["end"]) - float(row["start"]))],
                (sat - 0.28, 0.56),
                facecolors=color,
                edgecolors=color,
                linewidth=0.22,
                zorder=3,
            )
        ax.set_ylim(-0.7, 7.7)
        ax.set_yticks(range(8))
        ax.set_yticklabels([f"S{i+1}" for i in range(8)])
        ax.set_xlim(0, 3600)
        ax.set_xlabel("Time (s)", fontsize=8.4)
        # preference label above each subplot, clear of bars
        ax.text(
            0.5, 1.015, label,
            transform=ax.transAxes, ha="center", va="bottom", fontsize=7.6,
        )
        polish(ax)
    axes[0].set_ylabel("Satellite", fontsize=8.4)

    handles = [
        Patch(facecolor=RED, label="Inserted task"),
        Patch(facecolor=BLUE, label="Original-plan task"),
    ]
    # legend below the figure, clear of subplots and labels
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=2,
        frameon=False,
        fontsize=7.5,
        handlelength=1.5,
        columnspacing=1.4,
    )
    fig.subplots_adjust(wspace=0.075, left=0.065, right=0.995, top=0.92, bottom=0.20)
    save(fig, "preference_gantt_triptych_compact")


if __name__ == "__main__":
    setup()
    plot_runtime()
    plot_gantt()
    print("\nDone.")
