from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(r"E:\论文材料\DynamicReschedule_MultiSat")
WORK = ROOT / "results" / "KDPCS_overleaf_clean_20260709_final_fig8_fig10_polished_work"
FIG = WORK / "figures"
PREVIEW = FIG / "preview_v14"
GANTT = (
    ROOT
    / "results"
    / "_tmp_gantt_candidate_search_20260709"
    / "s8_n400_inst9"
    / "preference_tradeoff_schedules.csv"
)

BLUE = "#1d3f79"
RED = "#c83a35"
GRID = "#e3e3e3"


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
    ax.grid(True, axis="x", color=GRID, linewidth=0.55, zorder=0)
    ax.tick_params(axis="both", labelsize=7.2, width=0.6, length=2.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)
        spine.set_color("#222222")


def save(fig):
    pdf = FIG / "preference_gantt_triptych_compact.pdf"
    png = PREVIEW / "preference_gantt_triptych_compact.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.006)
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.006)
    plt.close(fig)
    print(pdf)


def main():
    setup()
    df = pd.read_csv(GANTT)
    prefs = [
        (0, r"$\omega=(1,0)$"),
        (2, r"$\omega=(0.5,0.5)$"),
        (4, r"$\omega=(0,1)$"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.34), sharex=True, sharey=True)
    for ax, (pid, label) in zip(axes, prefs):
        sub = df[df["pref_id"] == pid]
        for _, row in sub.iterrows():
            is_original = bool(row["original_scheduled"])
            color = BLUE if is_original else RED
            sat = int(row["satellite_id"])
            ax.broken_barh(
                [(float(row["start"]), float(row["end"]) - float(row["start"]))],
                (sat - 0.27, 0.54),
                facecolors=color,
                edgecolors=color,
                linewidth=0.22,
                zorder=3,
            )
        ax.set_ylim(-0.7, 7.7)
        ax.set_yticks(range(8))
        ax.set_yticklabels([f"S{i+1}" for i in range(8)])
        ax.set_xlim(0, 3600)
        ax.set_xlabel("Time (s)", fontsize=8.3)
        ax.text(
            0.5,
            1.015,
            label,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
        polish(ax)
    axes[0].set_ylabel("Satellite", fontsize=8.3)

    handles = [
        Patch(facecolor=RED, label="Inserted task"),
        Patch(facecolor=BLUE, label="Original-plan task"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=2,
        frameon=False,
        fontsize=7.5,
        handlelength=1.5,
        columnspacing=1.5,
    )
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.22, top=0.83, wspace=0.075)
    save(fig)


if __name__ == "__main__":
    main()
