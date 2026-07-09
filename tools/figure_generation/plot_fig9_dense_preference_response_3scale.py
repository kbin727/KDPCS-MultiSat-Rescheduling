# -*- coding: utf-8 -*-
"""Draw Fig. 9 dense preference-response panels for three benchmark scales."""

import argparse
from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams


BLUE = "#153E6F"
RED = "#B33A3A"
GRID = "#D9DDE3"


def configure_style():
    rcParams["font.family"] = "Times New Roman"
    rcParams["mathtext.fontset"] = "stix"
    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"] = 42
    rcParams["axes.linewidth"] = 0.78
    rcParams["axes.edgecolor"] = "#222222"
    rcParams["axes.grid"] = True
    rcParams["grid.color"] = GRID
    rcParams["grid.linewidth"] = 0.40
    rcParams["grid.alpha"] = 0.70
    rcParams["axes.axisbelow"] = True
    rcParams["xtick.direction"] = "out"
    rcParams["ytick.direction"] = "out"
    rcParams["xtick.major.width"] = 0.75
    rcParams["ytick.major.width"] = 0.75
    rcParams["xtick.major.size"] = 3.0
    rcParams["ytick.major.size"] = 3.0


def draw_panel(csv_path: Path, out_dir: Path, name: str, scale_label: str, ylabel: bool = False, legend: bool = False):
    configure_style()
    df = pd.read_csv(csv_path).sort_values("w2")
    fig, ax = plt.subplots(figsize=(3.14, 1.88))

    ax.plot(
        df["w2"],
        df["f1"],
        color=BLUE,
        lw=1.32,
        marker="o",
        markersize=2.25,
        markerfacecolor="white",
        markeredgewidth=0.82,
        label=r"$f_1$",
    )
    ax.plot(
        df["w2"],
        df["f2"],
        color=RED,
        lw=1.32,
        marker="s",
        markersize=2.15,
        markerfacecolor="white",
        markeredgewidth=0.82,
        label=r"$f_2$",
    )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.04)
    ax.set_xticks([0, 0.25, 0.50, 0.75, 1.00])
    ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1.00"], fontsize=7.7)
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
    ax.tick_params(axis="y", labelsize=7.7, pad=1.2)
    ax.tick_params(axis="x", pad=1.2)
    ax.set_xlabel(r"Preference weight $\omega_2$", fontsize=8.4, labelpad=1.2)
    if ylabel:
        ax.set_ylabel("Objective value", fontsize=8.4, labelpad=1.2)
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])

    ax.text(
        0.035,
        0.93,
        scale_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.6,
        color="#222222",
        bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.82),
        zorder=10,
    )

    ax.legend(
        loc="center right",
        fontsize=6.7,
        frameon=True,
        framealpha=0.95,
        edgecolor="#B8B8B8",
        borderpad=0.20,
        handlelength=1.18,
        labelspacing=0.16,
    )
    fig.tight_layout(pad=0.04)

    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = out_dir / "preview_final"
    preview_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{name}.pdf"
    png = preview_dir / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(png, dpi=360, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return pdf, png


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", required=True)
    parser.add_argument("--medium", required=True)
    parser.add_argument("--large", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out = Path(args.out_dir)
    for path, name, scale_label, ylabel, legend in [
        (args.small, "preference_dense_tradeoff_s5_t150", "S5-T150", True, False),
        (args.medium, "preference_dense_tradeoff_s10_t300", "S10-T300", False, False),
        (args.large, "preference_dense_tradeoff_s20_t600", "S20-T600", False, True),
    ]:
        pdf, png = draw_panel(Path(path), out, name, scale_label, ylabel=ylabel, legend=legend)
        print(pdf)
        print(png)


if __name__ == "__main__":
    main()
