# -*- coding: utf-8 -*-
"""Draw Fig. 8 preference-count sensitivity with HV, IGD, and runtime."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D


K_VALUES = [20, 50, 100, 150, 200]


def configure_style():
    rcParams["font.family"] = "Times New Roman"
    rcParams["mathtext.fontset"] = "stix"
    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"] = 42
    rcParams["axes.linewidth"] = 0.85
    rcParams["axes.edgecolor"] = "#222222"
    rcParams["axes.grid"] = True
    rcParams["grid.color"] = "#D8D8D8"
    rcParams["grid.linewidth"] = 0.45
    rcParams["grid.alpha"] = 0.75
    rcParams["axes.axisbelow"] = True
    rcParams["xtick.direction"] = "out"
    rcParams["ytick.direction"] = "out"
    rcParams["xtick.major.width"] = 0.75
    rcParams["ytick.major.width"] = 0.75
    rcParams["xtick.major.size"] = 3.0
    rcParams["ytick.major.size"] = 3.0


def save(fig, out_dir, name):
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{name}.pdf"
    png = out_dir / "preview_final" / f"{name}.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(png, dpi=360, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return pdf, png


def draw_boxline(rows, metric, ylabel, out_dir, name, ylim=None, log_y=False):
    configure_style()
    fig, ax = plt.subplots(figsize=(3.25, 2.08))
    values = [rows.loc[rows["num_prefs"] == k, metric].to_numpy(dtype=float) for k in K_VALUES]
    pos = np.arange(len(K_VALUES))

    ax.boxplot(
        values,
        positions=pos,
        widths=0.50,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="#1B365D", lw=1.20),
        whiskerprops=dict(color="#4A4A4A", lw=0.78),
        capprops=dict(color="#4A4A4A", lw=0.78),
        boxprops=dict(facecolor="#9DB8D8", edgecolor="#3F5F86", lw=0.78, alpha=0.58),
    )

    rng = np.random.RandomState(17)
    for idx, vals in enumerate(values):
        jitter = rng.uniform(-0.135, 0.135, len(vals))
        ax.scatter(
            np.full(len(vals), idx) + jitter,
            vals,
            s=7.0,
            color="#2F5E8E",
            alpha=0.22,
            linewidths=0,
            zorder=3,
        )

    means = [float(np.mean(v)) for v in values]
    ax.plot(
        pos,
        means,
        color="#123B6D",
        lw=1.55,
        marker="o",
        markersize=4.3,
        markerfacecolor="white",
        markeredgecolor="#123B6D",
        markeredgewidth=1.1,
        zorder=5,
    )
    ax.axvline(1, color="#8B8B8B", lw=0.75, linestyle=(0, (4, 3)), alpha=0.75, zorder=1)

    ax.set_xticks(pos)
    ax.set_xticklabels([str(k) for k in K_VALUES], fontsize=8.4)
    ax.set_xlabel(r"Preference count $K$", fontsize=9.2, labelpad=1.5)
    ax.set_ylabel(ylabel, fontsize=9.2, labelpad=1.5)
    ax.tick_params(axis="y", labelsize=8.2, pad=1.5)
    ax.tick_params(axis="x", pad=1.5)
    ax.set_xlim(-0.55, len(K_VALUES) - 0.45)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if log_y:
        ax.set_yscale("log")
        ax.grid(which="minor", linestyle=":", linewidth=0.35, alpha=0.5)

    handles = [
        Line2D([0], [0], color="#123B6D", lw=1.55, marker="o", markersize=4.3,
               markerfacecolor="white", markeredgecolor="#123B6D", label="Mean"),
        Line2D([0], [0], color="#8B8B8B", lw=0.75, linestyle=(0, (4, 3)), label=r"$K=50$"),
    ]
    ax.legend(
        handles=handles,
        loc="best",
        fontsize=7.2,
        frameon=True,
        framealpha=0.94,
        edgecolor="#BBBBBB",
        borderpad=0.25,
        handlelength=1.6,
        labelspacing=0.25,
    )
    fig.tight_layout(pad=0.08)
    return save(fig, out_dir, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    rows = pd.read_csv(args.rows)
    if len(rows) != 450:
        raise ValueError(f"Expected 450 sensitivity rows, got {len(rows)}")
    out_dir = Path(args.out_dir)

    draw_boxline(
        rows,
        "normalized_hv",
        r"HV retention (\%)",
        out_dir,
        "preference_hv_sensitivity",
        ylim=(96.8, 100.35),
    )
    draw_boxline(
        rows,
        "igd_to_k200",
        r"IGD to $K=200$",
        out_dir,
        "preference_igd_sensitivity",
        ylim=(-0.0007, max(0.023, rows["igd_to_k200"].max() * 1.08)),
    )
    draw_boxline(
        rows,
        "seconds",
        r"Runtime (s)",
        out_dir,
        "preference_runtime_sensitivity",
        log_y=True,
    )

    summary = rows.groupby("num_prefs").agg(
        normalized_hv_mean=("normalized_hv", "mean"),
        igd_mean=("igd_to_k200", "mean"),
        runtime_mean=("seconds", "mean"),
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
