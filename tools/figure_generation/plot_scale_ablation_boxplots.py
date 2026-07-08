# -*- coding: utf-8 -*-
"""Publication-style boxplots for cross-scale KD-PCS ablations."""

import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "axes.linewidth": 0.55,
    "xtick.major.width": 0.45,
    "ytick.major.width": 0.45,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
})


GROUPS = ["Small", "Medium", "Large"]
VARIANT_LABELS = {
    "Full": "Full",
    "w/o exact-oracle teacher": "w/o Gurobi",
    "w/o trajectory-level labels": "w/o trajectory",
    "w/o preference conditioning": "w/o preference",
}
COLORS = {
    "Full": ("#4393C3", "#2166AC"),
    "w/o exact-oracle teacher": ("#66C2A5", "#2D8B6F"),
    "w/o trajectory-level labels": ("#FC8D62", "#D9513F"),
    "w/o preference conditioning": ("#8DA0CB", "#4A6FBD"),
}


def _read_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ["hv", "igd", "hv_gap_to_full_percent"]:
            row[key] = float(row[key])
    return rows


def _ordered_variants(rows):
    seen = []
    for row in rows:
        v = row["variant"]
        if v not in seen:
            seen.append(v)
    order = [
        "Full",
        "w/o exact-oracle teacher",
        "w/o trajectory-level labels",
        "w/o preference conditioning",
    ]
    return [v for v in order if v in seen] + [v for v in seen if v not in order]


def _values(rows, metric):
    out = defaultdict(lambda: defaultdict(list))
    for row in rows:
        out[row["scale_group"]][row["variant"]].append(row[metric])
    return out


def _draw_grouped_box(rows, metric, ylabel, output_base, include_full=True):
    variants = _ordered_variants(rows)
    if not include_full:
        variants = [v for v in variants if v != "Full"]
    data = _values(rows, metric)

    col_w = 3.5
    fig, ax = plt.subplots(figsize=(col_w, 2.75))
    fig.subplots_adjust(top=0.80)

    n_var = len(variants)
    centers = np.arange(len(GROUPS)) + 1
    width = min(0.16, 0.64 / max(n_var, 1))
    offsets = (np.arange(n_var) - (n_var - 1) / 2.0) * width * 1.15

    rng = np.random.default_rng(42)
    legend_handles = []
    for vi, variant in enumerate(variants):
        face, edge = COLORS.get(variant, ("#BDBDBD", "#636363"))
        series = [data[group].get(variant, []) for group in GROUPS]
        positions = centers + offsets[vi]
        bp = ax.boxplot(
            series,
            positions=positions,
            widths=width,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 0.9},
            whiskerprops={"linewidth": 0.55, "color": edge},
            capprops={"linewidth": 0.55, "color": edge},
            boxprops={"linewidth": 0.75, "edgecolor": edge},
        )
        for box in bp["boxes"]:
            box.set_facecolor(face)
            box.set_alpha(0.72)
        for gi, vals in enumerate(series):
            if not vals:
                continue
            jitter = rng.uniform(-width * 0.28, width * 0.28, size=len(vals))
            ax.scatter(
                np.full(len(vals), positions[gi]) + jitter,
                vals,
                s=5.0,
                color=edge,
                alpha=0.35,
                linewidths=0,
                zorder=3,
            )
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, facecolor=face, edgecolor=edge, alpha=0.72, linewidth=0.75)
        )

    if metric == "hv_gap_to_full_percent":
        ax.axhline(0.0, color=COLORS["Full"][1], linewidth=0.8, linestyle="--", zorder=0)
    ax.set_xticks(centers)
    ax.set_xticklabels(GROUPS)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.45, len(GROUPS) + 0.55)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(True, axis="y", linestyle="-", linewidth=0.3, alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    labels = [VARIANT_LABELS.get(v, v) for v in variants]
    ncol = len(labels)
    ax.legend(
        legend_handles,
        labels,
        loc="upper center",
        framealpha=0.96,
        edgecolor="#CCCCCC",
        handlelength=0.9,
        fontsize=7.5,
        ncol=ncol,
        columnspacing=0.9,
        handletextpad=0.35,
        bbox_to_anchor=(0.5, 1.16),
    )

    for ext in ["pdf", "png", "svg"]:
        fig.savefig(f"{output_base}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _write_summary(rows, output_dir):
    variants = _ordered_variants(rows)
    summary = {}
    for group in GROUPS:
        summary[group] = {}
        for variant in variants:
            vals = [r for r in rows if r["scale_group"] == group and r["variant"] == variant]
            if not vals:
                continue
            hv = np.array([r["hv"] for r in vals], dtype=float)
            igd = np.array([r["igd"] for r in vals], dtype=float)
            gap = np.array([r["hv_gap_to_full_percent"] for r in vals], dtype=float)
            summary[group][variant] = {
                "records": int(len(vals)),
                "hv_mean": float(hv.mean()),
                "hv_std": float(hv.std(ddof=1)) if len(hv) > 1 else 0.0,
                "igd_mean": float(igd.mean()),
                "igd_std": float(igd.std(ddof=1)) if len(igd) > 1 else 0.0,
                "gap_to_full_mean_percent": float(gap.mean()),
                "gap_to_full_std_percent": float(gap.std(ddof=1)) if len(gap) > 1 else 0.0,
            }
    path = os.path.join(output_dir, "scale_ablation_boxplot_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="scale_ablation")
    args = parser.parse_args()

    rows = _read_rows(args.csv)
    os.makedirs(args.output_dir, exist_ok=True)
    _draw_grouped_box(
        rows,
        "hv",
        "Hypervolume",
        os.path.join(args.output_dir, f"{args.prefix}_hv_boxplot"),
        include_full=True,
    )
    _draw_grouped_box(
        rows,
        "igd",
        "IGD",
        os.path.join(args.output_dir, f"{args.prefix}_igd_boxplot"),
        include_full=True,
    )
    _draw_grouped_box(
        rows,
        "hv_gap_to_full_percent",
        "HV gap to full method (%)",
        os.path.join(args.output_dir, f"{args.prefix}_gap_boxplot"),
        include_full=False,
    )
    summary = _write_summary(rows, args.output_dir)
    print(f"Saved figures to {args.output_dir}")
    print(f"Saved {summary}")


if __name__ == "__main__":
    main()
