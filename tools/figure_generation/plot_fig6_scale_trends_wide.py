# -*- coding: utf-8 -*-
"""Draw wide Fig. 6 scale-trend panels for the KDPCS paper."""

import argparse
from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams


METHOD_ORDER = ["pcformer", "monp", "nsga2", "spea2", "ibea", "moead"]
LABELS = {
    "pcformer": "KDPCS",
    "monp": "MONP",
    "nsga2": "NSGA-II",
    "spea2": "SPEA2",
    "ibea": "IBEA",
    "moead": "MOEA/D",
}
COLORS = {
    "pcformer": "#143E6E",
    "monp": "#6E6E6E",
    "nsga2": "#2D7F5E",
    "spea2": "#7B5EA7",
    "ibea": "#D07A2D",
    "moead": "#B99A2F",
}
MARKERS = {
    "pcformer": "o",
    "monp": "s",
    "nsga2": "^",
    "spea2": "D",
    "ibea": "v",
    "moead": "P",
}
SCALE_ORDER = [
    "s3_n100", "s3_n150", "s3_n200",
    "s5_n100", "s5_n150", "s5_n200",
    "s8_n200", "s8_n300", "s8_n400",
    "s10_n200", "s10_n300", "s10_n400",
    "s15_n400", "s15_n600", "s15_n800",
    "s20_n400", "s20_n600", "s20_n800",
]


def scale_label(scale: str) -> str:
    s_part, n_part = scale.split("_")
    return f"S{s_part[1:]}-T{n_part[1:]}"


def configure():
    rcParams["font.family"] = "Times New Roman"
    rcParams["mathtext.fontset"] = "stix"
    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"] = 42
    rcParams["axes.linewidth"] = 0.72
    rcParams["axes.edgecolor"] = "#222222"
    rcParams["axes.grid"] = True
    rcParams["grid.color"] = "#DADDE3"
    rcParams["grid.linewidth"] = 0.40
    rcParams["grid.alpha"] = 0.72
    rcParams["axes.axisbelow"] = True
    rcParams["xtick.direction"] = "out"
    rcParams["ytick.direction"] = "out"
    rcParams["xtick.major.width"] = 0.72
    rcParams["ytick.major.width"] = 0.72
    rcParams["xtick.major.size"] = 2.7
    rcParams["ytick.major.size"] = 2.7


def draw_metric(df: pd.DataFrame, metric: str, ylabel: str, out_pdf: Path, log_y: bool = False):
    configure()
    fig, ax = plt.subplots(figsize=(3.62, 2.18))
    x = list(range(len(SCALE_ORDER)))
    for method in METHOD_ORDER:
        sub = (
            df[df["method"] == method]
            .set_index("scale")
            .reindex(SCALE_ORDER)
        )
        lw = 1.58 if method == "pcformer" else 0.92
        ms = 3.75 if method == "pcformer" else 3.05
        ax.plot(
            x,
            sub[metric].values,
            color=COLORS[method],
            marker=MARKERS[method],
            lw=lw,
            markersize=ms,
            markerfacecolor=COLORS[method] if method == "pcformer" else "white",
            markeredgewidth=0.82,
            label=LABELS[method],
            zorder=6 if method == "pcformer" else 4,
        )

    ax.set_xlim(-0.45, len(SCALE_ORDER) - 0.55)
    ax.set_xticks(x)
    ax.set_xticklabels([scale_label(s) for s in SCALE_ORDER], rotation=45, ha="right", fontsize=5.8)
    ax.tick_params(axis="y", labelsize=6.9, pad=1.0)
    ax.tick_params(axis="x", pad=0.6)
    ax.set_ylabel(ylabel, fontsize=7.6, labelpad=1.2)
    if log_y:
        ax.set_yscale("log")
    legend = ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.055, 1.0, 0.075),
        mode="expand",
        ncol=6,
        fontsize=4.95,
        frameon=True,
        framealpha=0.96,
        edgecolor="#C9C9C9",
        handlelength=0.68,
        columnspacing=0.12,
        borderpad=0.12,
        labelspacing=0.08,
        borderaxespad=0.0,
    )
    for text in legend.get_texts():
        if text.get_text() == "KDPCS":
            text.set_fontweight("bold")
    fig.tight_layout(pad=0.10)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.01)
    preview = out_pdf.parent / "preview_final" / out_pdf.name.replace(".pdf", ".png")
    preview.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(preview, dpi=360, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return out_pdf, preview


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.summary)
    out = Path(args.out_dir)
    jobs = [
        ("hv", "HV", out / "scale_trend_hv_20260705.pdf", False),
        ("igd", "IGD", out / "scale_trend_igd_20260705.pdf", False),
        ("seconds", r"Runtime $T$ (s)", out / "scale_trend_runtime_20260705.pdf", True),
    ]
    for metric, ylabel, path, log_y in jobs:
        pdf, png = draw_metric(df, metric, ylabel, path, log_y=log_y)
        print(pdf)
        print(png)


if __name__ == "__main__":
    main()
