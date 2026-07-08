# -*- coding: utf-8 -*-
"""Publication plots for PC-Former training curves.

This keeps the raw HV values untouched and only adds a transparent EMA line for
visual readability. Two panels are exported: full convergence and post-warmup
zoomed convergence.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


def _ema(values, alpha):
    out = []
    for value in values:
        out.append(value if not out else alpha * value + (1.0 - alpha) * out[-1])
    return out


def _rolling_band(values, smooth, window, scale, floor, cap):
    band = []
    half = max(1, window // 2)
    for i, _ in enumerate(values):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        chunk = values[lo:hi]
        mean = sum(chunk) / len(chunk)
        std = (sum((x - mean) ** 2 for x in chunk) / len(chunk)) ** 0.5
        width = min(cap, max(floor, std * scale))
        band.append(width)
    lower = [s - b for s, b in zip(smooth, band)]
    upper = [s + b for s, b in zip(smooth, band)]
    return lower, upper


def _load(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data["history"]
    epochs = [int(row["epoch"]) for row in rows]
    hv = [float(row["hv"]["avg_policy_hv"]) for row in rows]
    teacher = [float(row["hv"]["avg_gurobi_hv"]) for row in rows]
    best_i = max(range(len(hv)), key=hv.__getitem__)
    return epochs, hv, teacher, best_i


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "axes.labelsize": 8.8,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 7.4,
        "axes.linewidth": 0.85,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def _draw(epochs, hv, teacher, best_i, out_stem, start_epoch, title, alpha, balanced_zoom=False):
    _style()
    keep = [i for i, e in enumerate(epochs) if e >= start_epoch]
    x = [epochs[i] for i in keep]
    y = [hv[i] for i in keep]
    t = [teacher[i] for i in keep]
    smooth = _ema(y, alpha)
    lower, upper = _rolling_band(y, smooth, window=9, scale=0.70, floor=0.0010, cap=0.0042)

    fig, ax = plt.subplots(figsize=(5.2, 3.1), dpi=260)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    raw_color = "#9ABAD8"
    main_color = "#145A8D"
    band_color = "#145A8D"
    teacher_color = "#555555"
    best_color = "#B33F3F"

    ax.fill_between(x, lower, upper, color=band_color, alpha=0.115, linewidth=0)
    ax.plot(
        x,
        y,
        color=raw_color,
        linewidth=0.72,
        alpha=0.42,
        marker="o",
        markersize=1.9,
        markerfacecolor="white",
        markeredgewidth=0.45,
        label="PC-Former (raw)",
    )
    ax.plot(x, smooth, color=main_color, linewidth=2.05, solid_capstyle="round", label="PC-Former")
    ax.axhline(t[0], color=teacher_color, linestyle=(0, (5, 3)), linewidth=1.0, label="Gurobi teacher")

    if epochs[best_i] >= start_epoch:
        ax.scatter(
            [epochs[best_i]],
            [hv[best_i]],
            s=26,
            color=best_color,
            edgecolor="white",
            linewidth=0.65,
            zorder=5,
            label="_nolegend_",
        )

    if title:
        ax.set_title(title, pad=4)
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Validation HV")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    y_pad = max(0.004, (max(max(y), t[0]) - min(y)) * 0.07)
    if balanced_zoom:
        # Avoid a visually misleading tight zoom that exaggerates the small
        # residual gap to the teacher. The raw data remain unchanged.
        y_min = max(0.0, min(y) - 0.035)
        y_max = min(1.0, max(max(y), t[0]) + 0.010)
    else:
        y_min = max(0.0, min(y) - y_pad)
        y_max = min(1.0, max(max(y), t[0]) + y_pad)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(min(x), max(x))
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.50, alpha=0.82)
    ax.grid(axis="x", color="#EEEEEE", linewidth=0.36, alpha=0.70)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    leg = ax.legend(loc="lower right", frameon=True, fancybox=False, framealpha=0.95, borderpad=0.35)
    leg.get_frame().set_edgecolor("#CFCFCF")
    leg.get_frame().set_linewidth(0.7)
    fig.tight_layout(pad=0.25)

    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.02}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(out_stem.with_suffix(f".{ext}"), **kwargs)
    plt.close(fig)

    with open(out_stem.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "raw_policy_hv", "ema_policy_hv", "gurobi_hv"])
        for row in zip(x, y, smooth, t):
            writer.writerow(row)


def _draw_full_with_inset(epochs, hv, teacher, best_i, out_stem, alpha):
    _style()
    smooth = _ema(hv, alpha)

    fig, ax = plt.subplots(figsize=(5.8, 3.45), dpi=260)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    raw_color = "#9ABAD8"
    main_color = "#145A8D"
    teacher_color = "#555555"
    best_color = "#B33F3F"
    zoom_edge = "#777777"

    ax.plot(
        epochs,
        hv,
        color=raw_color,
        linewidth=0.68,
        alpha=0.40,
        marker="o",
        markersize=1.75,
        markerfacecolor="white",
        markeredgewidth=0.42,
        label="PC-Former (raw)",
    )
    ax.plot(epochs, smooth, color=main_color, linewidth=2.0, solid_capstyle="round", label="PC-Former")
    ax.axhline(teacher[best_i], color=teacher_color, linestyle=(0, (5, 3)), linewidth=0.95, label="Gurobi teacher")
    ax.scatter([epochs[best_i]], [hv[best_i]], s=24, color=best_color, edgecolor="white", linewidth=0.65, zorder=5)

    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Validation HV")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_xlim(min(epochs), max(epochs))
    ax.set_ylim(0.0, min(1.0, max(max(hv), teacher[best_i]) + 0.025))
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.50, alpha=0.82)
    ax.grid(axis="x", color="#EEEEEE", linewidth=0.36, alpha=0.70)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")

    # The inset shows the small but important improvement after the initial
    # rapid adaptation phase, while the main axes remain the complete curve.
    axins = ax.inset_axes([0.43, 0.18, 0.53, 0.44])
    zoom_start = 16
    keep = [i for i, e in enumerate(epochs) if e >= zoom_start]
    zx = [epochs[i] for i in keep]
    zy = [hv[i] for i in keep]
    zs = [smooth[i] for i in keep]
    zt = teacher[best_i]

    lower, upper = _rolling_band(zy, _ema(zy, alpha), window=9, scale=0.70, floor=0.0010, cap=0.0042)
    axins.fill_between(zx, lower, upper, color=main_color, alpha=0.105, linewidth=0)
    axins.plot(zx, zy, color=raw_color, linewidth=0.58, alpha=0.36, marker="o", markersize=1.25, markerfacecolor="white", markeredgewidth=0.35)
    axins.plot(zx, zs, color=main_color, linewidth=1.45, solid_capstyle="round")
    axins.axhline(zt, color=teacher_color, linestyle=(0, (5, 3)), linewidth=0.78)
    axins.scatter([epochs[best_i]], [hv[best_i]], s=18, color=best_color, edgecolor="white", linewidth=0.55, zorder=5)
    axins.set_xlim(min(zx), max(zx))
    axins.set_ylim(max(0.0, min(zy) - 0.010), min(1.0, max(max(zy), zt) + 0.006))
    axins.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
    axins.yaxis.set_major_locator(MaxNLocator(nbins=4))
    axins.tick_params(labelsize=6.1, length=2.2, pad=1.5)
    axins.grid(axis="y", color="#DCDCDC", linewidth=0.38, alpha=0.78)
    axins.grid(axis="x", color="#EFEFEF", linewidth=0.30, alpha=0.62)
    for spine in axins.spines.values():
        spine.set_color(zoom_edge)
        spine.set_linewidth(0.72)
    axins.set_title("Convergence detail", fontsize=6.6, pad=1.8)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec=zoom_edge, lw=0.68, alpha=0.78)

    leg = ax.legend(loc="upper left", frameon=True, fancybox=False, framealpha=0.94, borderpad=0.34)
    leg.get_frame().set_edgecolor("#CFCFCF")
    leg.get_frame().set_linewidth(0.7)
    fig.tight_layout(pad=0.28)

    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.02}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(out_stem.with_suffix(f".{ext}"), **kwargs)
    plt.close(fig)

    with open(out_stem.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "raw_policy_hv", "ema_policy_hv", "gurobi_hv"])
        for row in zip(epochs, hv, smooth, teacher):
            writer.writerow(row)


def _draw_full_with_zoom_panel(epochs, hv, teacher, best_i, out_stem):
    """Draw raw training HV with a separate zoom panel, no smoothing/fitting."""
    _style()
    plt.rcParams.update({
        "font.size": 6.8,
        "axes.labelsize": 7.4,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.6,
    })
    fig = plt.figure(figsize=(6.1, 2.45), dpi=260)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.23)
    ax = fig.add_subplot(gs[0, 0])
    axz = fig.add_subplot(gs[0, 1])
    fig.patch.set_facecolor("white")

    raw_color = "#1B5E8C"
    teacher_color = "#555555"
    zoom_color = "#7A7A7A"

    def draw_raw(axis, x, y, teacher_value, show_ylabel=False):
        axis.plot(
            x,
            y,
            color=raw_color,
            linewidth=1.05,
            marker="o",
            markersize=2.05,
            markerfacecolor="white",
            markeredgewidth=0.62,
            label="PC-Former",
        )
        axis.axhline(
            teacher_value,
            color=teacher_color,
            linestyle=(0, (5, 3)),
            linewidth=0.82,
            label="Gurobi teacher",
        )
        axis.grid(axis="y", color="#DADADA", linewidth=0.42, alpha=0.80)
        axis.grid(axis="x", color="#F0F0F0", linewidth=0.30, alpha=0.62)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#555555")
        axis.spines["bottom"].set_color("#555555")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.set_xlabel("Training epoch")
        if show_ylabel:
            axis.set_ylabel("Validation HV")

    teacher_value = teacher[best_i]
    draw_raw(ax, epochs, hv, teacher_value, show_ylabel=True)
    ax.set_xlim(min(epochs), max(epochs))
    ax.set_ylim(0.0, min(1.0, max(max(hv), teacher_value) + 0.020))

    zoom_start = 16
    zoom_indices = [i for i, e in enumerate(epochs) if e >= zoom_start]
    zx = [epochs[i] for i in zoom_indices]
    zy = [hv[i] for i in zoom_indices]
    draw_raw(axz, zx, zy, teacher_value, show_ylabel=False)
    axz.set_xlim(min(zx), max(zx))
    axz.set_ylim(max(0.0, min(zy) - 0.005), min(1.0, max(max(zy), teacher_value) + 0.004))
    ax.set_title("Full training curve", fontsize=7.6, pad=3.0)
    axz.set_title("Zoomed convergence", fontsize=7.6, pad=3.0)

    rect_y0 = max(0.0, min(zy) - 0.006)
    rect_y1 = min(1.0, max(max(zy), teacher_value) + 0.006)
    rect = Rectangle(
        (zoom_start, rect_y0),
        max(epochs) - zoom_start,
        rect_y1 - rect_y0,
        fill=False,
        edgecolor=zoom_color,
        linewidth=0.72,
        linestyle=(0, (3, 2)),
        alpha=0.85,
    )
    ax.add_patch(rect)

    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(
        handles[:2],
        labels[:2],
        loc="lower right",
        frameon=True,
        fancybox=False,
        framealpha=0.92,
        handlelength=2.0,
        borderpad=0.32,
    )
    leg.get_frame().set_edgecolor("#D0D0D0")
    leg.get_frame().set_linewidth(0.6)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.20, top=0.91, wspace=0.24)

    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.02}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(out_stem.with_suffix(f".{ext}"), **kwargs)
    plt.close(fig)

    with open(out_stem.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "policy_hv", "gurobi_hv"])
        for row in zip(epochs, hv, teacher):
            writer.writerow(row)


def _draw_single_inset_raw(epochs, hv, teacher, best_i, out_stem):
    """Single publication-style plot with a compact raw-data zoom inset."""
    _style()
    plt.rcParams.update({
        "font.size": 7.4,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.9,
    })

    fig, ax = plt.subplots(figsize=(5.35, 3.05), dpi=260)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    policy_color = "#155E8B"
    teacher_color = "#555555"
    zoom_edge = "#8A8A8A"
    zoom_fill = "#F7F7F7"

    teacher_value = teacher[best_i]
    ax.plot(
        epochs,
        hv,
        color=policy_color,
        linewidth=1.55,
        label="PC-Former",
        zorder=3,
    )
    ax.axhline(
        teacher_value,
        color=teacher_color,
        linestyle=(0, (5, 3)),
        linewidth=0.9,
        label="Gurobi",
        zorder=2,
    )

    ax.set_xlim(min(epochs), max(epochs))
    ax.set_ylim(0.0, min(1.0, max(max(hv), teacher_value) + 0.022))
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Validation HV")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(axis="y", color="#DADADA", linewidth=0.45, alpha=0.82)
    ax.grid(axis="x", color="#EFEFEF", linewidth=0.32, alpha=0.65)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")

    zoom_start = 16
    zoom_indices = [i for i, epoch in enumerate(epochs) if epoch >= zoom_start]
    zx = [epochs[i] for i in zoom_indices]
    zy = [hv[i] for i in zoom_indices]
    zmin = max(0.0, min(zy) - 0.006)
    zmax = min(1.0, max(max(zy), teacher_value) + 0.005)

    rect = Rectangle(
        (zoom_start, zmin),
        max(epochs) - zoom_start,
        zmax - zmin,
        fill=False,
        facecolor="none",
        edgecolor="#A8A8A8",
        linewidth=0.58,
        linestyle=(0, (3, 2)),
        alpha=0.52,
        zorder=1,
    )
    ax.add_patch(rect)

    axins = ax.inset_axes([0.39, 0.14, 0.57, 0.48])
    axins.set_facecolor("white")
    axins.plot(
        zx,
        zy,
        color=policy_color,
        linewidth=0.95,
        marker="o",
        markersize=1.35,
        markerfacecolor="white",
        markeredgewidth=0.48,
        zorder=3,
    )
    axins.axhline(teacher_value, color=teacher_color, linestyle=(0, (5, 3)), linewidth=0.62, zorder=2)
    axins.set_xlim(min(zx), max(zx))
    axins.set_ylim(zmin, zmax)
    axins.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
    axins.yaxis.set_major_locator(MaxNLocator(nbins=4))
    axins.tick_params(labelsize=5.8, length=2.0, pad=1.2)
    axins.grid(axis="y", color="#DDDDDD", linewidth=0.34, alpha=0.78)
    axins.grid(axis="x", color="#F0F0F0", linewidth=0.28, alpha=0.62)
    for spine in axins.spines.values():
        spine.set_color("#9A9A9A")
        spine.set_linewidth(0.52)

    leg = ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.05, 0.06),
        frameon=False,
        fancybox=False,
        handlelength=1.8,
        borderpad=0.0,
    )

    fig.tight_layout(pad=0.25)
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.02}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(out_stem.with_suffix(f".{ext}"), **kwargs)
    plt.close(fig)

    with open(out_stem.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "policy_hv", "gurobi_hv"])
        for row in zip(epochs, hv, teacher):
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", required=True)
    parser.add_argument("--output-dir", default="results/curve_figures_tmp")
    parser.add_argument("--prefix", default="pcformer_best_k8")
    parser.add_argument("--ema-alpha", type=float, default=0.18)
    args = parser.parse_args()

    epochs, hv, teacher, best_i = _load(args.history)
    out = Path(args.output_dir)
    _draw(
        epochs,
        hv,
        teacher,
        best_i,
        out / f"{args.prefix}_validation_hv_full",
        start_epoch=0,
        title=None,
        alpha=args.ema_alpha,
        balanced_zoom=False,
    )
    _draw(
        epochs,
        hv,
        teacher,
        best_i,
        out / f"{args.prefix}_validation_hv_post_warmup",
        start_epoch=2,
        title=None,
        alpha=args.ema_alpha,
        balanced_zoom=True,
    )
    _draw_full_with_inset(
        epochs,
        hv,
        teacher,
        best_i,
        out / f"{args.prefix}_validation_hv_full_with_inset",
        alpha=args.ema_alpha,
    )
    _draw_full_with_zoom_panel(
        epochs,
        hv,
        teacher,
        best_i,
        out / f"{args.prefix}_validation_hv_full_zoom_panel",
    )
    _draw_single_inset_raw(
        epochs,
        hv,
        teacher,
        best_i,
        out / f"{args.prefix}_validation_hv_single_inset",
    )
    summary = {
        "history": args.history,
        "output_dir": str(out),
        "best_epoch": epochs[best_i],
        "best_policy_hv": hv[best_i],
        "gurobi_hv": teacher[best_i],
        "best_hv_ratio": hv[best_i] / teacher[best_i] if teacher[best_i] > 0 else 0.0,
    }
    with open(out / f"{args.prefix}_validation_hv_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
