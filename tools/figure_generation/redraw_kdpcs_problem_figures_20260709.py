import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


ROOT = r"E:\论文材料\DynamicReschedule_MultiSat"
WORK = os.path.join(ROOT, "results", "KDPCS_overleaf_clean_20260709_polished_work")
FIG = os.path.join(WORK, "figures")
PREVIEW = os.path.join(FIG, "preview_v14")
os.makedirs(PREVIEW, exist_ok=True)

PREF_SUMMARY = os.path.join(
    ROOT, "results", "server_preference_sensitivity_20260708",
    "preference_count_sensitivity", "preference_count_sensitivity_summary.csv")
PREF_ROWS = os.path.join(
    ROOT, "results", "server_preference_sensitivity_20260708",
    "preference_count_sensitivity", "preference_count_sensitivity_rows.csv")
DENSE_S5 = os.path.join(
    ROOT, "results", "server_preference_sensitivity_20260708",
    "tradeoff_dense_s5_n150_inst0_p51", "preference_tradeoff_objectives.csv")
DENSE_S10 = os.path.join(
    ROOT, "results", "server_preference_sensitivity_20260708",
    "tradeoff_dense_s10_n300_inst0_p51", "preference_tradeoff_objectives.csv")
GANTT = os.path.join(
    ROOT, "results", "server_preference_sensitivity_20260708",
    "tradeoff_gantt_s5_n150_inst0", "preference_tradeoff_schedules.csv")
COMPLEX = os.path.join(
    ROOT, "results", "final_kdpcs_all18_merged_20260705",
    "complexity_validation", "kdpcs_complexity_points.csv")

BLUE = "#163a76"
RED = "#bf3b36"
GREEN = "#4f9462"
GRAY = "#666666"
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
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.75,
        "axes.edgecolor": "#222222",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def polish(ax, grid_axis="both"):
    ax.grid(True, axis=grid_axis, color=LIGHT_GRID, linewidth=0.55, alpha=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7.5, width=0.7, length=3)
    for sp in ax.spines.values():
        sp.set_linewidth(0.75)


def save(fig, name):
    pdf = os.path.join(FIG, name + ".pdf")
    png = os.path.join(PREVIEW, name + ".png")
    fig.savefig(pdf, format="pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(png, format="png", dpi=320, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return pdf


def plot_pref_sensitivity():
    rows = pd.read_csv(PREF_ROWS)
    rows["scale_group"] = rows["scale_group"].astype(str).str.lower().str.strip()
    rows = rows[rows["scale_group"] == "all"].copy()
    if rows.empty:
        raw = pd.read_csv(PREF_ROWS)
        rows = raw.copy()
    summary = pd.read_csv(PREF_SUMMARY)
    summary["scale_group"] = summary["scale_group"].astype(str).str.lower().str.strip()
    summary = summary[summary["scale_group"] == "all"].sort_values("num_prefs")
    ks = [20, 50, 100, 150, 200]
    x = np.arange(len(ks))

    def boxline(data_by_k, ylabel, name, ylim_pad=0.12, ref_text=None, ylim=None):
        fig, ax = plt.subplots(figsize=(3.55, 1.78))
        data = [np.asarray(data_by_k[k], dtype=float) for k in ks]
        bp = ax.boxplot(
            data, positions=x, widths=0.48, patch_artist=True, showfliers=False,
            medianprops={"color": "#1a1a1a", "linewidth": 0.8},
            boxprops={"linewidth": 0.75, "color": BLUE},
            whiskerprops={"linewidth": 0.65, "color": BLUE},
            capprops={"linewidth": 0.65, "color": BLUE},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("#dbe5f2")
            patch.set_alpha(0.82)
        means = [np.mean(d) for d in data]
        ax.plot(x, means, color=BLUE, linewidth=1.35, marker="o", markersize=3.8,
                markerfacecolor="white", markeredgewidth=1.0, zorder=5)
        ax.axvline(1, color="#999999", linewidth=0.7, linestyle=(0, (3, 3)), zorder=1)
        if ref_text:
            ax.text(1.03, 0.06, ref_text, transform=ax.get_xaxis_transform(),
                    fontsize=6.8, color="#555555", va="bottom")
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel(r"Preference count $K$", fontsize=8.8)
        ax.set_ylabel(ylabel, fontsize=8.8)
        y_all = np.concatenate(data)
        if ylim is None:
            yr = y_all.max() - y_all.min()
            ax.set_ylim(y_all.min() - ylim_pad * yr, y_all.max() + ylim_pad * yr)
        else:
            ax.set_ylim(*ylim)
        polish(ax)
        ax.legend(
            handles=[
                Patch(facecolor="#dbe5f2", edgecolor=BLUE, label="90 instances"),
                Line2D([0], [0], color=BLUE, marker="o", markerfacecolor="white",
                       linewidth=1.35, markersize=3.8, label="Mean")
            ],
            loc="upper left", bbox_to_anchor=(0.0, 1.22), ncol=2,
            fontsize=7.0, frameon=False, handlelength=1.5, columnspacing=1.2,
        )
        return save(fig, name)

    key_cols = ["scale", "instance_idx"]
    base = rows[rows["num_prefs"] == 200][key_cols + ["hv"]].rename(columns={"hv": "hv_200"})
    ratio_rows = rows.merge(base, on=key_cols, how="left")
    ratio_rows["hv_ratio"] = 100.0 * ratio_rows["hv"] / ratio_rows["hv_200"]
    hv_data = {k: ratio_rows[ratio_rows["num_prefs"] == k]["hv_ratio"].values for k in ks}
    rt_data = {k: rows[rows["num_prefs"] == k]["seconds"].values for k in ks}

    boxline(hv_data, r"HV retention (\%)", "preference_hv_sensitivity",
            ylim=(96.0, 100.6), ref_text=r"$K=50$")
    boxline(rt_data, "Runtime (s)", "preference_runtime_sensitivity",
            ylim_pad=0.10, ref_text=r"$K=50$")


def plot_dense_response():
    s5 = pd.read_csv(DENSE_S5)
    s10 = pd.read_csv(DENSE_S10)
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 1.95), sharex=True)
    for ax, df, lab in [(axes[0], s5, "S5-T150"), (axes[1], s10, "S10-T300")]:
        df = df.sort_values("w2")
        ax.plot(df["w2"], df["f1"], color=RED, lw=1.25, marker="o",
                markersize=2.8, markerfacecolor="white", markeredgewidth=0.8,
                label=r"$f_1$")
        ax.plot(df["w2"], df["f2"], color=BLUE, lw=1.25, marker="s",
                markersize=2.6, markerfacecolor="white", markeredgewidth=0.8,
                label=r"$f_2$")
        ax.text(0.02, 0.93, lab, transform=ax.transAxes, fontsize=7.6,
                va="top", ha="left",
                bbox=dict(facecolor="white", edgecolor="#cccccc", linewidth=0.45, alpha=0.9, pad=1.5))
        ax.set_xlabel(r"Preference weight $\omega_2$", fontsize=8.8)
        ax.set_ylabel("Objective value", fontsize=8.8)
        ax.set_xlim(-0.02, 1.02)
        polish(ax)
    axes[1].legend(loc="lower right", fontsize=7.0, frameon=True,
                   framealpha=0.92, borderpad=0.25, handlelength=1.3)
    fig.subplots_adjust(wspace=0.20, left=0.075, right=0.995, top=0.98, bottom=0.25)
    save(fig, "preference_dense_tradeoff_response")


def plot_gantt_triptych():
    df = pd.read_csv(GANTT)
    prefs = [(0, r"$\omega=(1,0)$"), (2, r"$\omega=(0.5,0.5)$"), (4, r"$\omega=(0,1)$")]
    colors = {"emergency": "#c83a35", "original": "#1d3f79"}

    fig, axes = plt.subplots(1, 3, figsize=(7.05, 1.95), sharex=True, sharey=True)
    for ax, (pid, label) in zip(axes, prefs):
        sub = df[(df["pref_id"] == pid) & (df["is_emergency"] | df["original_scheduled"])].copy()
        for _, r in sub.iterrows():
            sat = int(r["satellite_id"])
            y = sat
            if bool(r["is_emergency"]):
                c = colors["emergency"]
            elif bool(r["original_scheduled"]):
                c = colors["original"]
            else:
                continue
            ax.broken_barh([(float(r["start"]), float(r["end"]) - float(r["start"]))],
                           (y - 0.28, 0.56), facecolors=c, edgecolors=c, linewidth=0.25)
        ax.set_ylim(-0.6, 4.6)
        ax.set_yticks(range(5))
        ax.set_yticklabels([f"S{i+1}" for i in range(5)])
        ax.set_xlim(0, 3600)
        ax.set_xlabel("Time (s)", fontsize=8.5)
        ax.text(0.03, 0.91, label, transform=ax.transAxes, fontsize=7.6,
                va="top", ha="left",
                bbox=dict(facecolor="white", edgecolor="#cccccc", linewidth=0.45, alpha=0.92, pad=1.2))
        polish(ax, grid_axis="x")
        ax.tick_params(axis="y", labelsize=7.5)
    axes[0].set_ylabel("Satellite", fontsize=8.5)
    handles = [
        Patch(facecolor=colors["emergency"], label="Emergency task"),
        Patch(facecolor=colors["original"], label="Original-plan task"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.04),
               ncol=3, frameon=False, fontsize=7.2, handlelength=1.6, columnspacing=1.2)
    fig.subplots_adjust(wspace=0.08, left=0.065, right=0.995, top=0.84, bottom=0.25)
    save(fig, "preference_gantt_triptych_compact")


def plot_complexity():
    df = pd.read_csv(COMPLEX)
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
    ax.set_xlabel(r"Candidate pairs $C$ (million)", fontsize=8.8)
    ax.set_ylabel("Runtime (s)", fontsize=8.8)
    ax.set_xlim(-0.7, len(agg) - 0.3)
    polish(ax, grid_axis="y")
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
    setup()
    plot_pref_sensitivity()
    plot_dense_response()
    plot_gantt_triptych()
    plot_complexity()
    print("redrawn figures saved to", FIG)
