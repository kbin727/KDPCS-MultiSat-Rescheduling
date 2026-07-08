"""Final polish for KDPCS Fig. 8, Fig. 9, and Fig. 10.

Outputs are written to both the final Overleaf work directory and the GitHub
release paper_tex figure directory so that the manuscript package and the
public archive remain synchronized.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(r"E:\论文材料\DynamicReschedule_MultiSat")
FINAL_FIG = ROOT / "results" / "KDPCS_overleaf_final_20260709_clean" / "figures"
RELEASE_FIG = (
    ROOT
    / "results"
    / "KDPCS_github_release_20260707"
    / "paper_tex"
    / "figures"
)
RELEASE_MIRROR = ROOT / "results" / "KDPCS_github_release_20260707" / "paper_figures"
PREVIEW = ROOT / "results" / "KDPCS_overleaf_final_20260709_clean" / "figures" / "preview_final"

PREF_ROWS = (
    ROOT
    / "results"
    / "server_preference_sensitivity_20260708"
    / "preference_count_sensitivity"
    / "preference_count_sensitivity_rows.csv"
)
DENSE_S5 = (
    ROOT
    / "results"
    / "server_preference_sensitivity_20260708"
    / "tradeoff_dense_s5_n150_inst0_p51"
    / "preference_tradeoff_objectives.csv"
)
DENSE_S10 = (
    ROOT
    / "results"
    / "server_preference_sensitivity_20260708"
    / "tradeoff_dense_s10_n300_inst0_p51"
    / "preference_tradeoff_objectives.csv"
)
GANTT = (
    ROOT
    / "results"
    / "_tmp_gantt_candidate_search_20260709"
    / "s8_n400_inst9"
    / "preference_tradeoff_schedules.csv"
)
GANTT_OBJ = (
    ROOT
    / "results"
    / "_tmp_gantt_candidate_search_20260709"
    / "s8_n400_inst9"
    / "preference_tradeoff_objectives.csv"
)

BLUE = "#173f7a"
RED = "#bf3f38"
GOLD = "#d59b2d"
GRAY = "#6f7782"
BOX = "#c8d8ef"
GRID = "#dddddd"
TEXT = "#222222"


def style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.65,
            "axes.edgecolor": TEXT,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
        }
    )
    FINAL_FIG.mkdir(parents=True, exist_ok=True)
    RELEASE_FIG.mkdir(parents=True, exist_ok=True)
    RELEASE_MIRROR.mkdir(parents=True, exist_ok=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)


def polish(ax, grid_axis="y"):
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.5, alpha=0.75, zorder=0)
    ax.tick_params(labelsize=7.4, pad=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)
        spine.set_color(TEXT)


def save(fig, name):
    outputs = [FINAL_FIG / f"{name}.pdf", RELEASE_FIG / f"{name}.pdf", RELEASE_MIRROR / f"{name}.pdf"]
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", pad_inches=0.004)
    fig.savefig(PREVIEW / f"{name}.png", dpi=350, bbox_inches="tight", pad_inches=0.004)
    print(f"saved {name}")
    plt.close(fig)


def preference_data():
    rows = pd.read_csv(PREF_ROWS)
    ref = rows[rows["num_prefs"] == 200].set_index(["scale", "instance_idx"])["hv"]
    rows["normalized_hv"] = [
        100.0 * row.hv / ref.loc[(row.scale, row.instance_idx)] for row in rows.itertuples()
    ]
    return rows


def draw_mean_sem(ax, x, means, sems, color, marker="o"):
    ax.fill_between(x, means - sems, means + sems, color=color, alpha=0.16, linewidth=0)
    ax.plot(
        x,
        means,
        color=color,
        linewidth=1.55,
        marker=marker,
        markersize=3.7,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=1.0,
        zorder=5,
    )


def fig8():
    rows = preference_data()
    ks = np.array([20, 50, 100, 150, 200])

    hv_means, hv_sems, rt_means, rt_sems = [], [], [], []
    hv_groups, rt_groups = [], []
    for k in ks:
        sub = rows[rows["num_prefs"] == k]
        hv = sub["normalized_hv"].to_numpy()
        rt = sub["seconds"].to_numpy()
        hv_groups.append(hv)
        rt_groups.append(rt)
        hv_means.append(hv.mean())
        hv_sems.append(hv.std(ddof=1) / np.sqrt(len(hv)))
        rt_means.append(rt.mean())
        rt_sems.append(rt.std(ddof=1) / np.sqrt(len(rt)))
    hv_means = np.array(hv_means)
    hv_sems = np.array(hv_sems)
    rt_means = np.array(rt_means)
    rt_sems = np.array(rt_sems)

    # HV: compact zoomed line + faint distribution. This is more readable than
    # a full-range boxplot because the useful variation is about two percent.
    fig, ax = plt.subplots(figsize=(3.35, 1.38))
    x = np.arange(len(ks))
    rng = np.random.default_rng(11)
    for idx, vals in enumerate(hv_groups):
        jitter = rng.uniform(-0.065, 0.065, len(vals))
        ax.scatter(
            np.full(len(vals), idx) + jitter,
            vals,
            s=5,
            color=BLUE,
            alpha=0.16,
            linewidths=0,
            zorder=2,
        )
    draw_mean_sem(ax, x, hv_means, hv_sems, BLUE)
    ax.axhline(100.0, color=GRAY, linewidth=0.65, linestyle=(0, (3, 3)), zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlim(-0.35, len(ks) - 0.65)
    ax.set_ylim(97.6, 100.35)
    ax.set_xlabel(r"Preference count $K$", fontsize=8.2, labelpad=1.5)
    ax.set_ylabel("Normalized HV (%)", fontsize=8.2, labelpad=2)
    polish(ax)
    ax.legend(
        [Line2D([0], [0], color=BLUE, marker="o", markerfacecolor="white", linewidth=1.55)],
        ["Mean ± SEM"],
        loc="lower right",
        fontsize=6.7,
        handlelength=1.6,
        borderaxespad=0.25,
    )
    save(fig, "preference_hv_sensitivity")

    # Runtime: compact boxplot + mean line. No engineering labels in legend.
    fig, ax = plt.subplots(figsize=(3.35, 1.38))
    bp = ax.boxplot(
        rt_groups,
        positions=x,
        widths=0.42,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": TEXT, "linewidth": 0.8},
        boxprops={"color": BLUE, "linewidth": 0.65},
        whiskerprops={"color": BLUE, "linewidth": 0.55},
        capprops={"color": BLUE, "linewidth": 0.55},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(BOX)
        patch.set_alpha(0.8)
    ax.plot(
        x,
        rt_means,
        color=BLUE,
        linewidth=1.35,
        marker="o",
        markersize=3.4,
        markerfacecolor="white",
        markeredgewidth=1.0,
        zorder=6,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlim(-0.35, len(ks) - 0.65)
    ax.set_ylim(0, max(np.percentile(np.concatenate(rt_groups), 98), rt_means.max()) * 1.15)
    ax.set_xlabel(r"Preference count $K$", fontsize=8.2, labelpad=1.5)
    ax.set_ylabel("Runtime (s)", fontsize=8.2, labelpad=2)
    polish(ax)
    ax.legend(
        [
            Patch(facecolor=BOX, edgecolor=BLUE, alpha=0.8),
            Line2D([0], [0], color=BLUE, marker="o", markerfacecolor="white", linewidth=1.35),
        ],
        ["Distribution", "Mean"],
        loc="upper left",
        fontsize=6.6,
        handlelength=1.4,
        borderaxespad=0.25,
        ncol=1,
    )
    save(fig, "preference_runtime_sensitivity")


def fig9():
    panels = [
        (pd.read_csv(DENSE_S5).sort_values("w2"), "S5-T150"),
        (pd.read_csv(DENSE_S10).sort_values("w2"), "S10-T300"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 1.65), sharey=False)
    for ax, (df, label) in zip(axes, panels):
        w2 = df["w2"].to_numpy()
        f1 = df["f1"].to_numpy()
        f2 = df["f2"].to_numpy()
        ax.plot(w2, f1, color=BLUE, linewidth=1.35, marker="o", markersize=2.1, markevery=5, label=r"$f_1$")
        ax.plot(w2, f2, color=RED, linewidth=1.35, marker="s", markersize=2.1, markevery=5, label=r"$f_2$")
        ax.text(0.02, 0.95, label, transform=ax.transAxes, ha="left", va="top", fontsize=8.0, color=TEXT)
        ax.set_xlabel(r"Preference weight $\omega_2$", fontsize=8.1, labelpad=1.5)
        ax.set_ylabel("Objective value", fontsize=8.1, labelpad=2)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.legend(loc="center right", fontsize=6.8, handlelength=1.4, borderaxespad=0.25)
        polish(ax, grid_axis="both")
    fig.subplots_adjust(left=0.07, right=0.995, top=0.98, bottom=0.26, wspace=0.18)
    save(fig, "preference_dense_tradeoff_response")


def fig10():
    df = pd.read_csv(GANTT)
    obj = pd.read_csv(GANTT_OBJ).set_index("pref_id")
    prefs = [(0, r"$\omega=(1,0)$"), (2, r"$\omega=(0.5,0.5)$"), (4, r"$\omega=(0,1)$")]
    fig, axes = plt.subplots(1, 3, figsize=(6.85, 2.02), sharex=True, sharey=True)
    for ax, (pid, label) in zip(axes, prefs):
        sub = df[df["pref_id"] == pid]
        for row in sub.itertuples():
            sat = int(row.satellite_id)
            color = BLUE if bool(row.original_scheduled) else RED
            ax.broken_barh(
                [(float(row.start), float(row.end) - float(row.start))],
                (sat - 0.27, 0.54),
                facecolors=color,
                edgecolors=color,
                linewidth=0.18,
                zorder=3,
            )
        ax.set_xlim(0, 3600)
        ax.set_ylim(-0.65, 7.65)
        ax.set_yticks(range(8))
        ax.set_yticklabels([f"S{i+1}" for i in range(8)])
        ax.set_xlabel("Time (s)", fontsize=7.9, labelpad=1.5)
        f1 = float(obj.loc[pid, "f1"])
        f2 = float(obj.loc[pid, "f2"])
        label_with_obj = rf"{label}" + "\n" + rf"$f_1={f1:.3f},\ f_2={f2:.3f}$"
        ax.text(
            0.5,
            1.025,
            label_with_obj,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=6.8,
            linespacing=0.95,
        )
        polish(ax, grid_axis="x")
    axes[0].set_ylabel("Satellite", fontsize=7.9, labelpad=2)
    fig.legend(
        handles=[Patch(facecolor=RED, label="Inserted task"), Patch(facecolor=BLUE, label="Original-plan task")],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.012),
        ncol=2,
        fontsize=7.0,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.5,
    )
    fig.subplots_adjust(left=0.06, right=0.995, top=0.81, bottom=0.25, wspace=0.075)
    save(fig, "preference_gantt_triptych_compact")


def main():
    style()
    fig8()
    fig9()
    fig10()


if __name__ == "__main__":
    main()
