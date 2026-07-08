"""
Preference sensitivity analysis figures for KDPCS paper.
Top-journal style: clean, professional, vector PDF + PNG preview.

Generates:
  1. preference_hv_sensitivity.pdf
  2. preference_nd_sensitivity.pdf
  3. preference_runtime_sensitivity.pdf
  4. preference_quality_efficiency_tradeoff.pdf
  5. preference_dense_tradeoff_response.pdf
  6. preference_representative_gantt.pdf  (optional, only if clean)

No in-figure titles. All captions are left to LaTeX.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.ticker as mticker

# -----------------------------------------------------------------------------
# Global style
# -----------------------------------------------------------------------------
rcParams["font.family"] = "Times New Roman"
rcParams["mathtext.fontset"] = "stix"
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
rcParams["axes.linewidth"] = 0.9
rcParams["xtick.major.width"] = 0.8
rcParams["ytick.major.width"] = 0.8
rcParams["xtick.major.size"] = 3.5
rcParams["ytick.major.size"] = 3.5
rcParams["xtick.direction"] = "out"
rcParams["ytick.direction"] = "out"
rcParams["axes.grid"] = True
rcParams["grid.color"] = "#DDDDDD"
rcParams["grid.linewidth"] = 0.6
rcParams["grid.alpha"] = 0.7
rcParams["legend.frameon"] = True
rcParams["legend.framealpha"] = 0.9
rcParams["legend.edgecolor"] = "#BBBBBB"
rcParams["legend.fancybox"] = False
rcParams["axes.axisbelow"] = True

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT = r"E:\论文材料\DynamicReschedule_MultiSat\results\server_preference_sensitivity_20260708"
ROWS_CSV = os.path.join(ROOT, "preference_count_sensitivity",
                        "preference_count_sensitivity_rows.csv")
SUMMARY_CSV = os.path.join(ROOT, "preference_count_sensitivity",
                           "preference_count_sensitivity_summary.csv")
DENSE_S5 = os.path.join(ROOT, "tradeoff_dense_s5_n150_inst0_p51",
                        "preference_tradeoff_objectives.csv")
DENSE_S10 = os.path.join(ROOT, "tradeoff_dense_s10_n300_inst0_p51",
                         "preference_tradeoff_objectives.csv")
GANTT_S5 = os.path.join(ROOT, "tradeoff_gantt_s5_n150_inst0",
                        "preference_tradeoff_schedules.csv")
GANTT_S10 = os.path.join(ROOT, "tradeoff_gantt_s10_n300_inst0",
                         "preference_tradeoff_schedules.csv")

OUT_DIR = os.path.join(ROOT, "figures_trae_final")
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Palette (high-end, restrained)
# -----------------------------------------------------------------------------
# "all" gets the boldest dark-blue, scale groups softer.
COLOR_ALL = "#1f3b73"      # deep navy
COLOR_SMALL = "#4a7ab8"    # mid blue
COLOR_MEDIUM = "#6fb39a"   # muted teal-green
COLOR_LARGE = "#c97a4a"    # muted terracotta
COLOR_F1 = "#1f3b73"       # deep navy for f1
COLOR_F2 = "#c0413b"       # muted red-orange for f2
COLOR_MEAN = "#1f3b73"
COLOR_BOX = "#7fa3d6"
COLOR_SCATTER = "#34568b"

K_VALUES = [20, 50, 100, 150, 200]
K_MAIN = 50  # main-experiment K


def save_fig(fig, name):
    """Save figure as cropped PDF + PNG preview."""
    pdf_path = os.path.join(OUT_DIR, name + ".pdf")
    png_path = os.path.join(OUT_DIR, name + ".png")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.008)
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight",
                pad_inches=0.008)
    plt.close(fig)
    print(f"  saved: {pdf_path}")
    print(f"  saved: {png_path}")
    return pdf_path, png_path


# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
rows = pd.read_csv(ROWS_CSV)
summary = pd.read_csv(SUMMARY_CSV)
dense_s5 = pd.read_csv(DENSE_S5)
dense_s10 = pd.read_csv(DENSE_S10)
print(f"rows: {len(rows)}, summary: {len(summary)}, "
      f"dense_s5: {len(dense_s5)}, dense_s10: {len(dense_s10)}")

# -----------------------------------------------------------------------------
# Figure 1: HV sensitivity vs K
#   - 4 mean curves (small/medium/large/all), with shaded SEM bands
#   - light vertical dashed line at K=50
# -----------------------------------------------------------------------------
def fig1_hv_sensitivity():
    print("\n[Fig 1] preference_hv_sensitivity.pdf")
    fig, ax = plt.subplots(figsize=(5.2, 3.4))

    groups = [("all", COLOR_ALL, 2.1, "o"),
              ("small", COLOR_SMALL, 1.3, "s"),
              ("medium", COLOR_MEDIUM, 1.3, "^"),
              ("large", COLOR_LARGE, 1.3, "D")]
    for grp, color, lw, mk in groups:
        sub = summary[summary["scale_group"] == grp].sort_values("num_prefs")
        K = sub["num_prefs"].values
        mean = sub["avg_hv"].values
        std = sub["std_hv"].values
        n = sub["records"].values
        sem = std / np.sqrt(n)
        # shade
        ax.fill_between(K, mean - sem, mean + sem, color=color, alpha=0.16,
                        linewidth=0, zorder=2)
        # mean line
        ax.plot(K, mean, color=color, lw=lw, marker=mk, markersize=5.0,
                markerfacecolor="white", markeredgecolor=color,
                markeredgewidth=1.4, zorder=4,
                label=("All" if grp == "all" else
                       grp.capitalize()))
        # subtle individual-instance scatter to show data mass
        sub_rows = rows[rows["scale_group"] == grp]
        for Kval in K_VALUES:
            inst = sub_rows[sub_rows["num_prefs"] == Kval]["hv"].values
            jitter = np.random.RandomState(42).uniform(-3.5, 3.5, len(inst))
            ax.scatter(np.full_like(inst, Kval, dtype=float) + jitter,
                       inst, color=color, s=4, alpha=0.18, linewidths=0,
                       zorder=1)

    # highlight K=50
    ax.axvline(K_MAIN, color="#888888", lw=0.9, linestyle=(0, (4, 3)),
               alpha=0.7, zorder=1)

    ax.set_xlabel(r"Preference count $K$", fontsize=11)
    ax.set_ylabel("Hypervolume (HV)", fontsize=11)
    ax.set_xticks(K_VALUES)
    ax.set_xlim(8, 212)
    ax.tick_params(labelsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12),
              ncol=4, fontsize=9.5, handlelength=2.0, borderpad=0.4,
              columnspacing=1.2)
    ax.set_ylim(bottom=0.29)
    fig.tight_layout()
    return save_fig(fig, "preference_hv_sensitivity")


# -----------------------------------------------------------------------------
# Figure 2: ND points sensitivity vs K  (box + jitter + mean line)
#   - single 'all' panel to keep it clean
# -----------------------------------------------------------------------------
def fig2_nd_sensitivity():
    print("\n[Fig 2] preference_nd_sensitivity.pdf")
    fig, ax = plt.subplots(figsize=(5.2, 3.4))

    # Aggregate all 90 instances per K (30 small + 30 medium + 30 large)
    data_all = []
    for K in K_VALUES:
        sub = rows[rows["num_prefs"] == K]
        data_all.append(sub["nd_points"].values)

    positions = np.arange(len(K_VALUES))
    bp = ax.boxplot(data_all, positions=positions, widths=0.55,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color=COLOR_ALL, lw=1.4),
                    whiskerprops=dict(color="#444444", lw=0.9),
                    capprops=dict(color="#444444", lw=0.9),
                    boxprops=dict(facecolor=COLOR_BOX, alpha=0.55,
                                  edgecolor="#444444", lw=0.9))
    # jitter scatter
    rng = np.random.RandomState(7)
    for i, vals in enumerate(data_all):
        jitter = rng.uniform(-0.16, 0.16, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   s=10, color=COLOR_SCATTER, alpha=0.38, linewidths=0,
                   zorder=3)
    # mean line
    means = [np.mean(d) for d in data_all]
    ax.plot(positions, means, color=COLOR_MEAN, lw=1.8, marker="o",
            markersize=5.5, markerfacecolor="white",
            markeredgecolor=COLOR_MEAN, markeredgewidth=1.5,
            zorder=5, label="Mean")

    ax.set_xticks(positions)
    ax.set_xticklabels([str(K) for K in K_VALUES])
    ax.set_xlabel(r"Preference count $K$", fontsize=11)
    ax.set_ylabel("Number of non-dominated solutions", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.legend(loc="upper left", fontsize=9.5, borderpad=0.4)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return save_fig(fig, "preference_nd_sensitivity")


# -----------------------------------------------------------------------------
# Figure 3: Runtime sensitivity vs K (log y, mean +/- SEM, 4 curves)
# -----------------------------------------------------------------------------
def fig3_runtime_sensitivity():
    print("\n[Fig 3] preference_runtime_sensitivity.pdf")
    fig, ax = plt.subplots(figsize=(5.2, 3.4))

    groups = [("all", COLOR_ALL, 2.1, "o"),
              ("small", COLOR_SMALL, 1.3, "s"),
              ("medium", COLOR_MEDIUM, 1.3, "^"),
              ("large", COLOR_LARGE, 1.3, "D")]
    for grp, color, lw, mk in groups:
        sub = summary[summary["scale_group"] == grp].sort_values("num_prefs")
        K = sub["num_prefs"].values
        mean = sub["avg_seconds"].values
        std = sub["std_seconds"].values
        n = sub["records"].values
        sem = std / np.sqrt(n)
        ax.fill_between(K, mean - sem, mean + sem, color=color, alpha=0.16,
                        linewidth=0, zorder=2)
        ax.plot(K, mean, color=color, lw=lw, marker=mk, markersize=5.0,
                markerfacecolor="white", markeredgecolor=color,
                markeredgewidth=1.4, zorder=4,
                label=("All" if grp == "all" else grp.capitalize()))

    ax.axvline(K_MAIN, color="#888888", lw=0.9, linestyle=(0, (4, 3)),
               alpha=0.7, zorder=1)
    ax.set_yscale("log")
    ax.set_xlabel(r"Preference count $K$", fontsize=11)
    ax.set_ylabel("Runtime (s)", fontsize=11)
    ax.set_xticks(K_VALUES)
    ax.set_xlim(8, 212)
    ax.tick_params(labelsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12),
              ncol=4, fontsize=9.5, handlelength=2.0, borderpad=0.4,
              columnspacing=1.2)
    # minor grid restrained
    ax.grid(which="minor", linestyle=":", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    return save_fig(fig, "preference_runtime_sensitivity")


# -----------------------------------------------------------------------------
# Figure 4: Quality-efficiency tradeoff (HV vs runtime, log x)
#   - bubble size = avg ND points, color by K
#   - small/medium/large groups + all
# -----------------------------------------------------------------------------
def fig4_quality_efficiency_tradeoff():
    print("\n[Fig 4] preference_quality_efficiency_tradeoff.pdf")
    fig, ax = plt.subplots(figsize=(5.4, 3.6))

    # Color gradient over K (light blue -> deep navy)
    cmap_K = plt.cm.Blues
    K_norm = (np.array(K_VALUES) - 20) / (200 - 20)

    group_styles = {
        "small":  ("s", 0.9),
        "medium": ("^", 0.9),
        "large":  ("D", 0.9),
        "all":     ("o", 1.4),
    }

    # Plot per group, per K
    for grp, (mk, lw) in group_styles.items():
        sub = summary[summary["scale_group"] == grp].sort_values("num_prefs")
        for _, row in sub.iterrows():
            K = int(row["num_prefs"])
            color = cmap_K(0.25 + 0.6 * (K - 20) / 180)
            size = 30 + (row["avg_nd_points"] - 8) * 6.5
            size = max(20, min(220, size))
            is_all = (grp == "all")
            ax.scatter(row["avg_seconds"], row["avg_hv"],
                       s=size, c=[color], marker=mk,
                       edgecolors=("#1f3b73" if is_all else "#444444"),
                       linewidths=(1.4 if is_all else 0.7),
                       alpha=(0.95 if is_all else 0.75), zorder=5)
        # connect points of same group with thin line
        sub_sorted = sub.sort_values("num_prefs")
        ax.plot(sub_sorted["avg_seconds"], sub_sorted["avg_hv"],
                color="#888888", lw=0.6, alpha=0.5, zorder=2)

    # Annotate K=50 and K=200 for 'all' group
    sub_all = summary[summary["scale_group"] == "all"].set_index("num_prefs")
    p50 = sub_all.loc[50]
    p200 = sub_all.loc[200]
    ax.annotate(r"$K=50$", xy=(p50["avg_seconds"], p50["avg_hv"]),
                xytext=(0.55, p50["avg_hv"] - 0.005),
                fontsize=9.5, color=COLOR_ALL, ha="right",
                arrowprops=dict(arrowstyle="-", color=COLOR_ALL, lw=0.7,
                                shrinkA=0, shrinkB=2))
    ax.annotate(r"$K=200$", xy=(p200["avg_seconds"], p200["avg_hv"]),
                xytext=(9.0, p200["avg_hv"] + 0.003),
                fontsize=9.5, color=COLOR_ALL, ha="left",
                arrowprops=dict(arrowstyle="-", color=COLOR_ALL, lw=0.7,
                                shrinkA=0, shrinkB=2))

    ax.set_xscale("log")
    ax.set_xlabel("Runtime (s)", fontsize=11)
    ax.set_ylabel("Hypervolume (HV)", fontsize=11)
    ax.tick_params(labelsize=10)

    # Custom legend: group markers + size legend + K colorbar
    group_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#bbbbbb",
               markeredgecolor="#1f3b73", markersize=8, markeredgewidth=1.4,
               label="All"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#bbbbbb",
               markeredgecolor="#444444", markersize=7, markeredgewidth=0.8,
               label="Small"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#bbbbbb",
               markeredgecolor="#444444", markersize=7, markeredgewidth=0.8,
               label="Medium"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#bbbbbb",
               markeredgecolor="#444444", markersize=7, markeredgewidth=0.8,
               label="Large"),
    ]
    leg1 = ax.legend(handles=group_handles, loc="lower right",
                     fontsize=9, title="Scale", title_fontsize=9,
                     borderpad=0.4)
    ax.add_artist(leg1)

    # Colorbar for K
    sm = plt.cm.ScalarMappable(cmap=cmap_K,
                               norm=plt.Normalize(vmin=20, vmax=200))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, fraction=0.046, aspect=18)
    cbar.set_label(r"$K$", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    fig.tight_layout()
    return save_fig(fig, "preference_quality_efficiency_tradeoff")


# -----------------------------------------------------------------------------
# Figure 5: Dense tradeoff response (1x2 panel: S5-T150, S10-T300)
#   - x: omega2, y: objective value
#   - f1 deep navy, f2 muted red-orange
# -----------------------------------------------------------------------------
def fig5_dense_tradeoff_response():
    print("\n[Fig 5] preference_dense_tradeoff_response.pdf")
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=False)

    panels = [
        (dense_s5, r"$S5$-$T150$", axes[0]),
        (dense_s10, r"$S10$-$T300$", axes[1]),
    ]
    for df, label, ax in panels:
        df = df.sort_values("w2").reset_index(drop=True)
        w2 = df["w2"].values
        f1 = df["f1"].values
        f2 = df["f2"].values
        ax.plot(w2, f1, color=COLOR_F1, lw=1.7, marker="o", markersize=3.2,
                markerfacecolor="white", markeredgecolor=COLOR_F1,
                markeredgewidth=1.0, label=r"$f_1$ (emergency gain)", zorder=4)
        ax.plot(w2, f2, color=COLOR_F2, lw=1.7, marker="s", markersize=3.2,
                markerfacecolor="white", markeredgecolor=COLOR_F2,
                markeredgewidth=1.0, label=r"$f_2$ (plan stability)", zorder=4)
        ax.set_xlabel(r"Preference weight $\omega_2$", fontsize=11)
        ax.set_ylabel("Objective value", fontsize=11)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.tick_params(labelsize=10)
        # panel tag in upper-left corner (small, no title)
        ax.text(0.02, 0.97, label, transform=ax.transAxes,
                fontsize=10.5, va="top", ha="left",
                color="#333333")
        ax.legend(loc="center right", fontsize=9, borderpad=0.4,
                  handlelength=2.0)
        ax.grid(True, which="major", color="#DDDDDD", linewidth=0.6, alpha=0.7)

    fig.tight_layout()
    return save_fig(fig, "preference_dense_tradeoff_response")


# -----------------------------------------------------------------------------
# Figure 6 (optional): Representative Gantt for S10-T300 with 3 preferences
# -----------------------------------------------------------------------------
def fig6_representative_gantt():
    print("\n[Fig 6] preference_representative_gantt.pdf (optional)")
    try:
        g = pd.read_csv(GANTT_S10)
    except Exception as e:
        print(f"  skip: cannot read gantt csv ({e})")
        return None, None

    # pick pref_ids corresponding to w1,w2 closest to (1,0),(0.5,0.5),(0,1)
    prefs = g[["pref_id", "w1", "w2"]].drop_duplicates().sort_values("pref_id")
    targets = [(1.0, 0.0), (0.5, 0.5), (0.0, 1.0)]
    chosen = []
    for t in targets:
        d = ((prefs["w1"] - t[0]) ** 2 + (prefs["w2"] - t[1]) ** 2) ** 0.5
        idx = d.idxmin()
        chosen.append(int(prefs.loc[idx, "pref_id"]))
    # check task count
    total_tasks = 0
    for pid in chosen:
        total_tasks = max(total_tasks, len(g[g["pref_id"] == pid]))
    if total_tasks > 80:
        print(f"  skip: too many tasks ({total_tasks}), would be cluttered")
        return None, None

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.6), sharey=True)
    panel_titles = [r"$\omega=(1,0)$", r"$\omega=(0.5,0.5)$", r"$\omega=(0,1)$"]
    color_emer = "#c0413b"
    color_retained = "#1f3b73"
    color_other = "#9bb3d6"

    for ax, pid, title in zip(axes, chosen, panel_titles):
        sub = g[g["pref_id"] == pid].copy()
        # Determine all satellites present in this scale
        all_sats = sorted(g["satellite_id"].unique())
        sat_to_y = {s: i for i, s in enumerate(all_sats)}
        for _, row in sub.iterrows():
            y = sat_to_y[row["satellite_id"]]
            is_emer = bool(row["is_emergency"])
            is_retained = bool(row["original_scheduled"]) and not is_emer
            color = color_emer if is_emer else (color_retained if is_retained
                                                else color_other)
            ax.barh(y, row["end"] - row["start"], left=row["start"],
                    height=0.62, color=color, edgecolor="none", linewidth=0)
        ax.set_xlim(0, 3600)
        ax.set_ylim(-0.7, len(all_sats) - 0.3)
        ax.set_yticks(range(len(all_sats)))
        ax.set_yticklabels([f"S{s}" for s in all_sats], fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.text(0.02, 1.02, title, transform=ax.transAxes,
                fontsize=10.5, va="bottom", ha="left", color="#333333")
        ax.grid(True, axis="x", color="#DDDDDD", linewidth=0.5, alpha=0.6)
        ax.grid(False, axis="y")

    legend_handles = [
        Patch(facecolor=color_emer, label="Emergency"),
        Patch(facecolor=color_retained, label="Retained regular"),
        Patch(facecolor=color_other, label="Other regular"),
    ]
    axes[-1].legend(handles=legend_handles, loc="upper right",
                    fontsize=8.5, borderpad=0.4,
                    bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    return save_fig(fig, "preference_representative_gantt")


# -----------------------------------------------------------------------------
# Run all
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Plotting preference sensitivity figures")
    print("=" * 60)
    np.random.seed(0)
    f1 = fig1_hv_sensitivity()
    f2 = fig2_nd_sensitivity()
    f3 = fig3_runtime_sensitivity()
    f4 = fig4_quality_efficiency_tradeoff()
    f5 = fig5_dense_tradeoff_response()
    f6 = fig6_representative_gantt()
    print("\n" + "=" * 60)
    print("All figures generated.")
    print("=" * 60)
