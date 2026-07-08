"""
Redraw Fig.8, Fig.9, Fig.10, Fig.12 for KDPCS paper.
Package: KDPCS_overleaf_clean_20260709_concept_fixed

Fig.8: preference_hv_sensitivity.pdf + preference_runtime_sensitivity.pdf
       (box + mean line, HV as retention relative to K=200)
Fig.9: preference_dense_tradeoff_response.pdf (2 panels, compact)
Fig.10: preference_gantt_triptych_compact.pdf (3 panels, horizontal, 2 categories)
Fig.12: kdpcs_complexity_validation_20260705.pdf (bars + error bars + trend)

Task categories: only "Emergency task" and "Original-plan task".
No "Regular", "Retained regular", "Other regular".
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import matplotlib.ticker as mticker

# =============================================================================
# Global style
# =============================================================================
rcParams["font.family"] = "Times New Roman"
rcParams["mathtext.fontset"] = "stix"
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
rcParams["axes.linewidth"] = 0.7
rcParams["xtick.major.width"] = 0.6
rcParams["ytick.major.width"] = 0.6
rcParams["xtick.major.size"] = 3.0
rcParams["ytick.major.size"] = 3.0
rcParams["xtick.direction"] = "out"
rcParams["ytick.direction"] = "out"
rcParams["axes.grid"] = True
rcParams["grid.color"] = "#DDDDDD"
rcParams["grid.linewidth"] = 0.5
rcParams["grid.alpha"] = 0.6
rcParams["legend.frameon"] = True
rcParams["legend.framealpha"] = 0.92
rcParams["legend.edgecolor"] = "#BBBBBB"
rcParams["legend.fancybox"] = False
rcParams["axes.axisbelow"] = True

# =============================================================================
# Paths
# =============================================================================
WORK = r"E:\论文材料\DynamicReschedule_MultiSat\results\KDPCS_overleaf_clean_20260709_concept_fixed_work"
FIG_DIR = os.path.join(WORK, "figures")
PREVIEW_DIR = os.path.join(FIG_DIR, "preview_v13")
os.makedirs(PREVIEW_DIR, exist_ok=True)

DATA_ROOT = r"E:\论文材料\DynamicReschedule_MultiSat\results"
PREF_SUMMARY = os.path.join(DATA_ROOT,
    "server_preference_sensitivity_20260708", "preference_count_sensitivity",
    "preference_count_sensitivity_summary.csv")
PREF_ROWS = os.path.join(DATA_ROOT,
    "server_preference_sensitivity_20260708", "preference_count_sensitivity",
    "preference_count_sensitivity_rows.csv")
DENSE_S5 = os.path.join(DATA_ROOT,
    "server_preference_sensitivity_20260708",
    "tradeoff_dense_s5_n150_inst0_p51", "preference_tradeoff_objectives.csv")
DENSE_S10 = os.path.join(DATA_ROOT,
    "server_preference_sensitivity_20260708",
    "tradeoff_dense_s10_n300_inst0_p51", "preference_tradeoff_objectives.csv")
GANTT_CSV = os.path.join(DATA_ROOT,
    "server_preference_sensitivity_20260708",
    "tradeoff_gantt_s5_n150_inst0", "preference_tradeoff_schedules.csv")
COMPLEXITY_CSV = os.path.join(DATA_ROOT,
    "final_kdpcs_all18_merged_20260705", "complexity_validation",
    "kdpcs_complexity_points.csv")

# =============================================================================
# Colors
# =============================================================================
COLOR_KDPCS = "#1f3b73"       # deep navy
COLOR_F1 = "#1f3b73"          # deep navy for f1
COLOR_F2 = "#c0413b"          # muted red for f2
COLOR_EMER = "#c0413b"        # red for emergency tasks
COLOR_ORIG = "#1f3b73"        # deep blue for original-plan tasks
COLOR_MEAN = "#1f3b73"
COLOR_BOX = "#7fa3d6"
COLOR_SCATTER = "#34568b"

SCALE_LABELS = {
    "s3_n100": "S3-T100", "s3_n150": "S3-T150", "s3_n200": "S3-T200",
    "s5_n100": "S5-T100", "s5_n150": "S5-T150", "s5_n200": "S5-T200",
    "s8_n200": "S8-T200", "s8_n300": "S8-T300", "s8_n400": "S8-T400",
    "s10_n200": "S10-T200", "s10_n300": "S10-T300", "s10_n400": "S10-T400",
    "s15_n400": "S15-T400", "s15_n600": "S15-T600", "s15_n800": "S15-T800",
    "s20_n400": "S20-T400", "s20_n600": "S20-T600", "s20_n800": "S20-T800",
}
K_VALUES = [20, 50, 100, 150, 200]


def save_fig(fig, name):
    pdf_path = os.path.join(FIG_DIR, name + ".pdf")
    png_path = os.path.join(PREVIEW_DIR, name + ".png")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.005)
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight",
                pad_inches=0.005)
    plt.close(fig)
    print(f"  PDF: {pdf_path}")
    print(f"  PNG: {png_path}")
    return pdf_path, png_path


# =============================================================================
# Fig.8: Preference-count sensitivity
#   HV: box + mean line, y = retention relative to K=200 (%)
#   Runtime: box + mean line, raw seconds
# =============================================================================
def fig8_preference_sensitivity():
    print("\n=== Fig.8: Preference-count sensitivity ===")
    rows = pd.read_csv(PREF_ROWS)
    summary = pd.read_csv(PREF_SUMMARY)

    # K=200 reference HV (all group)
    ref_hv = summary[(summary["scale_group"] == "all") &
                     (summary["num_prefs"] == 200)]["avg_hv"].values[0]
    print(f"  K=200 reference HV: {ref_hv:.6f}")

    # --- HV retention plot ---
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    data_hv = []
    for K in K_VALUES:
        sub = rows[rows["num_prefs"] == K]
        # retention relative to K=200, in percent
        retention = (sub["hv"].values / ref_hv) * 100.0
        data_hv.append(retention)

    positions = np.arange(len(K_VALUES))
    bp = ax.boxplot(data_hv, positions=positions, widths=0.5,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color=COLOR_MEAN, lw=1.3),
                    whiskerprops=dict(color="#444444", lw=0.8),
                    capprops=dict(color="#444444", lw=0.8),
                    boxprops=dict(facecolor=COLOR_BOX, alpha=0.45,
                                  edgecolor="#444444", lw=0.8))
    # jitter scatter
    rng = np.random.RandomState(7)
    for i, vals in enumerate(data_hv):
        jitter = rng.uniform(-0.14, 0.14, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   s=9, color=COLOR_SCATTER, alpha=0.3, linewidths=0,
                   zorder=3)
    # mean line
    means = [np.mean(d) for d in data_hv]
    ax.plot(positions, means, color=COLOR_MEAN, lw=1.5, marker="o",
            markersize=4.5, markerfacecolor="white",
            markeredgecolor=COLOR_MEAN, markeredgewidth=1.2,
            zorder=6, label="Mean")
    # 100% reference line
    ax.axhline(100, color="#888888", lw=0.7, linestyle=(0, (3, 3)),
               alpha=0.6, zorder=1)

    ax.set_xticks(positions)
    ax.set_xticklabels([str(K) for K in K_VALUES])
    ax.set_xlabel(r"Preference count $K$", fontsize=9)
    ax.set_ylabel(r"HV retention vs.\ $K{=}200$ (\%)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_xlim(-0.6, len(K_VALUES) - 0.4)
    ax.legend(loc="lower right", fontsize=7.5, borderpad=0.3,
              handlelength=1.5)
    fig.tight_layout()
    save_fig(fig, "preference_hv_sensitivity")

    # --- Runtime plot ---
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    data_rt = [rows[rows["num_prefs"] == K]["seconds"].values for K in K_VALUES]
    bp = ax.boxplot(data_rt, positions=positions, widths=0.5,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color=COLOR_MEAN, lw=1.3),
                    whiskerprops=dict(color="#444444", lw=0.8),
                    capprops=dict(color="#444444", lw=0.8),
                    boxprops=dict(facecolor=COLOR_BOX, alpha=0.45,
                                  edgecolor="#444444", lw=0.8))
    rng = np.random.RandomState(7)
    for i, vals in enumerate(data_rt):
        jitter = rng.uniform(-0.14, 0.14, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   s=9, color=COLOR_SCATTER, alpha=0.3, linewidths=0,
                   zorder=3)
    means_rt = [np.mean(d) for d in data_rt]
    ax.plot(positions, means_rt, color=COLOR_MEAN, lw=1.5, marker="o",
            markersize=4.5, markerfacecolor="white",
            markeredgecolor=COLOR_MEAN, markeredgewidth=1.2,
            zorder=6, label="Mean")

    ax.set_xticks(positions)
    ax.set_xticklabels([str(K) for K in K_VALUES])
    ax.set_xlabel(r"Preference count $K$", fontsize=9)
    ax.set_ylabel("Runtime (s)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_xlim(-0.6, len(K_VALUES) - 0.4)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=7.5, borderpad=0.3,
              handlelength=1.5)
    fig.tight_layout()
    save_fig(fig, "preference_runtime_sensitivity")


# =============================================================================
# Fig.9: Dense tradeoff response (2 panels, compact)
# =============================================================================
def fig9_dense_tradeoff_response():
    print("\n=== Fig.9: Dense tradeoff response ===")
    df_s5 = pd.read_csv(DENSE_S5).sort_values("w2").reset_index(drop=True)
    df_s10 = pd.read_csv(DENSE_S10).sort_values("w2").reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=False)
    panels = [(df_s5, r"$S5$-$T150$", axes[0]),
              (df_s10, r"$S10$-$T300$", axes[1])]

    for df, label, ax in panels:
        w2 = df["w2"].values
        f1 = df["f1"].values
        f2 = df["f2"].values
        marker_step = 5
        ax.plot(w2, f1, color=COLOR_F1, lw=1.4, zorder=4,
                label=r"$f_1$ (emergency gain)")
        ax.plot(w2[::marker_step], f1[::marker_step], "o",
                markerfacecolor="white", markeredgecolor=COLOR_F1,
                markersize=2.5, markeredgewidth=0.8, zorder=5)
        ax.plot(w2, f2, color=COLOR_F2, lw=1.4, zorder=4,
                label=r"$f_2$ (plan stability)")
        ax.plot(w2[::marker_step], f2[::marker_step], "s",
                markerfacecolor="white", markeredgecolor=COLOR_F2,
                markersize=2.5, markeredgewidth=0.8, zorder=5)

        ax.set_xlabel(r"Preference weight $\omega_2$", fontsize=9)
        ax.set_ylabel("Objective value", fontsize=9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.tick_params(labelsize=8)
        ax.text(0.02, 0.97, label, transform=ax.transAxes,
                fontsize=9.5, va="top", ha="left", color="#333333")
        ax.legend(loc="center right", fontsize=7.5, borderpad=0.3,
                  handlelength=1.8)
        ax.grid(True, which="major", color="#DDDDDD", linewidth=0.5,
                alpha=0.6)

    fig.tight_layout()
    save_fig(fig, "preference_dense_tradeoff_response")


# =============================================================================
# Fig.10: Gantt triptych (3 panels, horizontal, compact)
#   Only 2 categories: Emergency task, Original-plan task
# =============================================================================
def fig10_gantt_triptych():
    print("\n=== Fig.10: Gantt triptych ===")
    g = pd.read_csv(GANTT_CSV)
    prefs = [
        (0, r"$\omega=(1,0)$"),
        (2, r"$\omega=(0.5,0.5)$"),
        (4, r"$\omega=(0,1)$"),
    ]
    n_sats = 5
    sat_labels = [f"S{i+1}" for i in range(n_sats)]
    t_max = 3600

    fig, axes = plt.subplots(3, 1, figsize=(6.8, 4.2), sharex=True)

    for ax, (pid, pref_label) in zip(axes, prefs):
        sub = g[g["pref_id"] == pid]
        for _, row in sub.iterrows():
            y = int(row["satellite_id"])
            is_emer = bool(row["is_emergency"])
            # Only two categories: emergency (red) or original-plan (blue)
            color = COLOR_EMER if is_emer else COLOR_ORIG
            ax.barh(y, row["end"] - row["start"], left=row["start"],
                    height=0.66, color=color, edgecolor="white",
                    linewidth=0.35, zorder=3)
        ax.set_xlim(0, t_max)
        ax.set_ylim(-0.6, n_sats - 0.4)
        ax.set_yticks(range(n_sats))
        ax.set_yticklabels(sat_labels, fontsize=8)
        ax.tick_params(labelsize=8)
        ax.text(0.008, 0.95, pref_label, transform=ax.transAxes,
                fontsize=9, va="top", ha="left", color="#222222",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="#cccccc", linewidth=0.5, alpha=0.9))
        ax.grid(True, axis="x", color="#DDDDDD", linewidth=0.4, alpha=0.5,
                zorder=0)
        ax.grid(False, axis="y")
        ax.invert_yaxis()

    axes[-1].set_xlabel("Time (s)", fontsize=9)
    for ax in axes[:-1]:
        ax.set_xlabel("")

    # Legend at top
    legend_handles = [
        Patch(facecolor=COLOR_EMER, edgecolor="white", linewidth=0.4,
              label="Emergency task"),
        Patch(facecolor=COLOR_ORIG, edgecolor="white", linewidth=0.4,
              label="Original-plan task"),
    ]
    axes[0].legend(handles=legend_handles, loc="upper center",
                   bbox_to_anchor=(0.5, 1.28), ncol=2, fontsize=8.5,
                   handlelength=1.5, borderpad=0.3, columnspacing=1.5,
                   framealpha=0.92)

    fig.tight_layout()
    save_fig(fig, "preference_gantt_triptych_compact")


# =============================================================================
# Fig.12: Complexity validation
#   Bars + error bars + linear trend (no raw scatter)
# =============================================================================
def fig12_complexity():
    print("\n=== Fig.12: Complexity validation ===")
    df = pd.read_csv(COMPLEXITY_CSV)
    df["C_million"] = df["candidate_pairs"] / 1e6
    df = df.sort_values("C_million").reset_index(drop=True)

    # Aggregate per scale
    agg = df.groupby("scale").agg(
        C_mean=("C_million", "mean"),
        C_std=("C_million", "std"),
        t_mean=("seconds", "mean"),
        t_std=("seconds", "std"),
        group=("group", "first"),
    ).reset_index()
    agg = agg.sort_values("C_mean").reset_index(drop=True)

    group_colors = {
        "Small": "#5a9a6a",
        "Medium": "#1f3b73",
        "Large": "#c0413b",
    }

    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    x_positions = np.arange(len(agg))
    bar_width = 0.6

    for i, row in agg.iterrows():
        color = group_colors.get(row["group"], "#888888")
        ax.bar(i, row["t_mean"], width=bar_width,
               color=color, edgecolor="white", linewidth=0.4,
               alpha=0.65, zorder=3)
        # error bar
        y_lo = max(0, row["t_mean"] - row["t_std"])
        y_hi = row["t_mean"] + row["t_std"]
        ax.plot([i, i], [y_lo, y_hi], color="#333333", lw=0.7,
                solid_capstyle="round", zorder=5)
        ax.plot([i - 0.1, i + 0.1], [y_hi, y_hi],
                color="#333333", lw=0.7, solid_capstyle="round", zorder=5)
        ax.plot([i - 0.1, i + 0.1], [y_lo, y_lo],
                color="#333333", lw=0.7, solid_capstyle="round", zorder=5)

    # Linear fit based on scale mean C and mean runtime (OLS)
    C_means = agg["C_mean"].values
    t_means = agg["t_mean"].values
    coeff = np.polyfit(C_means, t_means, 1)
    ax.plot(x_positions,
            np.polyval(coeff, C_means),
            color="#c0413b", lw=1.3, linestyle="--",
            marker="D", markersize=3.5, markerfacecolor="white",
            markeredgecolor="#c0413b", markeredgewidth=0.8,
            zorder=7, label="Linear fit")

    scale_labels = [SCALE_LABELS.get(s, s) for s in agg["scale"]]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(scale_labels, rotation=55, ha="right", fontsize=6.5)
    ax.set_xlabel(r"Candidate pairs $C$ (million)", fontsize=9)
    ax.set_ylabel("Runtime (s)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_xlim(-0.7, len(agg) - 0.3)

    legend_handles = [
        Patch(facecolor=group_colors["Small"], edgecolor="white", alpha=0.65,
              label="Small"),
        Patch(facecolor=group_colors["Medium"], edgecolor="white", alpha=0.65,
              label="Medium"),
        Patch(facecolor=group_colors["Large"], edgecolor="white", alpha=0.65,
              label="Large"),
        Line2D([0], [0], color="#c0413b", lw=1.3, linestyle="--",
               marker="D", markersize=3.5, markerfacecolor="white",
               markeredgecolor="#c0413b", markeredgewidth=0.8,
               label="Linear fit"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=7,
              borderpad=0.3, handlelength=1.3, labelspacing=0.25,
              columnspacing=1.0)
    fig.tight_layout()
    save_fig(fig, "kdpcs_complexity_validation_20260705")


# =============================================================================
# Run all
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Redrawing Fig.8, Fig.9, Fig.10, Fig.12")
    print("=" * 60)
    fig8_preference_sensitivity()
    fig9_dense_tradeoff_response()
    fig10_gantt_triptych()
    fig12_complexity()
    print("\n" + "=" * 60)
    print("All figures generated.")
    print("=" * 60)
