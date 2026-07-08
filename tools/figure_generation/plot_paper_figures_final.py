"""
Redraw all experimental figures for KDPCS paper.
Overleaf package: KDPCS_overleaf_clean_20260709_polished

Figures to redraw (15 total):
  Fig5/6: scale_trend_hv / scale_trend_igd / scale_trend_runtime  (3 PDFs)
  Fig7:   9 pareto front PDFs
  Fig8:   preference_hv_sensitivity / preference_runtime_sensitivity  (2 PDFs)
  Fig9:   kdpcs_complexity_validation  (1 PDF)

Style: IEEE TASE / NeurIPS top-journal. No in-figure titles.
All vector PDFs with tight crop. PNG previews included.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.ticker as mticker

# =============================================================================
# Global style
# =============================================================================
rcParams["font.family"] = "Times New Roman"
rcParams["mathtext.fontset"] = "stix"
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
rcParams["axes.linewidth"] = 0.8
rcParams["xtick.major.width"] = 0.7
rcParams["ytick.major.width"] = 0.7
rcParams["xtick.major.size"] = 3.0
rcParams["ytick.major.size"] = 3.0
rcParams["xtick.direction"] = "out"
rcParams["ytick.direction"] = "out"
rcParams["axes.grid"] = True
rcParams["grid.color"] = "#DDDDDD"
rcParams["grid.linewidth"] = 0.5
rcParams["grid.alpha"] = 0.65
rcParams["legend.frameon"] = True
rcParams["legend.framealpha"] = 0.92
rcParams["legend.edgecolor"] = "#BBBBBB"
rcParams["legend.fancybox"] = False
rcParams["axes.axisbelow"] = True

# =============================================================================
# Paths
# =============================================================================
WORK = r"E:\论文材料\DynamicReschedule_MultiSat\results\KDPCS_overleaf_clean_20260709_polished_work"
FIG_DIR = os.path.join(WORK, "figures")
PARETO_DIR = os.path.join(FIG_DIR, "pareto_final_all18_full_20260705")
PREVIEW_DIR = os.path.join(FIG_DIR, "preview_v13")
os.makedirs(PREVIEW_DIR, exist_ok=True)
os.makedirs(PARETO_DIR, exist_ok=True)

DATA_ROOT = r"E:\论文材料\DynamicReschedule_MultiSat\results"

SCALE_TREND_CSV = os.path.join(DATA_ROOT,
    "final_kdpcs_all18_merged_20260705", "scale_trend",
    "scale_trend_summary.csv")
COMPLEXITY_CSV = os.path.join(DATA_ROOT,
    "final_kdpcs_all18_merged_20260705", "complexity_validation",
    "kdpcs_complexity_points.csv")
PREF_SUMMARY = os.path.join(DATA_ROOT,
    "server_preference_sensitivity_20260708", "preference_count_sensitivity",
    "preference_count_sensitivity_summary.csv")
PREF_ROWS = os.path.join(DATA_ROOT,
    "server_preference_sensitivity_20260708", "preference_count_sensitivity",
    "preference_count_sensitivity_rows.csv")

KDPCS_FRONTS = os.path.join(DATA_ROOT,
    "server_final_kdpcs_policy_only_all18_20260705", "pareto_fronts",
    "final_kdpcs_policy_only_all18_fronts.csv")
BASELINE_FRONTS_1 = os.path.join(DATA_ROOT,
    "server_official_basilisk_incremental8_paper_suite_20260704",
    "pareto_fronts", "official_incremental8_basilisk_fronts.csv")
BASELINE_FRONTS_2 = os.path.join(DATA_ROOT,
    "server_official_basilisk_additional7_s20_paper_suite_20260705",
    "pareto_fronts", "official_additional6_s20_basilisk_fronts.csv")

# =============================================================================
# Mappings
# =============================================================================
SCALE_ORDER = [
    "s3_n100", "s3_n150", "s3_n200",
    "s5_n100", "s5_n150", "s5_n200",
    "s8_n200", "s8_n300", "s8_n400",
    "s10_n200", "s10_n300", "s10_n400",
    "s15_n400", "s15_n600", "s15_n800",
    "s20_n400", "s20_n600", "s20_n800",
]
SCALE_LABELS = {
    "s3_n100": "S3-T100", "s3_n150": "S3-T150", "s3_n200": "S3-T200",
    "s5_n100": "S5-T100", "s5_n150": "S5-T150", "s5_n200": "S5-T200",
    "s8_n200": "S8-T200", "s8_n300": "S8-T300", "s8_n400": "S8-T400",
    "s10_n200": "S10-T200", "s10_n300": "S10-T300", "s10_n400": "S10-T400",
    "s15_n400": "S15-T400", "s15_n600": "S15-T600", "s15_n800": "S15-T800",
    "s20_n400": "S20-T400", "s20_n600": "S20-T600", "s20_n800": "S20-T800",
}
METHOD_MAP = {
    "pcformer": "KDPCS", "monp": "MONP", "nsga2": "NSGA-II",
    "spea2": "SPEA2", "ibea": "IBEA", "moead": "MOEA/D",
}
METHOD_ORDER = ["KDPCS", "MONP", "NSGA-II", "SPEA2", "IBEA", "MOEA/D"]
COLORS = {
    "KDPCS": "#1f3b73",
    "MONP": "#c97a4a",
    "NSGA-II": "#5a9a6a",
    "SPEA2": "#8b6fa8",
    "IBEA": "#c0413b",
    "MOEA/D": "#d4a843",
}
MARKERS = {
    "KDPCS": "o", "MONP": "s", "NSGA-II": "^",
    "SPEA2": "D", "IBEA": "v", "MOEA/D": "P",
}
PARETO_SCALES = [
    "s3_n100", "s5_n150", "s5_n200",
    "s8_n200", "s10_n300", "s10_n400",
    "s15_n400", "s20_n600", "s20_n800",
]


def save_fig(fig, name, subdir=None):
    """Save figure as cropped PDF (to figures dir) + PNG preview."""
    if subdir:
        pdf_dir = os.path.join(FIG_DIR, subdir)
    else:
        pdf_dir = FIG_DIR
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, name + ".pdf")
    png_path = os.path.join(PREVIEW_DIR, name + ".png")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.006)
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight",
                pad_inches=0.006)
    plt.close(fig)
    print(f"  PDF: {pdf_path}")
    print(f"  PNG: {png_path}")
    return pdf_path, png_path


# =============================================================================
# Fig 5/6: Scale-wise trends (HV, IGD, Runtime)
# =============================================================================
def plot_scale_trends():
    print("\n=== Fig 5/6: Scale-wise trends ===")
    df = pd.read_csv(SCALE_TREND_CSV)
    df["method_label"] = df["method"].map(METHOD_MAP)
    df["scale_label"] = df["scale"].map(SCALE_LABELS)
    df["scale_order"] = df["scale"].map(
        {s: i for i, s in enumerate(SCALE_ORDER)})
    df = df.sort_values("scale_order")

    x = np.arange(len(SCALE_ORDER))
    x_labels = [SCALE_LABELS[s] for s in SCALE_ORDER]

    for metric, ylabel, fname, log_y in [
        ("hv", "Hypervolume (HV)", "scale_trend_hv_20260705", False),
        ("igd", "IGD", "scale_trend_igd_20260705", False),
        ("seconds", "Runtime (s)", "scale_trend_runtime_20260705", True),
    ]:
        fig, ax = plt.subplots(figsize=(4.2, 2.8))
        for m in METHOD_ORDER:
            sub = df[df["method_label"] == m].sort_values("scale_order")
            if sub.empty:
                continue
            xi = sub["scale_order"].values
            yi = sub[metric].values
            lw = 1.8 if m == "KDPCS" else 1.1
            ms = 5.5 if m == "KDPCS" else 4.0
            ax.plot(xi, yi, color=COLORS[m], lw=lw, marker=MARKERS[m],
                    markersize=ms, markerfacecolor="white" if m != "KDPCS"
                    else COLORS[m],
                    markeredgecolor=COLORS[m], markeredgewidth=1.3,
                    label=m, zorder=6 if m == "KDPCS" else 4)
        if log_y:
            ax.set_yscale("log")
            ax.grid(which="minor", linestyle=":", linewidth=0.3, alpha=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=55, ha="right", fontsize=7.5)
        ax.set_ylabel(ylabel, fontsize=9.5)
        ax.tick_params(labelsize=8)
        ax.set_xlim(-0.5, len(SCALE_ORDER) - 0.5)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15),
                  ncol=6, fontsize=7.8, handlelength=1.5,
                  columnspacing=1.0, borderpad=0.3)
        fig.tight_layout()
        save_fig(fig, fname)


# =============================================================================
# Fig 7: Pareto fronts (9 PDFs)
# =============================================================================
def pareto_front_max(f1, f2):
    """Non-dominated filter for maximization. Returns boolean mask."""
    n = len(f1)
    is_nd = np.ones(n, dtype=bool)
    f1 = np.asarray(f1, dtype=float)
    f2 = np.asarray(f2, dtype=float)
    for i in range(n):
        if not is_nd[i]:
            continue
        # j dominates i (max) if f1[j]>=f1[i] and f2[j]>=f2[i]
        # and at least one strictly greater
        dominates = (f1 >= f1[i]) & (f2 >= f2[i]) & \
                    ((f1 > f1[i]) | (f2 > f2[i]))
        dominates[i] = False
        if np.any(dominates):
            is_nd[i] = False
    return is_nd


def plot_pareto_fronts():
    print("\n=== Fig 7: Pareto fronts ===")
    # Load and combine all front data
    df_kdpcs = pd.read_csv(KDPCS_FRONTS)
    df_base1 = pd.read_csv(BASELINE_FRONTS_1)
    df_base2 = pd.read_csv(BASELINE_FRONTS_2)
    df_all = pd.concat([df_kdpcs, df_base1, df_base2], ignore_index=True)
    # Use front_type == 'all' (all solution points, before ND filtering)
    df_all = df_all[df_all["front_type"] == "all"].copy()
    df_all["method_label"] = df_all["method"].map(METHOD_MAP)

    for scale in PARETO_SCALES:
        label = SCALE_LABELS[scale]
        print(f"  plotting {scale} -> {label}")
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        scale_data = df_all[df_all["scale"] == scale]

        for m in METHOD_ORDER:
            sub = scale_data[scale_data["method_label"] == m]
            if sub.empty:
                continue
            f1 = sub["f1"].values
            f2 = sub["f2"].values
            # Pool all 10 instances, then ND filter
            nd_mask = pareto_front_max(f1, f2)
            f1_nd = f1[nd_mask]
            f2_nd = f2[nd_mask]
            # Sort by f1 for clean line
            sort_idx = np.argsort(f1_nd)
            f1_nd = f1_nd[sort_idx]
            f2_nd = f2_nd[sort_idx]

            ms = 4.0 if m == "KDPCS" else 3.0
            lw = 1.3 if m == "KDPCS" else 0.8
            alpha = 0.95 if m == "KDPCS" else 0.75
            ax.plot(f1_nd, f2_nd, color=COLORS[m], lw=lw,
                    marker=MARKERS[m], markersize=ms,
                    markerfacecolor=COLORS[m] if m == "KDPCS" else "white",
                    markeredgecolor=COLORS[m], markeredgewidth=0.8,
                    label=m, zorder=6 if m == "KDPCS" else 4, alpha=alpha)

        ax.set_xlabel(r"$f_1$ (emergency gain)", fontsize=9.5)
        ax.set_ylabel(r"$f_2$ (plan stability)", fontsize=9.5)
        ax.tick_params(labelsize=8)
        ax.legend(loc="lower left", fontsize=7.5, ncol=2,
                  borderpad=0.3, handlelength=1.2, columnspacing=0.8,
                  labelspacing=0.25)
        fig.tight_layout()
        fname = f"pareto_{scale}"
        save_fig(fig, fname, subdir="pareto_final_all18_full_20260705")


# =============================================================================
# Fig 8: Preference-count sensitivity (HV, Runtime)
# =============================================================================
def plot_preference_sensitivity():
    print("\n=== Fig 8: Preference sensitivity ===")
    summary = pd.read_csv(PREF_SUMMARY)
    rows = pd.read_csv(PREF_ROWS)
    K_values = [20, 50, 100, 150, 200]

    # --- HV sensitivity ---
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    sub_all = summary[summary["scale_group"] == "all"].sort_values("num_prefs")
    K = sub_all["num_prefs"].values
    mean = sub_all["avg_hv"].values
    std = sub_all["std_hv"].values
    n = sub_all["records"].values
    sem = std / np.sqrt(n)
    # error band
    ax.fill_between(K, mean - sem, mean + sem, color=COLORS["KDPCS"],
                    alpha=0.15, linewidth=0, zorder=2)
    # mean curve
    ax.plot(K, mean, color=COLORS["KDPCS"], lw=1.8, marker="o",
            markersize=5.5, markerfacecolor="white",
            markeredgecolor=COLORS["KDPCS"], markeredgewidth=1.4,
            zorder=5, label="Mean $\pm$ SEM")
    # K=50 reference line
    ax.axvline(50, color="#888888", lw=0.8, linestyle=(0, (4, 3)),
               alpha=0.7, zorder=1)
    ax.set_xlabel(r"Preference count $K$", fontsize=9.5)
    ax.set_ylabel("Hypervolume (HV)", fontsize=9.5)
    ax.set_xticks(K_values)
    ax.tick_params(labelsize=8)
    ax.set_xlim(10, 210)
    # Set y-range to show variation without exaggeration
    y_min = mean.min() - 3 * sem[mean.argmin()]
    y_max = mean.max() + 3 * sem[mean.argmax()]
    margin = (y_max - y_min) * 0.15
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.legend(loc="lower right", fontsize=8, borderpad=0.3,
              handlelength=1.8)
    fig.tight_layout()
    save_fig(fig, "preference_hv_sensitivity")

    # --- Runtime sensitivity ---
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    mean_rt = sub_all["avg_seconds"].values
    std_rt = sub_all["std_seconds"].values
    sem_rt = std_rt / np.sqrt(n)
    ax.fill_between(K, mean_rt - sem_rt, mean_rt + sem_rt,
                    color=COLORS["KDPCS"], alpha=0.15, linewidth=0,
                    zorder=2)
    ax.plot(K, mean_rt, color=COLORS["KDPCS"], lw=1.8, marker="o",
            markersize=5.5, markerfacecolor="white",
            markeredgecolor=COLORS["KDPCS"], markeredgewidth=1.4,
            zorder=5, label="Mean $\pm$ SEM")
    ax.axvline(50, color="#888888", lw=0.8, linestyle=(0, (4, 3)),
               alpha=0.7, zorder=1)
    ax.set_xlabel(r"Preference count $K$", fontsize=9.5)
    ax.set_ylabel("Runtime (s)", fontsize=9.5)
    ax.set_xticks(K_values)
    ax.tick_params(labelsize=8)
    ax.set_xlim(10, 210)
    ax.legend(loc="upper left", fontsize=8, borderpad=0.3,
              handlelength=1.8)
    fig.tight_layout()
    save_fig(fig, "preference_runtime_sensitivity")


# =============================================================================
# Fig 9: Complexity validation
# =============================================================================
def plot_complexity():
    print("\n=== Fig 9: Complexity validation ===")
    df = pd.read_csv(COMPLEXITY_CSV)
    # Convert candidate_pairs to millions
    df["C_million"] = df["candidate_pairs"] / 1e6
    # Sort by C
    df = df.sort_values("C_million").reset_index(drop=True)

    # Aggregate per scale
    agg = df.groupby("scale").agg(
        C_mean=("C_million", "mean"),
        C_std=("C_million", "std"),
        t_mean=("seconds", "mean"),
        t_std=("seconds", "std"),
        group=("group", "first"),
        satellites=("satellites", "first"),
        tasks=("tasks", "first"),
    ).reset_index()
    agg = agg.sort_values("C_mean").reset_index(drop=True)

    group_colors = {
        "Small": "#5a9a6a",
        "Medium": "#1f3b73",
        "Large": "#c0413b",
    }

    fig, ax = plt.subplots(figsize=(4.0, 2.8))

    # Scatter individual points (light)
    for grp, color in group_colors.items():
        sub = df[df["group"] == grp]
        ax.scatter(sub["C_million"], sub["seconds"], color=color,
                   s=14, alpha=0.25, linewidths=0, zorder=2)

    # Bar chart: mean per scale with error bars
    bar_width = 0.6
    x_positions = np.arange(len(agg))
    for i, row in agg.iterrows():
        color = group_colors.get(row["group"], "#888888")
        ax.bar(i, row["t_mean"], width=bar_width,
               color=color, edgecolor="white", linewidth=0.4,
               alpha=0.6, zorder=3)
        # error bar
        y_lo = max(0, row["t_mean"] - row["t_std"])
        y_hi = row["t_mean"] + row["t_std"]
        ax.plot([i, i], [y_lo, y_hi], color="#333333", lw=0.8,
                solid_capstyle="round", zorder=5)
        ax.plot([i - 0.12, i + 0.12], [y_hi, y_hi],
                color="#333333", lw=0.8, solid_capstyle="round", zorder=5)
        ax.plot([i - 0.12, i + 0.12], [y_lo, y_lo],
                color="#333333", lw=0.8, solid_capstyle="round", zorder=5)

    # Linear fit on all individual points (real C vs runtime)
    C_all = df["C_million"].values
    t_all = df["seconds"].values
    coeff = np.polyfit(C_all, t_all, 1)
    C_fit = np.linspace(0, agg["C_mean"].max() * 1.05, 100)
    t_fit = np.polyval(coeff, C_fit)
    # Map fit line to x positions (use scale order)
    ax.plot(x_positions,
            np.polyval(coeff, agg["C_mean"].values),
            color="#c0413b", lw=1.4, linestyle="--",
            marker="D", markersize=4, markerfacecolor="white",
            markeredgecolor="#c0413b", markeredgewidth=0.9,
            zorder=7, label="Linear fit")

    # X-axis labels: scale labels
    scale_labels = [SCALE_LABELS.get(s, s) for s in agg["scale"]]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(scale_labels, rotation=55, ha="right", fontsize=7.0)
    ax.set_xlabel(r"Candidate pairs $C$ (million)", fontsize=9.5)
    ax.set_ylabel("Runtime (s)", fontsize=9.5)
    ax.tick_params(labelsize=8)
    ax.set_xlim(-0.7, len(agg) - 0.3)

    # Legend: group colors + fit
    legend_handles = [
        Patch(facecolor=group_colors["Small"], edgecolor="white",
              alpha=0.6, label="Small"),
        Patch(facecolor=group_colors["Medium"], edgecolor="white",
              alpha=0.6, label="Medium"),
        Patch(facecolor=group_colors["Large"], edgecolor="white",
              alpha=0.6, label="Large"),
        Line2D([0], [0], color="#c0413b", lw=1.4, linestyle="--",
               marker="D", markersize=4, markerfacecolor="white",
               markeredgecolor="#c0413b", markeredgewidth=0.9,
               label="Linear fit"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=7.5,
              borderpad=0.3, handlelength=1.5, labelspacing=0.3)
    fig.tight_layout()
    save_fig(fig, "kdpcs_complexity_validation_20260705")


# =============================================================================
# Run all
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Redrawing all experimental figures")
    print("=" * 60)
    plot_scale_trends()
    plot_pareto_fronts()
    plot_preference_sensitivity()
    plot_complexity()
    print("\n" + "=" * 60)
    print("All figures generated.")
    print("=" * 60)
