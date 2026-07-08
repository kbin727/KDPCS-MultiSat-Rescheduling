# Figure Generation Scripts

This directory archives the scripts used to generate the final paper figures
from the released experimental records.  The scripts assume the directory
layout of the original project and are included to document the plotting
pipeline.

Key inputs are stored under `experimental_records/`, especially:

- `all_scale_comparison/scale_trend_summary.csv`
- `all_scale_comparison/kdpcs_complexity_points.csv`
- `kdpcs_policy_outputs/pareto_fronts/kdpcs_policy_fronts.csv`
- `ablation/scale_ablation_rows.csv`
- `preference_sensitivity/` for the preference-count, dense-tradeoff, and
  representative-schedule figures

`final_polish_fig8_9_10_20260709.py` records the final polishing pass for the
preference-sensitivity and schedule figures used in the submitted paper package.

The final figure files copied into the paper package are stored in
`paper_figures/`.
