# Experimental Records

This directory contains the final numerical records used by the paper. It is
organized by evaluation purpose rather than by temporary experiment folder
names.

| Directory | Contents |
|---|---|
| `all_scale_comparison/` | Final 18-scale comparison among KDPCS, MONP, NSGA-II, SPEA2, IBEA, and MOEA/D. Files include per-scale method rows, method summaries, scale-trend data, and KDPCS complexity-validation data. |
| `kdpcs_policy_outputs/` | Final KDPCS policy-only outputs on all 18 benchmark settings with 50 preferences, plus the KDPCS non-dominated fronts used for Pareto-front plots. |
| `ablation/` | Final method-level ablation rows and summary results. |
| `preference_sensitivity/` | Final preference-count sensitivity, dense preference-response, and representative schedule data used by the preference-analysis figures. |

Notes:

- Method names in the released CSV/JSON files follow the paper notation:
  KDPCS, MONP, NSGA-II, SPEA2, IBEA, and MOEA/D.
- Intermediate logs, checkpoint JSONL files, and duplicate figure exports are
  intentionally excluded from this release.
- Figure files used in the paper are stored separately under `paper_figures/`.
