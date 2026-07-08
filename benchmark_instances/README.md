# KDPCS Benchmark Instances

This directory contains the Basilisk-derived benchmark instances used in the
KDPCS paper.  The folder names follow the benchmark code used in the manuscript:

- `S<m>-T<N>` denotes a setting with `m` satellites and `N` tasks.
- Each setting contains 10 independent instances.
- The 18 settings are grouped into small, medium, and large regimes.

## Scale Groups

| Group | Satellites | Tasks | Settings |
|---|---:|---:|---|
| Small | 3, 5 | 100, 150, 200 | S3-T100 to S5-T200 |
| Medium | 8, 10 | 200, 300, 400 | S8-T200 to S10-T400 |
| Large | 15, 20 | 400, 600, 800 | S15-T400 to S20-T800 |

## Files

- `benchmark_manifest.csv`: machine-readable summary of all settings.
- `benchmark_manifest.json`: JSON version of the same metadata.
- `S<m>-T<N>/inst_*.pkl`: serialized benchmark instances.
- `S<m>-T<N>/manifest.json`: per-setting metadata inherited from the
  generated instance set.

## Notes

The instances are generated from the same feasibility model used in the paper:
satellite-target visibility, observation angles, task durations, priority
values, memory demand, energy budget, and original-plan indicators are included
for rescheduling experiments.

For the manuscript, an original-plan task is identified by
`original_scheduled=True`. All remaining tasks in the released rescheduling
pool are treated as emergency candidate tasks at the decision epoch. The
per-setting `manifest.json` files therefore include paper-level fields named
`paper_num_original_plan_tasks`, `paper_num_emergency_candidates`, and
`paper_emergency_candidate_ratio`. These are the fields that correspond to the
paper definition of the high-concurrency emergency setting.
