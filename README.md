# KDPCS Data and Reproducibility Package

This directory contains the data artifacts used for the paper
**"KDPCS: A MIP-Guided Knowledge-Distilled Preference-Conditioned Constructive Solver for Multi-Objective Multi-Satellite Rescheduling."**

The package is organized to separate benchmark instances, teacher data,
converted samples, and final experiment records.

## Contents

| Directory | Contents |
|---|---|
| `benchmark_instances/` | Basilisk-derived benchmark instances using the paper notation `S<m>-T<N>`, e.g., `S3-T100` denotes 3 satellites and 100 tasks. There are 18 benchmark settings and 10 instances per setting. |
| `teacher_split_p50/` | Final Gurobi/MIP teacher data with 50 preference directions. `train_raw/` contains 590 training instances and `val_raw/` contains 10 validation instances used for model selection. `test_raw_30/` contains 30 additional complete teacher instances plus a manifest file; these instances are not used to train or select the final policy. |
| `converted_samples/` | Placeholder for converted trajectory samples. The full PyTorch cache is larger than 6 GB and is not stored in git; use `tools/build_trajectory_samples_from_raw.py` to regenerate trajectory-label samples from `teacher_split_p50/`. |
| `experimental_records/` | Final numerical records used by the paper, organized into all-scale comparison, final KDPCS outputs, preference-sensitivity analysis, and ablation results. |
| `paper_figures/` | Figure assets generated from the released experiment records. |
| `tools/` | Data conversion utilities for replaying released Gurobi teacher schedules into preference-conditioned trajectory labels. |

## Simulation Environment

The benchmark instances are generated from a Basilisk-based constellation
simulation following the AEOSBench-style setting. Basilisk is used to propagate
the sampled low-earth-orbit satellites and compute satellite-target visibility
windows and observation attitudes. The released scheduling records then use
these Basilisk-derived windows, angles, task attributes, and resource
parameters.

## Important Data Notes

- Objective values and plots in the paper use the final KDPCS policy and the final merged evaluation records.
- In the manuscript, original-plan tasks are the tasks with `original_scheduled=True`.
  Emergency candidate tasks are the remaining tasks considered at the rescheduling
  epoch (`original_scheduled=False`). Under this paper definition, the
  benchmark emergency-candidate ratio ranges from 0.65 to 0.91 across the 180
  released instances.
- Policy-network source code and trained checkpoints are not included in this data release.
- The converted supervised sample cache is larger than 6 GB and is not tracked by git. If the binary cache is needed, publish it as a separate release asset or regenerate samples from the raw teacher records.
- Figure-generation scripts are provided under `tools/figure_generation/`; they are included to document how the released figures were produced from the released experiment records.
- The raw teacher split is included. To regenerate trajectory-label samples:

```bash
python tools/build_trajectory_samples_from_raw.py \
  --raw-dir teacher_split_p50/train_raw \
  --output converted_samples/train_trajectory_samples.jsonl \
  --expand-to-record-prefs
```

## Benchmark Naming

The paper uses the code `S<m>-T<N>` for benchmark settings:

- `S3-T100`: 3 satellites, 100 tasks.
- `S10-T400`: 10 satellites, 400 tasks.
- `S20-T800`: 20 satellites, 800 tasks.

Each benchmark setting contains ten independent Basilisk-derived instances.

## Loading Benchmark Instances

Benchmark `.pkl` files store a small metadata wrapper and a serialized
`instance_data` object:

```python
import pickle

with open("benchmark_instances/S3-T100/inst_0.pkl", "rb") as f:
    record = pickle.load(f)

instance = pickle.loads(record["instance_data"])
print(instance.num_satellites, len(instance.tasks))
```

The minimal `environment.py` included in this repository provides the class and
helper functions required to unpickle the released instances and teacher
records.
