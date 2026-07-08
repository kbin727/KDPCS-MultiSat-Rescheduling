# Data Conversion Tools

`build_trajectory_samples_from_raw.py` converts released Gurobi/MIP teacher
records into preference-conditioned trajectory labels. It is intended for data
reuse and reproducibility. It does not include the KDPCS policy network,
training code, or checkpoints.

The `figure_generation/` subdirectory archives the plotting scripts used to
generate the final paper figures from the released experimental records.  These
scripts are provided for transparency of the reported plots; they are not
required for loading the benchmark instances or teacher data.

Example:

```bash
python tools/build_trajectory_samples_from_raw.py \
  --raw-dir teacher_split_p50/train_raw \
  --output converted_samples/train_trajectory_samples.jsonl \
  --expand-to-record-prefs
```

Use `--expand-to-record-prefs` to emit one trajectory label for every stored
preference direction in each raw record.

Benchmark instance files use the same serialized `instance_data` convention.
Load a benchmark record with `pickle.load`, then call
`pickle.loads(record["instance_data"])` to recover the scheduling instance.
