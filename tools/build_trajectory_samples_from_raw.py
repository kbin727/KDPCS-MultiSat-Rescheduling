# -*- coding: utf-8 -*-
"""Convert released Gurobi teacher records into trajectory-label samples.

The output is a JSONL file. Each line contains one preference-conditioned
teacher trajectory reconstructed from the stored Gurobi schedule:

    {
      "source_file": "inst_0000.pkl",
      "preference": [w1, w2],
      "f1": ...,
      "f2": ...,
      "trajectory": [
        {"step": 0, "satellite": 1, "task": 7},
        ...
      ]
    }

This script only replays teacher schedules and exports trajectory labels. It
does not contain the KDPCS policy network or training code.
"""

import argparse
import json
import os
import pickle
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from environment import (  # noqa: E402
    compute_sat_attitude_transition,
    compute_task_energy,
    get_sat_energy_capacity,
    get_sat_memory_capacity,
)


def _load_record(path):
    with open(path, "rb") as f:
        record = pickle.load(f)
    if "instance_data" not in record or "raw_solutions" not in record:
        raise ValueError(f"Invalid teacher record: {path}")
    instance = pickle.loads(record["instance_data"])
    return record, instance


def _normalize_sequences(sequences):
    out = {}
    for key, value in (sequences or {}).items():
        out[int(key)] = [int(v) for v in value]
    return out


def extract_trajectory(instance, sequences):
    """Replay a multi-satellite sequence using the released feasibility model."""
    tasks = instance.tasks
    num_sats = int(instance.num_satellites)
    sequences = _normalize_sequences(sequences)

    positions = {s: 0 for s in range(num_sats)}
    sat_times = {s: 0.0 for s in range(num_sats)}
    sat_rolls = {s: 0.0 for s in range(num_sats)}
    sat_pitches = {s: 0.0 for s in range(num_sats)}
    sat_memory = {s: get_sat_memory_capacity(instance, s) for s in range(num_sats)}
    sat_energy = {s: get_sat_energy_capacity(instance, s) for s in range(num_sats)}
    scheduled = set()
    trajectory = []

    while True:
        best = None
        best_start = float("inf")

        for sat_id in range(num_sats):
            seq = sequences.get(sat_id, [])
            while positions[sat_id] < len(seq) and int(seq[positions[sat_id]]) in scheduled:
                positions[sat_id] += 1
            if positions[sat_id] >= len(seq):
                continue

            task_idx = int(seq[positions[sat_id]])
            task = tasks[task_idx]
            if sat_id not in task.get("satellite_vtw", {}):
                positions[sat_id] += 1
                continue

            vtw = task["satellite_vtw"][sat_id]
            trans_time = compute_sat_attitude_transition(
                instance,
                sat_id,
                sat_rolls[sat_id],
                sat_pitches[sat_id],
                vtw["roll"],
                vtw["pitch"],
            )
            start_time = max(sat_times[sat_id] + trans_time, float(vtw["vtw_start"]))
            if start_time + float(task["duration"]) > float(vtw["vtw_end"]):
                positions[sat_id] += 1
                continue
            if float(task["memory"]) > sat_memory[sat_id]:
                positions[sat_id] += 1
                continue
            energy = compute_task_energy(instance, sat_id, trans_time, task["duration"])
            if energy > sat_energy[sat_id]:
                positions[sat_id] += 1
                continue

            if start_time < best_start:
                best_start = start_time
                best = (sat_id, task_idx, trans_time, start_time)

        if best is None:
            break

        sat_id, task_idx, trans_time, start_time = best
        task = tasks[task_idx]
        vtw = task["satellite_vtw"][sat_id]
        duration = float(task["duration"])
        energy = compute_task_energy(instance, sat_id, trans_time, duration)

        trajectory.append({
            "step": len(trajectory),
            "satellite": int(sat_id),
            "task": int(task_idx),
            "start_time": float(start_time),
            "end_time": float(start_time + duration),
        })
        scheduled.add(task_idx)
        sat_times[sat_id] = start_time + duration
        sat_rolls[sat_id] = float(vtw["roll"])
        sat_pitches[sat_id] = float(vtw["pitch"])
        sat_memory[sat_id] -= float(task["memory"])
        sat_energy[sat_id] -= energy
        positions[sat_id] += 1

    return trajectory


def iter_teacher_solutions(record, expand_to_record_prefs=False):
    raw_solutions = record.get("raw_solutions", [])
    if not expand_to_record_prefs:
        for sol in raw_solutions:
            yield sol
        return

    prefs = record.get("prefs")
    if not prefs:
        raise ValueError("Record has no stored prefs for --expand-to-record-prefs")
    for pref in prefs:
        w1, w2 = float(pref[0]), float(pref[1])
        teacher = max(raw_solutions, key=lambda sol: w1 * float(sol["f1"]) + w2 * float(sol["f2"]))
        sol = dict(teacher)
        sol["preference"] = [w1, w2]
        yield sol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, help="Directory containing inst_*.pkl teacher records.")
    parser.add_argument("--output", required=True, help="Output JSONL file.")
    parser.add_argument("--expand-to-record-prefs", action="store_true")
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    files = sorted(
        os.path.join(args.raw_dir, name)
        for name in os.listdir(args.raw_dir)
        if name.startswith("inst_") and name.endswith(".pkl")
    )
    if args.max_records is not None:
        files = files[: args.max_records]
    if not files:
        raise FileNotFoundError(args.raw_dir)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    num_rows = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for path in files:
            record, instance = _load_record(path)
            for sol in iter_teacher_solutions(record, args.expand_to_record_prefs):
                trajectory = extract_trajectory(instance, sol.get("sequences", {}))
                row = {
                    "source_file": os.path.basename(path),
                    "num_satellites": int(instance.num_satellites),
                    "num_tasks": int(len(instance.tasks)),
                    "preference": [float(v) for v in sol["preference"]],
                    "f1": float(sol["f1"]),
                    "f2": float(sol["f2"]),
                    "trajectory": trajectory,
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                num_rows += 1

    print(f"Saved {num_rows} trajectory samples to {args.output}")


if __name__ == "__main__":
    main()

