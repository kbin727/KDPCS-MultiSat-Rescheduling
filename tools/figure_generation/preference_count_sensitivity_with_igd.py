# -*- coding: utf-8 -*-
"""Evaluate preference-count sensitivity and save HV, IGD, runtime, and fronts.

IGD is computed per instance using the K=200 policy front as the reference
front, which makes the sensitivity analysis independent of scale difficulty.
"""

import argparse
import csv
import glob
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_pcformer_benchmark import rollout_batched_preferences
from mo_vector_policy import MultiSatVectorActionPolicy
from train import compute_hv, generate_dirichlet_endpoint_prefs, generate_uniform_prefs
from tools.compare_policy_moea_gurobi_raw import _nondominated_points


def _idx(path):
    return int(os.path.basename(path)[5:-4])


def _load_instance(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return pickle.loads(payload["instance_data"])


def _pview(args):
    return SimpleNamespace(
        max_steps=args.policy_max_steps,
        stop_logit_threshold=-1e8,
        decode_gain_bias=args.decode_gain_bias,
    )


def _igd(points, reference):
    if not points or not reference:
        return float("inf")
    pts = np.asarray(points, dtype=float)
    ref = np.asarray(reference, dtype=float)
    dists = []
    for r in ref:
        diff = pts - r
        dists.append(float(np.sqrt(np.sum(diff * diff, axis=1)).min()))
    return float(np.mean(dists))


def _scale_group(scale):
    m = int(scale.split("_", 1)[0][1:])
    if m <= 5:
        return "small"
    if m <= 10:
        return "medium"
    return "large"


def _prefs(num_prefs, args, seed):
    if args.pref_mode == "uniform":
        return generate_uniform_prefs(num_prefs)
    return generate_dirichlet_endpoint_prefs(num_prefs, alpha=args.dirichlet_alpha, seed=seed)


@torch.no_grad()
def _rollout_policy(model, instance, prefs, args):
    batch_size = int(args.policy_pref_batch_size or 0)
    if batch_size <= 0 or batch_size >= len(prefs):
        sols, seconds = rollout_batched_preferences(model, instance, prefs, _pview(args))
    else:
        sols = []
        t0 = time.time()
        for start in range(0, len(prefs), batch_size):
            part, _ = rollout_batched_preferences(
                model,
                instance,
                prefs[start:start + batch_size],
                _pview(args),
            )
            sols.extend(part)
        seconds = time.time() - t0
    points = [(float(s["f1"]), float(s["f2"])) for s in sols]
    scheduled = [float(s.get("scheduled", 0.0)) for s in sols]
    return {
        "points": points,
        "nd_points": _nondominated_points(points),
        "seconds": float(seconds),
        "avg_scheduled": float(np.mean(scheduled)) if scheduled else 0.0,
    }


def _write_outputs(out_dir, rows, fronts):
    out_dir.mkdir(parents=True, exist_ok=True)
    row_csv = out_dir / "preference_count_sensitivity_rows.csv"
    fields = [
        "num_prefs", "scale", "scale_group", "instance_idx", "seed",
        "satellites", "tasks", "hv", "normalized_hv", "igd_to_k200",
        "nd_points", "seconds", "seconds_per_pref", "avg_scheduled",
    ]
    with row_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})

    summary = []
    for num_prefs in sorted({r["num_prefs"] for r in rows}):
        subset_k = [r for r in rows if r["num_prefs"] == num_prefs]
        for group in ["small", "medium", "large", "all"]:
            subset = subset_k if group == "all" else [r for r in subset_k if r["scale_group"] == group]
            if not subset:
                continue
            hv = np.asarray([r["hv"] for r in subset], dtype=float)
            nhv = np.asarray([r["normalized_hv"] for r in subset], dtype=float)
            igd = np.asarray([r["igd_to_k200"] for r in subset], dtype=float)
            sec = np.asarray([r["seconds"] for r in subset], dtype=float)
            nd = np.asarray([r["nd_points"] for r in subset], dtype=float)
            summary.append({
                "num_prefs": num_prefs,
                "scale_group": group,
                "records": len(subset),
                "avg_hv": float(hv.mean()),
                "std_hv": float(hv.std(ddof=1)) if len(hv) > 1 else 0.0,
                "avg_normalized_hv": float(nhv.mean()),
                "std_normalized_hv": float(nhv.std(ddof=1)) if len(nhv) > 1 else 0.0,
                "avg_igd_to_k200": float(igd.mean()),
                "std_igd_to_k200": float(igd.std(ddof=1)) if len(igd) > 1 else 0.0,
                "avg_nd_points": float(nd.mean()),
                "avg_seconds": float(sec.mean()),
                "std_seconds": float(sec.std(ddof=1)) if len(sec) > 1 else 0.0,
                "avg_seconds_per_pref": float((sec / max(num_prefs, 1)).mean()),
            })
    summary_csv = out_dir / "preference_count_sensitivity_summary.csv"
    fields = [
        "num_prefs", "scale_group", "records", "avg_hv", "std_hv",
        "avg_normalized_hv", "std_normalized_hv", "avg_igd_to_k200",
        "std_igd_to_k200", "avg_nd_points", "avg_seconds",
        "std_seconds", "avg_seconds_per_pref",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)

    fronts_csv = out_dir / "preference_count_sensitivity_fronts.csv"
    with fronts_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "num_prefs", "scale", "instance_idx", "seed",
                "front_type", "point_idx", "f1", "f2",
            ],
        )
        writer.writeheader()
        for item in fronts:
            for front_type, pts in [("all", item["points"]), ("nd", item["nd_points"])]:
                for pidx, (f1, f2) in enumerate(pts):
                    writer.writerow({
                        "num_prefs": item["num_prefs"],
                        "scale": item["scale"],
                        "instance_idx": item["instance_idx"],
                        "seed": item["seed"],
                        "front_type": front_type,
                        "point_idx": pidx,
                        "f1": f1,
                        "f2": f2,
                    })

    with (out_dir / "preference_count_sensitivity.json").open("w", encoding="utf-8") as f:
        json.dump({"rows": rows, "summary": summary, "fronts": fronts}, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-root", required=True)
    parser.add_argument("--policy-model", required=True)
    parser.add_argument("--policy-ablation", default="final_pair_self_cross")
    parser.add_argument("--preference-counts", nargs="+", type=int, default=[20, 50, 100, 150, 200])
    parser.add_argument("--pref-mode", choices=["uniform", "dirichlet_endpoint"], default="uniform")
    parser.add_argument("--dirichlet-alpha", type=float, default=1.0)
    parser.add_argument("--max-records-per-scale", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--policy-pref-batch-size", type=int, default=10)
    parser.add_argument("--policy-max-steps", type=int, default=5000)
    parser.add_argument("--decode-gain-bias", type=float, default=0.0)
    parser.add_argument("--scales", nargs="*", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiSatVectorActionPolicy(ablation=args.policy_ablation)
    model.load_state_dict(torch.load(args.policy_model, map_location="cpu"), strict=True)
    model.eval().to(device)

    out_dir = Path(args.output_dir)
    rows = []
    fronts = []
    done = set()
    if args.resume:
        row_csv = out_dir / "preference_count_sensitivity_rows.csv"
        fronts_csv = out_dir / "preference_count_sensitivity_fronts.csv"
        if row_csv.exists():
            with row_csv.open("r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    parsed = {
                        "num_prefs": int(row["num_prefs"]),
                        "scale": row["scale"],
                        "scale_group": row["scale_group"],
                        "instance_idx": int(row["instance_idx"]),
                        "seed": int(row["seed"]),
                        "satellites": int(row["satellites"]),
                        "tasks": int(row["tasks"]),
                        "hv": float(row["hv"]),
                        "normalized_hv": float(row["normalized_hv"]),
                        "igd_to_k200": float(row["igd_to_k200"]),
                        "nd_points": int(row["nd_points"]),
                        "seconds": float(row["seconds"]),
                        "seconds_per_pref": float(row["seconds_per_pref"]),
                        "avg_scheduled": float(row["avg_scheduled"]),
                    }
                    rows.append(parsed)
            counts_by_instance = {}
            for row in rows:
                key = (row["scale"], row["instance_idx"])
                counts_by_instance.setdefault(key, set()).add(row["num_prefs"])
            expected = set(args.preference_counts)
            done = {key for key, values in counts_by_instance.items() if values >= expected}
        if fronts_csv.exists():
            grouped = {}
            with fronts_csv.open("r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    key = (
                        int(row["num_prefs"]),
                        row["scale"],
                        int(row["instance_idx"]),
                        int(row["seed"]),
                    )
                    item = grouped.setdefault(key, {"points": [], "nd_points": []})
                    target = item["nd_points"] if row["front_type"] == "nd" else item["points"]
                    target.append((float(row["f1"]), float(row["f2"])))
            for (num_prefs, scale, instance_idx, seed), item in grouped.items():
                fronts.append({
                    "num_prefs": num_prefs,
                    "scale": scale,
                    "instance_idx": instance_idx,
                    "seed": seed,
                    "points": item["points"],
                    "nd_points": item["nd_points"],
                })
    print("Preference-count sensitivity with IGD")
    print("device:", device)
    print("scales:", args.scales)
    print("preference counts:", args.preference_counts, flush=True)

    for scale in args.scales:
        folder = Path(args.instance_root) / scale
        files = sorted(glob.glob(str(folder / "inst_*.pkl")), key=_idx)[: args.max_records_per_scale]
        if not files:
            raise FileNotFoundError(folder)
        for path in files:
            instance = _load_instance(path)
            instance_idx = _idx(path)
            if (scale, instance_idx) in done:
                print(f"{scale} inst={instance_idx}: resume skip", flush=True)
                continue
            seed = int(args.seed) + int(instance_idx)
            per_k = {}
            for num_prefs in args.preference_counts:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                prefs = _prefs(num_prefs, args, seed)
                out = _rollout_policy(model, instance, prefs, args)
                per_k[int(num_prefs)] = out
                fronts.append({
                    "num_prefs": int(num_prefs),
                    "scale": scale,
                    "instance_idx": int(instance_idx),
                    "seed": int(seed),
                    "points": out["points"],
                    "nd_points": out["nd_points"],
                })
                print(
                    f"{scale} inst={instance_idx} K={num_prefs}: "
                    f"HV={compute_hv(out['points']):.6f} ND={len(out['nd_points'])} "
                    f"time={out['seconds']:.2f}s",
                    flush=True,
                )

            ref_k = max(args.preference_counts)
            ref_front = per_k[ref_k]["nd_points"]
            ref_hv = float(compute_hv(per_k[ref_k]["points"]))
            for num_prefs in args.preference_counts:
                out = per_k[int(num_prefs)]
                hv = float(compute_hv(out["points"]))
                row = {
                    "num_prefs": int(num_prefs),
                    "scale": scale,
                    "scale_group": _scale_group(scale),
                    "instance_idx": int(instance_idx),
                    "seed": int(seed),
                    "satellites": int(instance.num_satellites),
                    "tasks": int(len(instance.tasks)),
                    "hv": hv,
                    "normalized_hv": float(100.0 * hv / ref_hv) if ref_hv > 1e-12 else 0.0,
                    "igd_to_k200": _igd(out["nd_points"], ref_front),
                    "nd_points": int(len(out["nd_points"])),
                    "seconds": float(out["seconds"]),
                    "seconds_per_pref": float(out["seconds"] / max(num_prefs, 1)),
                    "avg_scheduled": float(out["avg_scheduled"]),
                }
                rows.append(row)
            _write_outputs(out_dir, rows, fronts)

    _write_outputs(out_dir, rows, fronts)
    print("Saved:", out_dir)


if __name__ == "__main__":
    main()
