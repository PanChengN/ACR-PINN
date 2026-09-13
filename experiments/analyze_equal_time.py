#!/usr/bin/env python
"""Audit and aggregate the Helmholtz (4,4) equal-time study."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "equal_time.json"
SEEDS = tuple(range(19018, 19023))


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"No rows available for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    problem = config["problems"]["helmholtz_4_4"]
    budget = float(problem["environment"]["WALL_TIME_BUDGET_SEC"])
    threshold = float(problem["environment"]["ERROR_THRESHOLD_L2"])
    rows = []
    problems = []
    hashes_by_seed = {seed: set() for seed in SEEDS}

    for method_key, condition in config["conditions"].items():
        for seed in SEEDS:
            tag = f"equal_time_helmholtz_4_4_{method_key}_seed{seed}"
            log_dir = args.root / f"results_helmholtz_a1_4.0_a2_4.0_{tag}" / "runs" / "run_1" / "logs"
            metrics_path = log_dir / "wall_time_metrics.json"
            metadata_path = log_dir / "run_metadata.json"
            if not metrics_path.is_file() or not metadata_path.is_file():
                problems.append(f"missing outputs: {tag}")
                continue
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            seeds = metadata["seeds"]
            data_hash = metadata["dataset_sha256"]
            hashes_by_seed[seed].add(data_hash)
            expected = {"sample_seed": seed + 1000, "init_seed": seed, "optimizer_order_seed": seed + 2000}
            if any(int(seeds[key]) != value for key, value in expected.items()):
                problems.append(f"seed mismatch: {tag}")
            observed = float(metrics["observed_training_time_sec"])
            if not metrics["stopped_by_wall_time"]:
                problems.append(f"did not stop by wall time: {tag}")
            if observed < budget or observed > budget + 10.0:
                problems.append(f"wall-time tolerance failed: {tag} ({observed:.3f}s)")
            if float(metrics["relative_l2_threshold"]) != threshold:
                problems.append(f"threshold mismatch: {tag}")
            rows.append({
                "problem": "Helmholtz (4,4)",
                "method_key": method_key,
                "architecture": condition["architecture"],
                "optimizer": condition["optimizer"],
                "master_seed": seed,
                "sample_seed": seed + 1000,
                "optimizer_order_seed": seed + 2000,
                "dataset_sha256": data_hash,
                "wall_time_budget_sec": budget,
                "observed_training_time_sec": observed,
                "completed_iterations": int(metrics["completed_iterations"]),
                "final_relative_l2": float(metrics["final_relative_l2"]),
                "final_relative_linf": float(metrics["final_relative_linf"]),
                "final_mse": float(metrics["final_mse"]),
                "threshold_relative_l2": threshold,
                "threshold_reached": bool(metrics["threshold_reached"]),
                "time_to_threshold_sec": metrics["time_to_threshold_sec"],
                "threshold_iteration": metrics["threshold_iteration"],
            })

    for seed, hashes in hashes_by_seed.items():
        if len(hashes) != 1:
            problems.append(f"paired dataset hash mismatch for seed {seed}: {sorted(hashes)}")

    summary = []
    for method_key in config["conditions"]:
        subset = [row for row in rows if row["method_key"] == method_key]
        if len(subset) != len(SEEDS):
            continue
        reached = [float(row["time_to_threshold_sec"]) for row in subset if row["threshold_reached"]]
        summary.append({
            "method_key": method_key,
            "n": len(subset),
            "wall_time_sec_mean": np.mean([row["observed_training_time_sec"] for row in subset]),
            "wall_time_sec_std": np.std([row["observed_training_time_sec"] for row in subset], ddof=1),
            "iterations_mean": np.mean([row["completed_iterations"] for row in subset]),
            "iterations_std": np.std([row["completed_iterations"] for row in subset], ddof=1),
            "fixed_time_l2_mean": np.mean([row["final_relative_l2"] for row in subset]),
            "fixed_time_l2_std": np.std([row["final_relative_l2"] for row in subset], ddof=1),
            "fixed_time_linf_mean": np.mean([row["final_relative_linf"] for row in subset]),
            "fixed_time_mse_mean": np.mean([row["final_mse"] for row in subset]),
            "threshold_reached_n": len(reached),
            "time_to_threshold_sec_median_reached": np.median(reached) if reached else "",
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        write_csv(args.output_dir / "run_level.csv", rows)
    if summary:
        write_csv(args.output_dir / "summary.csv", summary)
    audit = {
        "passed": not problems and len(rows) == 25 and len(summary) == 5,
        "expected_jobs": 25,
        "completed_jobs": len(rows),
        "methods": len(summary),
        "paired_seeds": list(SEEDS),
        "wall_time_budget_sec": budget,
        "relative_l2_threshold": threshold,
        "problems": problems,
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
