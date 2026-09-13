#!/usr/bin/env python
"""Run one GPU shard of the Helmholtz (4,4) equal-time study."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "equal_time.json"


def jobs(config, seeds):
    tag_prefix = config["job_tag_prefix"]
    for method_key, condition in config["conditions"].items():
        for master_seed in seeds:
            yield {
                "method_key": method_key,
                "architecture": condition["architecture"],
                "optimizer": condition["optimizer"],
                "master_seed": master_seed,
                "result_tag": f"{tag_prefix}_{method_key}_seed{master_seed}",
            }


def result_path(job):
    return (
        ROOT
        / f"results_helmholtz_a1_4.0_a2_4.0_{job['result_tag']}"
        / "runs"
        / "run_1"
        / "logs"
        / "wall_time_metrics.json"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cuda-device", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--seed-start", type=int, default=19018)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must satisfy 0 <= index < shard-count")

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seeds = tuple(range(args.seed_start, args.seed_start + args.seed_count))
    assigned = [job for index, job in enumerate(jobs(config, seeds)) if index % args.shard_count == args.shard_index]
    print(json.dumps({
        "experiment_id": config["experiment_id"],
        "cuda_device": args.cuda_device,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "jobs": len(assigned),
        "wall_time_budget_sec": config["problems"]["helmholtz_4_4"]["environment"]["WALL_TIME_BUDGET_SEC"],
    }, indent=2), flush=True)

    for index, job in enumerate(assigned, start=1):
        if result_path(job).is_file():
            print(f"[{index}/{len(assigned)}] {job['result_tag']} (already complete)", flush=True)
            continue
        command = [
            sys.executable,
            str(ROOT / "experiments" / "run_experiment.py"),
            "--config", str(config_path),
            "--problem", "helmholtz_4_4",
            "--architecture", job["architecture"],
            "--optimizer", job["optimizer"],
            "--num-runs", "1",
            "--result-tag", job["result_tag"],
            "--master-seed-base", str(job["master_seed"]),
            "--sample-seed-base", str(job["master_seed"] + 1000),
            "--init-seed-base", str(job["master_seed"]),
            "--optimizer-order-seed-base", str(job["master_seed"] + 2000),
            "--cuda-device", args.cuda_device,
        ]
        print(f"[{index}/{len(assigned)}] {job['result_tag']}", flush=True)
        if args.dry_run:
            print(" ".join(command), flush=True)
        else:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
