#!/usr/bin/env python
"""Aggregate the joint and separate space--time encoding experiments."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = ('final_l2', 'final_linf', 'final_mse')


def read_summary(result_dir):
    path = result_dir / 'aggregate' / 'logs' / 'summary.csv'
    with path.open(newline='', encoding='utf-8') as handle:
        return [row for row in csv.DictReader(handle) if row.get('run', '').isdigit()]


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def result_directory(root, result_tag):
    return root / f'results_klein_gordon_{result_tag}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    run_rows = []
    for job in manifest['jobs']:
        rows = read_summary(result_directory(args.root, job['result_tag']))
        if len(rows) != 5:
            raise ValueError(f"{job['result_tag']}: expected five completed runs, found {len(rows)}")
        horizon = float(job['problem'].removeprefix('klein_gordon_t'))
        for row in rows:
            run_rows.append({
                'problem': job['problem'],
                'time_horizon': horizon,
                'architecture': job['architecture'],
                'optimizer': job['optimizer'],
                'result_tag': job['result_tag'],
                'run': int(row['run']),
                'sample_seed': int(row['sample_seed']),
                'init_seed': int(row['init_seed']),
                'optimizer_order_seed': int(row['optimizer_order_seed']),
                **{metric: float(row[metric]) for metric in METRICS},
            })

    summary_rows = []
    for job in manifest['jobs']:
        selected = [row for row in run_rows if row['result_tag'] == job['result_tag']]
        for metric in METRICS:
            values = np.asarray([row[metric] for row in selected])
            summary_rows.append({
                'problem': job['problem'],
                'architecture': job['architecture'],
                'optimizer': job['optimizer'],
                'metric': metric,
                'n': len(values),
                'mean': float(values.mean()),
                'std': float(values.std(ddof=1)),
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / 'run_level_metrics.csv', run_rows)
    write_csv(args.output_dir / 'summary.csv', summary_rows)
    audit = {
        'passed': True,
        'conditions': len(manifest['jobs']),
        'runs': len(run_rows),
        'metrics': list(METRICS),
    }
    (args.output_dir / 'audit.json').write_text(
        json.dumps(audit, indent=2) + '\n', encoding='utf-8'
    )
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
