#!/usr/bin/env python
"""Audit and summarize task-gradient comparison against protocol-matched Std/GC references."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from summarize_paired_results import METRICS, bootstrap_interval, holm_adjust, read_runs


FOLDERS = {
    'burgers': {
        'Std-PINN': 'results_optimizer_burgers_sum',
        'GCR-PINN': 'results_optimizer_burgers_pcgrad',
        'MGDA': 'results_optimizer_burgers_mgda',
        'NTK weighting': 'results_optimizer_burgers_ntk',
        'GradNorm': 'results_optimizer_burgers_gradnorm',
    },
    'helmholtz_4_4': {
        'Std-PINN': 'results_helmholtz_a1_4.0_a2_4.0_optimizer_helmholtz_4_4_sum',
        'GCR-PINN': 'results_helmholtz_a1_4.0_a2_4.0_optimizer_helmholtz_4_4_pcgrad',
        'MGDA': 'results_helmholtz_a1_4.0_a2_4.0_optimizer_helmholtz_4_4_mgda',
        'NTK weighting': 'results_helmholtz_a1_4.0_a2_4.0_optimizer_helmholtz_4_4_ntk',
        'GradNorm': 'results_helmholtz_a1_4.0_a2_4.0_optimizer_helmholtz_4_4_gradnorm',
    },
}

EXPECTED_RUNS = 5


def seed_key(row):
    return row['sample_seed'], row['init_seed'], row['optimizer_order_seed']


def read_metadata(folder):
    records = {}
    run_dirs = sorted((folder / 'runs').glob('run_*'))
    if len(run_dirs) != EXPECTED_RUNS:
        raise ValueError(f'{folder}: expected {EXPECTED_RUNS} run directories, found {len(run_dirs)}')
    for run_dir in run_dirs:
        metadata_path = run_dir / 'logs' / 'run_metadata.json'
        checkpoint_path = run_dir / 'checkpoints' / 'model_final.pt'
        if not metadata_path.exists() or not checkpoint_path.exists():
            raise ValueError(f'{run_dir}: missing metadata or fixed-final checkpoint')
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        if metadata.get('checkpoint_policy') != 'fixed_final_iteration':
            raise ValueError(f'{run_dir}: checkpoint policy is not fixed_final_iteration')
        if metadata.get('test_metrics_used_for_selection') is not False:
            raise ValueError(f'{run_dir}: test metrics may have been used for selection')
        seeds = metadata['seeds']
        key = (int(seeds['sample_seed']), int(seeds['init_seed']), int(seeds['optimizer_order_seed']))
        if key in records:
            raise ValueError(f'{folder}: duplicate seed bundle {key}')
        records[key] = metadata
    return records


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    generator = np.random.default_rng(20260901)
    summaries, tests, audit_rows = [], [], []

    for problem, mapping in FOLDERS.items():
        data = {label: read_runs(args.root / folder) for label, folder in mapping.items()}
        metadata = {label: read_metadata(args.root / folder) for label, folder in mapping.items()}
        keys = {label: {seed_key(row) for row in rows} for label, rows in data.items()}
        reference_keys = keys['Std-PINN']
        if len(reference_keys) != EXPECTED_RUNS:
            raise ValueError(f'{problem}: expected {EXPECTED_RUNS} unique seed bundles')
        if any(value != reference_keys for value in keys.values()):
            raise ValueError(f'{problem}: summary seed bundles are not identical')
        if any(set(value) != reference_keys for value in metadata.values()):
            raise ValueError(f'{problem}: metadata seed bundles do not match summaries')

        for key in sorted(reference_keys):
            hashes = {label: records[key]['dataset_sha256'] for label, records in metadata.items()}
            if len(set(hashes.values())) != 1:
                raise ValueError(f'{problem} seed {key}: paired training-data hashes differ: {hashes}')
            audit_rows.append({
                'problem': problem,
                'sample_seed': key[0],
                'init_seed': key[1],
                'optimizer_order_seed': key[2],
                'dataset_sha256': next(iter(hashes.values())),
            })

        for label, rows in data.items():
            by_key = {seed_key(row): row for row in rows}
            for metric in METRICS:
                values = np.array([by_key[key][metric] for key in sorted(reference_keys)])
                summaries.append({
                    'problem': problem,
                    'condition': label,
                    'metric': metric,
                    'n': len(values),
                    'mean': float(values.mean()),
                    'std': float(values.std(ddof=1)),
                })

        reference = {seed_key(row): row for row in data['Std-PINN']}
        problem_tests = []
        for label, rows in data.items():
            if label == 'Std-PINN':
                continue
            candidate = {seed_key(row): row for row in rows}
            for metric in METRICS:
                base = np.array([reference[key][metric] for key in sorted(reference_keys)])
                values = np.array([candidate[key][metric] for key in sorted(reference_keys)])
                differences = base - values
                try:
                    p_value = float(wilcoxon(differences, alternative='greater').pvalue)
                except ValueError:
                    p_value = 1.0
                problem_tests.append({
                    'problem': problem,
                    'reference': 'Std-PINN',
                    'condition': label,
                    'metric': metric,
                    'n': len(differences),
                    'reference_mean': float(base.mean()),
                    'condition_mean': float(values.mean()),
                    'paired_improvement_mean': float(differences.mean()),
                    'paired_improvement_ci95': bootstrap_interval(differences, generator, 10000),
                    'wilcoxon_p': p_value,
                })
        holm_adjust(problem_tests)
        tests.extend(problem_tests)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / 'method_summary.csv', summaries)
    write_csv(args.output_dir / 'paired_comparisons.csv', tests)
    write_csv(args.output_dir / 'paired_dataset_audit.csv', audit_rows)
    audit = {
        'passed': True,
        'runs': sum(len(mapping) * EXPECTED_RUNS for mapping in FOLDERS.values()),
        'conditions': sum(len(mapping) for mapping in FOLDERS.values()),
        'paired_tests': len(tests),
        'paired_dataset_checks': len(audit_rows),
    }
    (args.output_dir / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
