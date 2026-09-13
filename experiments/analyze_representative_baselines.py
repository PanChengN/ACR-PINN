#!/usr/bin/env python
"""Audit and summarize the fixed-sampling representative-baseline comparison."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from audit_baseline_results import result_folder
from summarize_paired_results import (
    METRICS,
    bootstrap_interval,
    holm_adjust,
    read_runs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STRONG_MANIFEST = REPO_ROOT / 'configs' / 'representative_baselines_manifest.json'
DEFAULT_CORE_MANIFEST = REPO_ROOT / 'configs' / 'fixed_core_manifest.json'
LABELS = {
    ('mlp', 'sum'): 'Std-PINN',
    ('mlp', 'pcgrad'): 'GCR-PINN',
    ('lda', 'sum'): 'LDA-PINN',
    ('lda', 'pcgrad'): 'ACR-PINN',
    ('modified_mlp', 'lra'): 'M-MLP+LRA',
    ('fourier_mlp', 'sum'): 'FF-PINN',
    ('mlp', 'rba'): 'RBA-PINN',
    ('mlp', 'brdr'): 'BRDR-PINN',
}


def load_jobs(paths):
    jobs = []
    for path in paths:
        manifest = json.loads(path.read_text(encoding='utf-8'))
        for job in manifest['jobs']:
            key = (job['architecture'], job['optimizer'])
            if key not in LABELS:
                raise ValueError(f'No display label registered for {key}')
            jobs.append({**job, 'label': LABELS[key]})
    return jobs


def paired_record(reference_label, reference, condition_label, condition, metric, generator, repetitions):
    reference_by_key = {
        (row['sample_seed'], row['init_seed'], row['optimizer_order_seed']): row
        for row in reference
    }
    condition_by_key = {
        (row['sample_seed'], row['init_seed'], row['optimizer_order_seed']): row
        for row in condition
    }
    if set(reference_by_key) != set(condition_by_key):
        raise ValueError(f'Unmatched seed bundles: {reference_label} vs {condition_label}')
    keys = sorted(reference_by_key)
    base = np.array([reference_by_key[key][metric] for key in keys], dtype=float)
    values = np.array([condition_by_key[key][metric] for key in keys], dtype=float)
    differences = base - values
    try:
        p_value = float(wilcoxon(differences, alternative='greater', method='auto').pvalue)
    except ValueError:
        p_value = 1.0
    return {
        'reference': reference_label,
        'condition': condition_label,
        'metric': metric,
        'n': len(values),
        'reference_mean': float(base.mean()),
        'condition_mean': float(values.mean()),
        'paired_improvement_mean': float(differences.mean()),
        'paired_improvement_ci95': bootstrap_interval(differences, generator, repetitions),
        'wilcoxon_p': p_value,
    }


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--strong-manifest', type=Path, default=DEFAULT_STRONG_MANIFEST)
    parser.add_argument('--core-manifest', type=Path, default=DEFAULT_CORE_MANIFEST)
    parser.add_argument('--bootstrap-repetitions', type=int, default=10000)
    parser.add_argument('--bootstrap-seed', type=int, default=20260901)
    args = parser.parse_args()
    if args.bootstrap_repetitions < 1000:
        raise ValueError('Use at least 1000 bootstrap repetitions.')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_dir / 'integrity_audit.json'
    subprocess.run([
        sys.executable,
        str(REPO_ROOT / 'experiments' / 'audit_baseline_results.py'),
        '--root', str(args.root),
        '--output', str(audit_path),
        '--manifest', str(args.strong_manifest),
        '--manifest', str(args.core_manifest),
    ], cwd=REPO_ROOT, check=True)

    jobs = load_jobs([args.strong_manifest, args.core_manifest])
    jobs_by_problem = {}
    for job in jobs:
        jobs_by_problem.setdefault(job['problem'], []).append(job)

    generator = np.random.default_rng(args.bootstrap_seed)
    summaries, comparison_records = [], []
    run_data = {}
    for problem, problem_jobs in jobs_by_problem.items():
        labels = [job['label'] for job in problem_jobs]
        if len(labels) != 8 or len(set(labels)) != 8:
            raise ValueError(f'{problem}: expected 8 unique conditions, found {labels}')
        for job in problem_jobs:
            folder = result_folder(args.root, problem, job['result_tag'])
            rows = read_runs(folder)
            run_data[(problem, job['label'])] = rows
            for metric in METRICS:
                values = np.array([row[metric] for row in rows], dtype=float)
                summaries.append({
                    'problem': problem,
                    'condition': job['label'],
                    'metric': metric,
                    'n': len(values),
                    'mean': float(values.mean()),
                    'std': float(values.std(ddof=1)),
                })

        for metric in METRICS:
            metric_rows = [row for row in summaries if row['problem'] == problem and row['metric'] == metric]
            for rank, row in enumerate(sorted(metric_rows, key=lambda item: item['mean']), start=1):
                row['mean_rank'] = rank

        # Family 1: every method against the fixed-sampling Std-PINN reference.
        std_rows = run_data[(problem, 'Std-PINN')]
        std_family = []
        for label in labels:
            if label == 'Std-PINN':
                continue
            for metric in METRICS:
                record = paired_record(
                    'Std-PINN', std_rows, label, run_data[(problem, label)], metric,
                    generator, args.bootstrap_repetitions,
                )
                record.update({'problem': problem, 'family': 'all_vs_std'})
                std_family.append(record)
        holm_adjust(std_family)
        comparison_records.extend(std_family)

        # Family 2: the pre-specified ACR-PINN comparison against all alternatives.
        acr_rows = run_data[(problem, 'ACR-PINN')]
        acr_family = []
        for label in labels:
            if label == 'ACR-PINN':
                continue
            for metric in METRICS:
                # Reference is the alternative, so a positive difference means ACR improves.
                record = paired_record(
                    label, run_data[(problem, label)], 'ACR-PINN', acr_rows, metric,
                    generator, args.bootstrap_repetitions,
                )
                record.update({'problem': problem, 'family': 'acr_vs_all'})
                acr_family.append(record)
        holm_adjust(acr_family)
        comparison_records.extend(acr_family)

    write_csv(args.output_dir / 'method_summary.csv', summaries)
    write_csv(args.output_dir / 'paired_comparisons.csv', comparison_records)
    (args.output_dir / 'paired_comparisons.json').write_text(
        json.dumps(comparison_records, indent=2) + '\n', encoding='utf-8'
    )
    print(json.dumps({
        'passed': True,
        'audited_runs': 5 * len(jobs),
        'summary_rows': len(summaries),
        'paired_tests': len(comparison_records),
        'output_dir': str(args.output_dir),
    }, indent=2))


if __name__ == '__main__':
    main()
