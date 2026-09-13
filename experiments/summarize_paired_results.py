#!/usr/bin/env python
"""Create auditable paired summaries from fixed-final PINN result folders.

Input folders are the result directories that contain
``aggregate/logs/summary.csv``. The script verifies matched seed bundles before
computing descriptive metrics, paired bootstrap intervals, Wilcoxon tests, and
Holm-adjusted p-values. It never selects checkpoints or filters runs.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


METRICS = ('final_l2', 'final_linf', 'final_mse')


def read_runs(folder):
    path = Path(folder) / 'aggregate' / 'logs' / 'summary.csv'
    if not path.exists():
        raise FileNotFoundError(f'Missing aggregate summary: {path}')
    with path.open(newline='', encoding='utf-8') as handle:
        rows = [row for row in csv.DictReader(handle) if row.get('run', '').isdigit()]
    if not rows:
        raise ValueError(f'No completed runs in {path}')
    parsed = []
    for row in rows:
        parsed.append({
            'run': int(row['run']),
            'sample_seed': int(row['sample_seed']),
            'init_seed': int(row['init_seed']),
            'optimizer_order_seed': int(row['optimizer_order_seed']),
            **{metric: float(row[metric]) for metric in METRICS},
        })
    return parsed


def parse_condition(value):
    label, separator, folder = value.partition('=')
    if not separator or not label or not folder:
        raise argparse.ArgumentTypeError('conditions must use LABEL=RESULT_DIRECTORY')
    return label, Path(folder)


def bootstrap_interval(differences, generator, repetitions=10000):
    samples = generator.choice(differences, size=(repetitions, len(differences)), replace=True)
    means = samples.mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def holm_adjust(records):
    ordered = sorted(enumerate(records), key=lambda item: item[1]['wilcoxon_p'])
    previous = 0.0
    total = len(records)
    for rank, (index, record) in enumerate(ordered):
        adjusted = min(1.0, (total - rank) * record['wilcoxon_p'])
        previous = max(previous, adjusted)
        records[index]['holm_p'] = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=parse_condition, required=True)
    parser.add_argument('--condition', type=parse_condition, action='append', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--bootstrap-repetitions', type=int, default=10000)
    parser.add_argument('--bootstrap-seed', type=int, default=20260831)
    args = parser.parse_args()
    if args.bootstrap_repetitions < 1000:
        raise ValueError('Use at least 1000 bootstrap repetitions.')

    reference_label, reference_folder = args.reference
    reference = read_runs(reference_folder)
    reference_keys = [(row['sample_seed'], row['init_seed'], row['optimizer_order_seed']) for row in reference]
    if len(set(reference_keys)) != len(reference_keys):
        raise ValueError('Reference has duplicate seed bundles.')
    generator = np.random.default_rng(args.bootstrap_seed)
    records, run_rows = [], []
    for label, folder in args.condition:
        candidate = read_runs(folder)
        candidate_by_key = {
            (row['sample_seed'], row['init_seed'], row['optimizer_order_seed']): row for row in candidate
        }
        if set(candidate_by_key) != set(reference_keys):
            raise ValueError(f'{label} seed bundles do not exactly match {reference_label}.')
        for metric in METRICS:
            base = np.array([row[metric] for row in reference], dtype=float)
            values = np.array([candidate_by_key[key][metric] for key in reference_keys], dtype=float)
            # Positive difference means the candidate improves over reference.
            differences = base - values
            try:
                p_value = float(wilcoxon(differences, alternative='greater', method='auto').pvalue)
            except ValueError:
                p_value = 1.0
            records.append({
                'reference': reference_label, 'condition': label, 'metric': metric,
                'n': len(values), 'reference_mean': float(base.mean()),
                'condition_mean': float(values.mean()), 'condition_std': float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                'paired_improvement_mean': float(differences.mean()),
                'paired_improvement_ci95': bootstrap_interval(differences, generator, args.bootstrap_repetitions),
                'wilcoxon_p': p_value,
            })
        for key, base in zip(reference_keys, reference):
            row = {'condition': label, 'sample_seed': key[0], 'init_seed': key[1], 'optimizer_order_seed': key[2]}
            row.update({metric: candidate_by_key[key][metric] for metric in METRICS})
            run_rows.append(row)
    holm_adjust(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'paired_statistics.json').write_text(json.dumps(records, indent=2) + '\n', encoding='utf-8')
    for name, rows in (('paired_statistics.csv', records), ('run_level_metrics.csv', run_rows)):
        with (args.output_dir / name).open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({'reference': reference_label, 'comparisons': len(records), 'output_dir': str(args.output_dir)}, indent=2))


if __name__ == '__main__':
    main()
