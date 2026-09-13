#!/usr/bin/env python
"""Audit and decompose sampling-sensitivity's sampling/initialization matrix."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = ('final_l2', 'final_linf', 'final_mse')


def read_single_run(path):
    with path.open(newline='', encoding='utf-8') as handle:
        rows = [row for row in csv.DictReader(handle) if row.get('run', '').isdigit()]
    if len(rows) != 1:
        raise ValueError(f'Expected exactly one run in {path}, found {len(rows)}.')
    row = rows[0]
    return {key: float(row[key]) for key in METRICS}


def read_metadata(summary_path):
    result_dir = summary_path.parents[2]
    metadata_path = result_dir / 'runs' / 'run_1' / 'logs' / 'run_metadata.json'
    checkpoint_path = result_dir / 'runs' / 'run_1' / 'checkpoints' / 'model_final.pt'
    if not metadata_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(f'Missing metadata/checkpoint under {result_dir}')
    return json.loads(metadata_path.read_text(encoding='utf-8'))


def component_rows(values):
    """Balanced two-way ANOVA components, retaining raw estimates if negative."""
    matrix = np.asarray(values, dtype=float)
    n_sample, n_init = matrix.shape
    grand = matrix.mean()
    sample_mean, init_mean = matrix.mean(axis=1), matrix.mean(axis=0)
    ss_sample = n_init * np.square(sample_mean - grand).sum()
    ss_init = n_sample * np.square(init_mean - grand).sum()
    interaction = matrix - sample_mean[:, None] - init_mean[None, :] + grand
    ss_interaction = np.square(interaction).sum()
    ms_sample = ss_sample / (n_sample - 1)
    ms_init = ss_init / (n_init - 1)
    ms_interaction = ss_interaction / ((n_sample - 1) * (n_init - 1))
    estimates = {
        'sampling': (ms_sample - ms_interaction) / n_init,
        'initialization': (ms_init - ms_interaction) / n_sample,
        'interaction_residual': ms_interaction,
    }
    total = sum(max(0.0, value) for value in estimates.values())
    rows = []
    for source, estimate in estimates.items():
        rows.append({
            'source': source,
            'variance_estimate': float(estimate),
            'nonnegative_fraction': float(max(0.0, estimate) / total) if total else 0.0,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--results-root', type=Path, default=Path('.'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    observed = defaultdict(dict)
    run_rows = []
    missing = []
    for job in manifest['jobs']:
        tag = job['result_tag']
        matches = sorted(args.results_root.glob(f'results_*_{tag}/aggregate/logs/summary.csv'))
        if len(matches) != 1:
            missing.append({'result_tag': tag, 'matches': [str(item) for item in matches]})
            continue
        metrics = read_single_run(matches[0])
        metadata = read_metadata(matches[0])
        config, seeds = metadata['config'], metadata['seeds']
        expected = {
            'master_seed': job['master_seed_base'],
            'sample_seed': job['sample_seed_base'],
            'init_seed': job['init_seed_base'],
            'optimizer_order_seed': job['optimizer_order_seed_base'],
        }
        if any(int(seeds[key]) != value for key, value in expected.items()):
            raise ValueError(f'{tag}: recorded seed bundle does not match manifest')
        if config.get('network') != job['architecture'] or config.get('mode') != job['optimizer']:
            raise ValueError(f'{tag}: recorded method does not match manifest')
        if config.get('dynamic_sampling') is not True:
            raise ValueError(f'{tag}: sampling-sensitivity requires dynamic sampling')
        label = tag.removeprefix('sampling_h44_').rsplit('_s', 1)[0]
        sample_seed, init_seed = job['sample_seed_base'], job['init_seed_base']
        observed[label][sample_seed, init_seed] = metrics
        run_rows.append({
            'condition': label, 'sample_seed': sample_seed, 'init_seed': init_seed,
            'optimizer_order_seed': job['optimizer_order_seed_base'],
            'dataset_sha256': metadata['dataset_sha256'], 'summary_csv': str(matches[0]), **metrics,
        })
    if missing:
        raise SystemExit(json.dumps({'error': 'incomplete sampling-sensitivity matrix', 'missing': missing}, indent=2))

    config_path = args.manifest.parents[1] / manifest['config']
    policy = json.loads(config_path.read_text(encoding='utf-8'))['seed_policy']
    samples, inits = policy['factorial_sample_seeds'], policy['factorial_init_seeds']
    summaries = []
    components = []
    hashes_by_sample = defaultdict(set)
    for row in run_rows:
        hashes_by_sample[row['sample_seed']].add(row['dataset_sha256'])
    inconsistent_hashes = {
        sample_seed: sorted(values) for sample_seed, values in hashes_by_sample.items()
        if len(values) != 1
    }
    if inconsistent_hashes:
        raise ValueError(f'Paired methods/initializations have inconsistent data hashes: {inconsistent_hashes}')
    for label, cells in sorted(observed.items()):
        if set(cells) != {(s, i) for s in samples for i in inits}:
            raise ValueError(f'{label} does not contain the complete {len(samples)}-by-{len(inits)} matrix.')
        for metric in METRICS:
            matrix = [[cells[s, i][metric] for i in inits] for s in samples]
            values = np.asarray(matrix)
            summaries.append({'condition': label, 'metric': metric, 'n': values.size,
                              'mean': float(values.mean()), 'std': float(values.std(ddof=1))})
            for row in component_rows(matrix):
                components.append({'condition': label, 'metric': metric, **row})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in [('run_level_metrics.csv', run_rows), ('summary.csv', summaries),
                           ('variance_components.csv', components)]:
        with (args.output_dir / filename).open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    audit = {
        'passed': True,
        'conditions': len(observed),
        'runs': len(run_rows),
        'sampling_streams': len(samples),
        'initialization_streams': len(inits),
        'output_dir': str(args.output_dir),
    }
    (args.output_dir / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
