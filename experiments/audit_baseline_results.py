#!/usr/bin/env python
"""Audit the representative-baseline result matrix before analysis."""

import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / 'configs' / 'representative_baselines_manifest.json'
PROTOCOL_FIELDS = (
    'problem', 'epochs', 'learning_rate', 'lr_schedule', 'minimum_learning_rate',
    'warmup_epochs', 'num_residual', 'num_boundary', 'num_initial',
    'dynamic_sampling', 'dtype',
)


def result_folder(root, problem, tag):
    if problem == 'burgers':
        return root / f'results_{tag}'
    if problem == 'helmholtz_4_4':
        return root / f'results_helmholtz_a1_4.0_a2_4.0_{tag}'
    if problem == 'klein_gordon':
        return root / f'results_klein_gordon_{tag}'
    raise ValueError(problem)


def completed_summary_rows(folder):
    path = folder / 'aggregate' / 'logs' / 'summary.csv'
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8') as handle:
        return [row for row in csv.DictReader(handle) if row.get('run', '').isdigit()]


def final_history_epoch(run_dir):
    path = run_dir / 'logs' / 'history.csv'
    if not path.exists():
        return None
    with path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        return None
    return int(float(rows[-1][0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, action='append')
    args = parser.parse_args()

    manifest_paths = args.manifest or [DEFAULT_MANIFEST]
    manifests = [json.loads(path.read_text(encoding='utf-8')) for path in manifest_paths]
    failures, checks = [], []
    jobs_by_problem = {}
    seen_tags = set()
    for manifest in manifests:
        for job in manifest['jobs']:
            tag = job['result_tag']
            if tag in seen_tags:
                failures.append(f'duplicate result tag across manifests: {tag}')
            seen_tags.add(tag)
            jobs_by_problem.setdefault(job['problem'], []).append(job)

    for problem, jobs in jobs_by_problem.items():
        per_seed_hashes = {}
        protocol_signature = None
        for job in jobs:
            condition = f"{job['architecture']}+{job['optimizer']}"
            tag = job['result_tag']
            folder = result_folder(args.root, problem, tag)
            run_dirs = sorted((folder / 'runs').glob('run_*'))
            if len(run_dirs) != 5:
                failures.append(f'{problem}/{condition}: expected 5 runs, found {len(run_dirs)}')
                continue
            summary_rows = completed_summary_rows(folder)
            if len(summary_rows) != 5:
                failures.append(
                    f'{problem}/{condition}: expected 5 completed summary rows, '
                    f'found {len(summary_rows)}'
                )
            for run_dir in run_dirs:
                metadata_path = run_dir / 'logs' / 'run_metadata.json'
                if not metadata_path.exists():
                    failures.append(f'{problem}/{condition}/{run_dir.name}: missing metadata')
                    continue
                metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
                config, seeds = metadata['config'], metadata['seeds']
                if config.get('network') != job['architecture']:
                    failures.append(
                        f'{problem}/{condition}/{run_dir.name}: network is '
                        f"{config.get('network')!r}, expected {job['architecture']!r}"
                    )
                if config.get('mode') != job['optimizer']:
                    failures.append(
                        f'{problem}/{condition}/{run_dir.name}: mode is '
                        f"{config.get('mode')!r}, expected {job['optimizer']!r}"
                    )
                if config.get('dynamic_sampling') is not False:
                    failures.append(f'{problem}/{condition}/{run_dir.name}: dynamic sampling is not false')
                signature = tuple((field, config.get(field)) for field in PROTOCOL_FIELDS)
                if protocol_signature is None:
                    protocol_signature = signature
                elif signature != protocol_signature:
                    failures.append(
                        f'{problem}/{condition}/{run_dir.name}: training protocol differs '
                        'from the first audited condition'
                    )
                key = (
                    seeds['master_seed'], seeds['sample_seed'], seeds['init_seed'],
                    seeds['optimizer_order_seed'],
                )
                prior = per_seed_hashes.setdefault(key, metadata['dataset_sha256'])
                if prior != metadata['dataset_sha256']:
                    failures.append(f'{problem}/seed {key}: unequal training-data hashes across conditions')
                if job['optimizer'] == 'rba' and not (run_dir / 'logs' / 'rba_final_attention.npy').exists():
                    failures.append(f'{problem}/{condition}/{run_dir.name}: missing final attention')
                if job['optimizer'] == 'brdr' and not (run_dir / 'logs' / 'brdr_final_state.npz').exists():
                    failures.append(f'{problem}/{condition}/{run_dir.name}: missing final BRDR state')
                if not (run_dir / 'checkpoints' / 'model_final.pt').exists():
                    failures.append(f'{problem}/{condition}/{run_dir.name}: missing fixed-final checkpoint')
                final_epoch = final_history_epoch(run_dir)
                expected_final_epoch = int(config['epochs']) - 1
                if final_epoch != expected_final_epoch:
                    failures.append(
                        f'{problem}/{condition}/{run_dir.name}: final history epoch is '
                        f'{final_epoch}, expected {expected_final_epoch}'
                    )
                checks.append({'problem': problem, 'condition': condition, 'run': run_dir.name,
                               'result_tag': tag, 'master_seed': seeds['master_seed'],
                               'sample_seed': seeds['sample_seed'], 'init_seed': seeds['init_seed'],
                               'optimizer_order_seed': seeds['optimizer_order_seed'],
                               'final_history_epoch': final_epoch,
                               'dataset_sha256': metadata['dataset_sha256']})
        if len(per_seed_hashes) != 5:
            failures.append(f'{problem}: expected 5 paired sample bundles, found {len(per_seed_hashes)}')
    expected_checks = 5 * sum(len(manifest['jobs']) for manifest in manifests)
    if len(checks) != expected_checks:
        failures.append(f'expected {expected_checks} audited runs, found {len(checks)}')
    payload = {
        'passed': not failures,
        'manifest_ids': [manifest['manifest_id'] for manifest in manifests],
        'expected_runs': expected_checks,
        'failures': failures,
        'checks': checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed': payload['passed'], 'checks': len(checks), 'failures': failures}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
