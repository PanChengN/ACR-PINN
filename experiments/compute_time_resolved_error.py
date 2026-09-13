#!/usr/bin/env python
"""Backfill KG time-resolved errors from fixed-final checkpoints without retraining."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.models import build_model


def exact_solution(points):
    x = points[:, 0:1]
    t = points[:, 1:2]
    return x * torch.cos(5 * np.pi * t) + (x * t) ** 3


def normalize_xt(points, time_max):
    return torch.cat([
        2.0 * points[:, 0:1] - 1.0,
        2.0 * points[:, 1:2] / time_max - 1.0,
    ], dim=1)


def time_rows(model, time_max, n_test, device):
    x = np.linspace(0.0, 1.0, n_test)
    times = np.linspace(0.0, time_max, n_test)
    rows = []
    model.eval()
    with torch.no_grad():
        for time_value in times:
            physical = torch.tensor(
                np.column_stack([x, np.full_like(x, time_value)]),
                dtype=torch.float32,
                device=device,
            )
            exact = exact_solution(physical).cpu().numpy().reshape(-1)
            prediction = model(normalize_xt(physical, time_max)).cpu().numpy().reshape(-1)
            difference = prediction - exact
            rows.append({
                'time': float(time_value),
                'normalized_time': float(time_value / time_max),
                'relative_l2': float(np.linalg.norm(difference) / np.linalg.norm(exact)),
                'relative_linf': float(np.max(np.abs(difference)) / np.max(np.abs(exact))),
                'mse': float(np.mean(difference ** 2)),
            })
    return rows


def write_rows(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cpu')
    parser.add_argument('--n-test', type=int, default=200)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.n_test < 20:
        raise ValueError('n-test must be at least 20.')
    device = torch.device(args.device)

    generated, reused, all_rows = 0, 0, []
    run_dirs = sorted((args.result_dir / 'runs').glob('run_*'))
    if not run_dirs:
        raise FileNotFoundError(f'No run directories under {args.result_dir}')
    for run_dir in run_dirs:
        metadata_path = run_dir / 'logs' / 'run_metadata.json'
        checkpoint_path = run_dir / 'checkpoints' / 'model_final.pt'
        output_path = run_dir / 'logs' / 'time_error.csv'
        if not metadata_path.exists() or not checkpoint_path.exists():
            raise FileNotFoundError(f'Missing metadata/checkpoint in {run_dir}')
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        config = metadata['config']
        time_max = float(config.get('time_max', 1.0))
        if output_path.exists() and not args.overwrite:
            with output_path.open(newline='', encoding='utf-8') as handle:
                rows = list(csv.DictReader(handle))
            reused += 1
        else:
            state = load_checkpoint(checkpoint_path, device)
            recorded_stabilized = config.get('lda_stabilized')
            if recorded_stabilized is None:
                recorded_stabilized = 'residual_scales' in state
            recorded_residual_scale = config.get('lda_residual_scale')
            if recorded_residual_scale is None:
                recorded_residual_scale = 0.1
            model = build_model(
                config['network'],
                config.get('layers', [2, 50, 50, 50, 50, 1]),
                lda_stabilized=bool(recorded_stabilized),
                lda_residual_scale=float(recorded_residual_scale),
            ).to(device)
            incompatibility = model.load_state_dict(state, strict=False)
            allowed_missing = {'residual_scales'} if not recorded_stabilized else set()
            if set(incompatibility.missing_keys) - allowed_missing or incompatibility.unexpected_keys:
                raise RuntimeError(
                    f'Incompatible checkpoint {checkpoint_path}: '
                    f'missing={incompatibility.missing_keys}, '
                    f'unexpected={incompatibility.unexpected_keys}'
                )
            rows = time_rows(model, time_max, args.n_test, device)
            write_rows(output_path, rows)
            generated += 1
        master_seed = metadata['seeds']['master_seed']
        all_rows.extend({'master_seed': master_seed, **row} for row in rows)

    aggregate_dir = args.result_dir / 'aggregate' / 'logs'
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    tidy_path = aggregate_dir / 'time_error_run_level.csv'
    write_rows(tidy_path, all_rows)

    aggregate_rows = []
    for time_value in sorted({float(row['time']) for row in all_rows}):
        selected = [row for row in all_rows if float(row['time']) == time_value]
        aggregate = {
            'time': time_value,
            'normalized_time': float(selected[0]['normalized_time']),
        }
        for metric in ('relative_l2', 'relative_linf', 'mse'):
            values = np.array([float(row[metric]) for row in selected])
            aggregate[f'{metric}_mean'] = float(values.mean())
            aggregate[f'{metric}_std'] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        aggregate_rows.append(aggregate)
    write_rows(aggregate_dir / 'time_error_summary.csv', aggregate_rows)
    print(json.dumps({
        'result_dir': str(args.result_dir),
        'runs': len(run_dirs),
        'generated': generated,
        'reused': reused,
        'n_test': args.n_test,
        'run_level_output': str(tidy_path),
    }, indent=2))


if __name__ == '__main__':
    main()
