#!/usr/bin/env python
"""Audit LDA gate trajectories and fixed-final gate interventions for the LDA study."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.experiment_protocol import epoch_rng, fast_lhs
from common.models import build_model


INTERVENTIONS = ('native', 'uniform', 'swap', 'sample_permute', 'layer_mean')


def relative_metrics(prediction, reference):
    difference = prediction - reference
    return {
        'relative_l2': float(np.linalg.norm(difference) / np.linalg.norm(reference)),
        'relative_linf': float(np.max(np.abs(difference)) / np.max(np.abs(reference))),
        'mse': float(np.mean(difference ** 2)),
    }


def evaluation_data(problem, metadata, device):
    config = metadata['config']
    if problem == 'burgers':
        reference = scipy.io.loadmat(REPO_ROOT / 'data' / 'burgers_shock.mat')
        x = reference['x'].flatten()[:, None]
        t = reference['t'].flatten()[:, None]
        exact = np.real(reference['usol']).T.flatten()[:, None].astype(np.float32)
        X, T = np.meshgrid(x, t)
        physical = np.hstack([X.flatten()[:, None], T.flatten()[:, None]]).astype(np.float32)
        network = physical.copy()
        network[:, 1] = 2.0 * network[:, 1] - 1.0
        rng = epoch_rng(metadata['seeds']['sample_seed'], config['epochs'] - 1,
                        config.get('resample_every') or 1)
        calibration = fast_lhs(rng, config['num_residual'], 2)
        calibration[:, 0] = -1.0 + 2.0 * calibration[:, 0]
        calibration[:, 1] = 2.0 * calibration[:, 1] - 1.0
    elif problem == 'helmholtz_4_4':
        axis = np.linspace(-1.0, 1.0, 200, dtype=np.float32)
        X, Y = np.meshgrid(axis, axis)
        network = np.hstack([X.reshape(-1, 1), Y.reshape(-1, 1)]).astype(np.float32)
        exact = (np.sin(config['a1'] * np.pi * network[:, 0:1]) *
                 np.sin(config['a2'] * np.pi * network[:, 1:2])).astype(np.float32)
        rng = epoch_rng(metadata['seeds']['sample_seed'], config['epochs'] - 1,
                        config.get('resample_every') or 1)
        calibration = -1.0 + 2.0 * fast_lhs(rng, config['num_residual'], 2)
    elif problem == 'klein_gordon':
        axis = np.linspace(0.0, 1.0, 200, dtype=np.float32)
        X, T = np.meshgrid(axis, axis)
        physical = np.hstack([X.reshape(-1, 1), T.reshape(-1, 1)]).astype(np.float32)
        exact = (physical[:, 0:1] * np.cos(5.0 * np.pi * physical[:, 1:2]) +
                 (physical[:, 0:1] * physical[:, 1:2]) ** 3).astype(np.float32)
        network = 2.0 * physical - 1.0
        rng = epoch_rng(metadata['seeds']['sample_seed'], config['epochs'] - 1,
                        config.get('resample_every') or 1)
        calibration = 2.0 * fast_lhs(rng, config['num_residual'], 2) - 1.0
    else:
        raise ValueError(f'Unsupported problem: {problem}')
    return (
        torch.tensor(network, dtype=torch.float32, device=device),
        torch.tensor(calibration, dtype=torch.float32, device=device),
        exact,
    )


def gate_layer_means(model, calibration_inputs):
    with torch.no_grad():
        _, gates = model.forward_with_gates(calibration_inputs)
    return [gate.mean(dim=0) for gate in gates]


def analyze_interventions(problem, result_dir, device):
    rows = []
    for run_dir in sorted((result_dir / 'runs').glob('run_*')):
        metadata_path = run_dir / 'logs' / 'run_metadata.json'
        checkpoint_path = run_dir / 'checkpoints' / 'model_final.pt'
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        layers = metadata['config']['layers']
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        stabilized = metadata['config'].get('lda_stabilized')
        if stabilized is None:
            stabilized = 'residual_scales' in state
        model = build_model(
            'lda', layers,
            lda_stabilized=stabilized,
            lda_residual_scale=metadata['config'].get('lda_residual_scale') or 0.1,
        ).to(device)
        incompatible = model.load_state_dict(state, strict=stabilized)
        if not stabilized:
            if set(incompatible.missing_keys) != {'residual_scales'} or incompatible.unexpected_keys:
                raise RuntimeError(
                    f'Unexpected legacy LDA checkpoint mismatch: {incompatible}'
                )
        model.eval()
        test_inputs, calibration_inputs, reference = evaluation_data(problem, metadata, device)
        layer_means = gate_layer_means(model, calibration_inputs)
        native_metrics = None
        for mode in INTERVENTIONS:
            generator = None
            if mode == 'sample_permute':
                generator = torch.Generator(device=device).manual_seed(
                    int(metadata['seeds']['master_seed']) + 9000
                )
            with torch.no_grad():
                prediction = model.forward_with_gate_intervention(
                    test_inputs, mode,
                    layer_means=layer_means if mode == 'layer_mean' else None,
                    generator=generator,
                ).cpu().numpy()
            metrics = relative_metrics(prediction, reference)
            if native_metrics is None:
                native_metrics = metrics
            row = {
                'problem': problem,
                'result_dir': result_dir.name,
                'run': run_dir.name,
                'master_seed': metadata['seeds']['master_seed'],
                'optimizer': metadata['config']['mode'],
                'lda_stabilized': stabilized,
                'intervention': mode,
                **metrics,
            }
            for metric in ('relative_l2', 'relative_linf', 'mse'):
                row[f'{metric}_ratio_to_native'] = metrics[metric] / native_metrics[metric]
            rows.append(row)
    return rows


def collect_trajectory(problem, result_dir):
    rows = []
    for run_dir in sorted((result_dir / 'runs').glob('run_*')):
        metadata = json.loads(
            (run_dir / 'logs' / 'run_metadata.json').read_text(encoding='utf-8')
        )
        diagnostic_path = run_dir / 'logs' / 'diagnostics.jsonl'
        for line in diagnostic_path.read_text(encoding='utf-8').splitlines():
            record = json.loads(line)
            for gate in record.get('gates', []):
                rows.append({
                    'problem': problem,
                    'result_dir': result_dir.name,
                    'run': run_dir.name,
                    'master_seed': metadata['seeds']['master_seed'],
                    'optimizer': metadata['config']['mode'],
                    'iteration': record['iteration'],
                    **gate,
                })
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f'No records available for {path.name}.')
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--problem', choices=('burgers', 'helmholtz_4_4', 'klein_gordon'),
                        required=True)
    parser.add_argument('--result-dir', type=Path, action='append', required=True,
                        help='Repeat for direct-summation LDA and GCR-PINN result directories.')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    device = torch.device(args.device)
    trajectories = []
    interventions = []
    for result_dir in args.result_dir:
        trajectories.extend(collect_trajectory(args.problem, result_dir))
        interventions.extend(analyze_interventions(args.problem, result_dir, device))
    write_csv(args.output_dir / f'{args.problem}_gate_trajectory.csv', trajectories)
    write_csv(args.output_dir / f'{args.problem}_gate_interventions.csv', interventions)
    print(json.dumps({
        'problem': args.problem,
        'trajectory_rows': len(trajectories),
        'intervention_rows': len(interventions),
        'output_dir': str(args.output_dir),
    }, indent=2))


if __name__ == '__main__':
    main()
