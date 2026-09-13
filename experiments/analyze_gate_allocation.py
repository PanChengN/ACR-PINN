"""Aggregate recorded spatial/temporal branch gates without retraining."""
import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    out = args.output_dir
    records, sources, hashes = [], [], {}
    expected_iterations = list(range(0, 100000, 1000)) + [99999]
    for mode in ('sum', 'pcgrad'):
        for run in range(1, 6):
            folder = out / 'raw' / mode / f'run_{run}'
            metadata = json.loads((folder / 'run_metadata.json').read_text())
            config, seeds = metadata['config'], metadata['seeds']
            assert config['network'] == 'separate_st_lda'
            assert config['mode'] == mode and config['time_max'] == 1
            assert config['epochs'] == 100000
            assert config['effective_layers'] == [2, 50, 50, 50, 50, 1]
            assert seeds['master_seed'] == seeds['init_seed'] == 19017 + run
            assert seeds['sample_seed'] == 20017 + run
            assert seeds['optimizer_order_seed'] == 21017 + run
            assert hashes.setdefault(run, metadata['dataset_sha256']) == metadata['dataset_sha256']
            for name in ('run_metadata.json', 'diagnostics.jsonl'):
                path = folder / name
                sources.append({'path': str(path.relative_to(out)),
                                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            rows = [json.loads(line) for line in (folder / 'diagnostics.jsonl').read_text().splitlines() if line.strip()]
            assert [r['iteration'] for r in rows] == expected_iterations
            for row in rows:
                assert [g['layer'] for g in row['gates']] == list(range(4))
                for gate in row['gates']:
                    spatial, temporal = gate['mean_branch_1'], gate['mean_branch_2']
                    assert np.isfinite([spatial, temporal, gate['entropy_mean']]).all()
                    assert 0 <= spatial <= 1 and 0 <= temporal <= 1
                    assert abs(spatial + temporal - 1) < 2e-6
                    records.append({'optimizer': mode, 'run': run, 'master_seed': seeds['master_seed'],
                                    'iteration': row['iteration'], 'layer': gate['layer'] + 1,
                                    'spatial_weight': spatial, 'temporal_weight': temporal,
                                    'normalized_entropy': gate['entropy_mean'] / np.log(2)})
    summary = []
    for mode in ('sum', 'pcgrad'):
        for iteration in expected_iterations:
            for layer in range(1, 5):
                rows = [r for r in records if (r['optimizer'], r['iteration'], r['layer']) == (mode, iteration, layer)]
                assert len(rows) == 5
                aggregate = {'optimizer': mode, 'iteration': iteration, 'layer': layer, 'n': 5}
                for metric in ('spatial_weight', 'temporal_weight', 'normalized_entropy'):
                    values = np.array([r[metric] for r in rows])
                    aggregate[metric + '_mean'] = float(values.mean())
                    aggregate[metric + '_sd'] = float(values.std(ddof=1))
                summary.append(aggregate)
    endpoint = [r for r in summary if r['iteration'] == 99999]
    write_csv(out / 'branch_gate_run_level.csv', records)
    write_csv(out / 'branch_gate_trajectory.csv', summary)
    write_csv(out / 'branch_gate_final.csv', endpoint)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
    for col, mode in enumerate(('sum', 'pcgrad')):
        ax = axes[0, col]
        for layer, color in zip(range(1, 5), colors):
            rows = [r for r in summary if r['optimizer'] == mode and r['layer'] == layer]
            x = np.array([r['iteration'] for r in rows]) / 99999
            mean = np.array([r['spatial_weight_mean'] for r in rows])
            sd = np.array([r['spatial_weight_sd'] for r in rows])
            ax.plot(x, mean, color=color, label=f'Layer {layer}', lw=1.8)
            ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=.12, linewidth=0)
        ax.axhline(.5, color='0.45', ls='--', lw=.8)
        ax.set(xlabel='Training progress', ylabel='Mean spatial-branch gate weight',
               title=f'({chr(97 + col)}) {"Direct summation" if mode == "sum" else "GCR-PINN"}',
               xlim=(0, 1), ylim=(0, 1))
        ax.legend(ncol=2, fontsize=8, loc='lower left')
        ax.grid(axis='y', alpha=.15)
        ax = axes[1, col]
        rows = [r for r in endpoint if r['optimizer'] == mode]
        x = np.arange(1, 5)
        for branch, shift, color in [('spatial', -.17, '#0072B2'), ('temporal', .17, '#E69F00')]:
            means = [r[branch + '_weight_mean'] for r in rows]
            errors = [r[branch + '_weight_sd'] for r in rows]
            ax.bar(x + shift, means, width=.30, yerr=errors, capsize=3, color=color,
                   alpha=.85, label=f'{branch.capitalize()} branch')
        for run in range(1, 6):
            for layer in range(1, 5):
                r = next(r for r in records if (r['optimizer'], r['run'], r['iteration'], r['layer']) == (mode, run, 99999, layer))
                for branch, shift in [('spatial', -.17), ('temporal', .17)]:
                    ax.plot(layer + shift + (run - 3) * .025, r[branch + '_weight'], '.', color='0.20', ms=3)
        ax.axhline(.5, color='0.45', ls='--', lw=.8)
        ax.set(xticks=x, xlabel='Hidden layer', ylabel='Final mean gate weight', ylim=(0, 1),
               title=f'({chr(99 + col)}) Final recorded iteration: 99,999')
        ax.legend(fontsize=8, loc='lower left')
        ax.grid(axis='y', alpha=.15)
    fig.suptitle('Spatial and temporal branch weighting in separate-encoder LDA', fontsize=13)
    fig.supxlabel('Klein-Gordon, T = 1 | Five paired seeds | Mean +/- sample SD; dots: individual runs\n'
                  'Recorded gates averaged over 1,024 training interior points and 50 channels per layer', fontsize=9)
    fig.savefig(out / 'spatial_temporal_gate_weights.png', dpi=200)
    fig.savefig(out / 'spatial_temporal_gate_weights.pdf')
    plt.close(fig)
    audit = {'passed': True, 'runs': 10, 'layers': 4, 'recorded_iterations_per_run': 101,
             'run_layer_records': len(records), 'seed_pairing': '19018--19022',
             'branch_1': 'x only', 'branch_2': 't only',
             'probe': 'first 1024 resampled training interior points, normalized coordinates',
             'aggregation': 'mean over points/channels within run/layer, then mean and sample SD over 5 runs',
             'interpretation': 'Gate allocation in separate encoders; does not identify physical importance or joint-encoder branch semantics.',
             'python': sys.executable, 'python_version': platform.python_version(),
             'numpy': np.__version__, 'matplotlib': matplotlib.__version__, 'sources': sources}
    (out / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({
        'passed': True,
        'runs': audit['runs'],
        'final_records': len(endpoint),
        'output_dir': str(out),
    }, indent=2))


if __name__ == '__main__':
    main()
