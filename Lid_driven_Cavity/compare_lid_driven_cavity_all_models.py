#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BIN_SIZE = 100
RESULT_DIRS = [
    ('PINN-ATTN', 'results_Lid_driven_Cavity_pinn_attention'),
    ('PCGRAD-ATTN', 'results_Lid_driven_Cavity_pcgrad_attention'),
    ('PINN-MLP', 'results_Lid_driven_Cavity_pinn_mlp'),
    ('PCGRAD-MLP', 'results_Lid_driven_Cavity_pcgrad_mlp'),
]


def load_history(path):
    return np.genfromtxt(path, delimiter=',', names=True)


def load_run_histories(result_dir):
    runs_dir = os.path.join(result_dir, 'runs')
    if not os.path.isdir(runs_dir):
        return []
    histories = []
    for run_name in sorted(os.listdir(runs_dir)):
        hist_path = os.path.join(runs_dir, run_name, 'logs', 'history.csv')
        if not os.path.exists(hist_path):
            continue
        data = load_history(hist_path)
        if data.size == 0:
            continue
        histories.append(data)
    return histories


def compute_binned_stats(histories, metric_key, bin_size):
    if not histories:
        return None
    bins = {}
    for data in histories:
        iters = data['iter']
        values = data[metric_key]
        if iters.size == 0:
            continue
        base = iters[0]
        bin_index = ((iters - base) // bin_size).astype(int)
        for idx in np.unique(bin_index):
            mask = bin_index == idx
            if not np.any(mask):
                continue
            mean_val = np.mean(values[mask])
            bins.setdefault(idx, []).append(mean_val)
    if not bins:
        return None
    indices = np.array(sorted(bins.keys()), dtype=int)
    centers = indices * bin_size
    means = np.array([np.mean(bins[i]) for i in indices])
    stds = np.array([np.std(bins[i]) for i in indices])
    return centers, means, stds


histories = {}
for label, result_dir in RESULT_DIRS:
    runs = load_run_histories(os.path.join(BASE_DIR, result_dir))
    if not runs:
        print(f'Warning: no valid runs found for {result_dir}, skipping.')
        continue
    histories[label] = runs

if not histories:
    raise SystemExit('No history files found.')

out_dir = os.path.join(BASE_DIR, 'results_comparison')
os.makedirs(out_dir, exist_ok=True)


def save_plot(metric_key, title, ylabel, filename, series_keys):
    plt.figure(figsize=(6, 4))
    for label in series_keys:
        runs = histories.get(label)
        if not runs:
            continue
        stats = compute_binned_stats(runs, metric_key, BIN_SIZE)
        if stats is None:
            continue
        centers, means, stds = stats
        plt.semilogy(centers, means, label=label)
        plt.fill_between(centers, means - stds, means + stds, alpha=0.2)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f'Saved {title} plot to {out_path}')


PAIR_SETS = {
    'pinn_pcgrad_attn': ['PINN-ATTN', 'PCGRAD-ATTN'],
    'pinn_pcgrad_mlp': ['PINN-MLP', 'PCGRAD-MLP'],
}

for tag, series in PAIR_SETS.items():
    save_plot('total', 'Total Loss', 'Loss', f'total_loss_{tag}.pdf', series)
    save_plot('l2', 'Relative Error (L2)', 'Relative Error', f'relative_error_l2_{tag}.pdf', series)
    save_plot('linf', 'Absolute Error (Linf)', 'Absolute Error', f'absolute_error_linf_{tag}.pdf', series)
