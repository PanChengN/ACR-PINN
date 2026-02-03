#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIRS = [
    ('PINN-ATTN', 'results_pinn_attention'),
    ('PCGRAD-ATTN', 'results_pcgrad_attention'),
    ('PINN-MLP', 'results_pinn_mlp'),
    ('PCGRAD-MLP', 'results_pcgrad_mlp'),
]
BIN_SIZE = 100

DISPLAY_LABELS = {
    'PINN-MLP': 'Std-PINN',
    'PINN-ATTN': 'LDA-PINN',
    'PCGRAD-MLP': 'GC-PINN',
    'PCGRAD-ATTN': 'ACR-PINN',
}

JOURNAL_COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
SHADE_SCALE = 0.55
SMOOTH_WINDOW = 5

plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'axes.prop_cycle': plt.cycler(color=JOURNAL_COLORS),
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'lines.linewidth': 2.0,
    'legend.frameon': False,
})


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


def smooth_series(values, window):
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='same')


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
    handles = []
    labels = []
    for label in series_keys:
        runs = histories.get(label)
        if not runs:
            continue
        stats = compute_binned_stats(runs, metric_key, BIN_SIZE)
        if stats is None:
            continue
        centers, means, stds = stats
        means = smooth_series(means, SMOOTH_WINDOW)
        stds = smooth_series(stds, SMOOTH_WINDOW)
        display_label = DISPLAY_LABELS.get(label, label)
        line = plt.semilogy(centers, means, label=display_label)[0]
        plt.fill_between(
            centers,
            means - stds * SHADE_SCALE,
            means + stds * SHADE_SCALE,
            alpha=0.2,
        )
        handles.append(line)
        labels.append(display_label)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel(ylabel)
    if handles:
        plt.legend(handles, labels)
    plt.tight_layout()
    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f'Saved {title} plot to {out_path}')


ALL_MODELS = ['PINN-MLP', 'PINN-ATTN', 'PCGRAD-MLP', 'PCGRAD-ATTN']
save_plot('total', 'Total Loss', 'Loss', 'total_loss_all_models.pdf', ALL_MODELS)
save_plot('l2', r'Relative Error ($L_2$)', r'Relative Error ($L_2$)', 'relative_error_l2_all_models.pdf', ALL_MODELS)
save_plot('linf', r'Relative Error ($L_\infty$)', r'Relative Error ($L_\infty$)', 'relative_error_linf_all_models.pdf', ALL_MODELS)
