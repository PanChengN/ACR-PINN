#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
from torch.optim import Adam
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from common.reproducibility import (
    latin_hypercube, load_or_create_npz, optimizer_order_generator,
    seed_bundle, seed_model_initialization, spawned_rngs, write_run_metadata,
)
from common.models import build_model as build_shared_model, canonical_architecture
from common.optimization import (
    BalancedResidualDecayRate,
    canonical_optimizer,
    GradNormBalancer,
    GradientStatisticsLossBalancer,
    ResidualBasedAttention,
    hutchinson_ntk_traces,
    mgda_combined_gradients,
    ntk_loss_weights,
    project_conflicting_gradients,
)
from common.diagnostics import DiagnosticsRecorder, collect_task_gradients
from common.scheduling import build_learning_rate_scheduler
from common.cost_benchmark import CudaCostProfiler
from common.experiment_protocol import (
    epoch_rng, fast_lhs, warmup_cosine_learning_rate, write_sampling_descriptor,
)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.dpi'] = 300

# ============================================================
# Config
# ============================================================
MODE = canonical_optimizer(os.getenv('MODE', 'sum'))
NET_ARCH = canonical_architecture(os.getenv('NET_ARCH', 'mlp'))
LDA_STABILIZED = os.getenv('LDA_STABILIZED', '1').strip().lower() in ('1', 'true', 'yes')
LDA_RESIDUAL_SCALE = float(os.getenv('LDA_RESIDUAL_SCALE', '0.1'))
RUN_TAG = os.getenv('RESULT_TAG', f'{MODE}_{NET_ARCH}')
EPOCHS = int(os.getenv('EPOCHS', '40000'))
LR = float(os.getenv('LEARNING_RATE', '1e-3'))
LR_SCHEDULE = os.getenv('LR_SCHEDULE', 'constant')
LR_MIN = float(os.getenv('LR_MIN', '1e-5'))
WARMUP_EPOCHS = int(os.getenv('WARMUP_EPOCHS', '0'))
RECORD_EVERY = int(os.getenv('RECORD_EVERY', '100'))
DIAGNOSTICS_EVERY = int(os.getenv('DIAGNOSTICS_EVERY', str(RECORD_EVERY)))
WALL_TIME_BUDGET_SEC = float(os.getenv('WALL_TIME_BUDGET_SEC', '0'))
ERROR_THRESHOLD_L2 = float(os.getenv('ERROR_THRESHOLD_L2', '0'))
WALL_TIME_CHECK_EVERY = int(os.getenv('WALL_TIME_CHECK_EVERY', '25'))
NUM_RUNS = int(os.getenv('NUM_RUNS', '5'))
RUN_START_INDEX = int(os.getenv('RUN_START_INDEX', '0'))
SAMPLE_SEED_BASE = int(os.getenv('SAMPLE_SEED_BASE', '1234'))
INIT_SEED_BASE = int(os.getenv('INIT_SEED_BASE', '2234'))
OPTIMIZER_SEED_BASE = int(os.getenv('OPTIMIZER_SEED_BASE', '3234'))
FOURIER_FEATURE_COUNT = int(os.getenv('FOURIER_FEATURE_COUNT', '50'))
FOURIER_SCALE = float(os.getenv('FOURIER_SCALE', '1.0'))
GRADNORM_ALPHA = float(os.getenv('GRADNORM_ALPHA', '1.5'))
GRADNORM_LR = float(os.getenv('GRADNORM_LR', '0.025'))
NTK_UPDATE_EVERY = int(os.getenv('NTK_UPDATE_EVERY', '100'))
NTK_MAX_OUTPUTS = int(os.getenv('NTK_MAX_OUTPUTS', '128'))
LRA_UPDATE_EVERY = int(os.getenv('LRA_UPDATE_EVERY', '10'))
LRA_EMA = float(os.getenv('LRA_EMA', '0.9'))
RBA_ETA = float(os.getenv('RBA_ETA', '0.001'))
RBA_GAMMA = float(os.getenv('RBA_GAMMA', '0.999'))
BRDR_BETA_C = float(os.getenv('BRDR_BETA_C', '0.999'))
BRDR_BETA_W = float(os.getenv('BRDR_BETA_W', '0.999'))

NUM_F = int(os.getenv('HELMHOLTZ_NUM_F', '10000'))
NUM_B = int(os.getenv('HELMHOLTZ_NUM_B', '400'))
DYNAMIC_SAMPLING = os.getenv('HELMHOLTZ_DYNAMIC_SAMPLING', '0').strip().lower() in ('1', 'true', 'yes')
RESAMPLE_EVERY = int(os.getenv('HELMHOLTZ_RESAMPLE_EVERY', '1'))
if MODE in ('rba', 'brdr') and DYNAMIC_SAMPLING:
    raise ValueError(f'{MODE.upper()} requires fixed collocation points; set HELMHOLTZ_DYNAMIC_SAMPLING=0.')

A1 = float(os.getenv('HELMHOLTZ_A1', '4.0'))
A2 = float(os.getenv('HELMHOLTZ_A2', '4.0'))
K = float(os.getenv('HELMHOLTZ_K', '1.0'))

SAVE_DIR = f'results_helmholtz_a1_{A1}_a2_{A2}_{RUN_TAG}'
RUNS_DIR = os.path.join(SAVE_DIR, 'runs')
AGG_DIR = os.path.join(SAVE_DIR, 'aggregate')
AGG_FIG_DIR = os.path.join(AGG_DIR, 'figures')
AGG_LOG_DIR = os.path.join(AGG_DIR, 'logs')
AGG_CKPT_DIR = os.path.join(AGG_DIR, 'checkpoints')
for path in (SAVE_DIR, RUNS_DIR, AGG_DIR, AGG_FIG_DIR, AGG_LOG_DIR, AGG_CKPT_DIR):
    os.makedirs(path, exist_ok=True)

GENERATED_DATA_DIR = REPO_ROOT / 'generated_data' / 'helmholtz'

# ============================================================
# Device
# ============================================================
REQUESTED_DEVICE = os.getenv('DEVICE', 'auto').strip().lower()
device = (
    torch.device(REQUESTED_DEVICE) if REQUESTED_DEVICE != 'auto'
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print(f'Using device: {device}')

# ============================================================
# MLP
# ============================================================
class MLP(nn.Module):
    def __init__(self, layers):
        super().__init__()
        net = []
        for i in range(len(layers) - 2):
            lin = nn.Linear(layers[i], layers[i+1])
            nn.init.xavier_normal_(lin.weight, gain=5/3)
            nn.init.zeros_(lin.bias)
            net += [lin, nn.Tanh()]
        lin = nn.Linear(layers[-2], layers[-1])
        nn.init.xavier_normal_(lin.weight, gain=1)
        nn.init.zeros_(lin.bias)
        net.append(lin)
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class NetAttentionDynamic(nn.Module):
    """
    MLP variant with per-layer dynamic attention between two learned encoders.
    """
    def __init__(self, layers, activation=nn.Tanh()):
        super().__init__()
        if len(layers) < 3:
            raise ValueError("NetAttentionDynamic requires at least one hidden layer.")
        self.activation = activation
        self.linear = nn.ModuleList([
            nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)
        ])
        hidden_dims = [layers[i + 1] for i in range(len(layers) - 2)]
        self.encoder1 = nn.ModuleList([nn.Linear(layers[0], h) for h in hidden_dims])
        self.encoder2 = nn.ModuleList([nn.Linear(layers[0], h) for h in hidden_dims])
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3 * h, h),
                nn.Tanh(),
                nn.Linear(h, 2 * h)
            )
            for h in hidden_dims
        ])
        for lin in self.linear:
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
        for enc in list(self.encoder1) + list(self.encoder2):
            nn.init.xavier_normal_(enc.weight)
            nn.init.zeros_(enc.bias)

    def forward(self, x):
        a = x
        for idx in range(len(self.linear) - 1):
            a = self.activation(self.linear[idx](a))
            enc1 = self.activation(self.encoder1[idx](x))
            enc2 = self.activation(self.encoder2[idx](x))
            gate_input = torch.cat([a, enc1, enc2], dim=1)
            logits = self.gates[idx](gate_input)
            logits = logits.view(a.size(0), 2, a.size(1))
            weights = torch.softmax(logits, dim=1)
            attn = weights[:, 0, :] * enc1 + weights[:, 1, :] * enc2
            a = a + attn
        return self.linear[-1](a)


def build_model():
    layers = [2, 50, 50, 50, 50, 1]
    return build_shared_model(
        NET_ARCH, layers, lda_stabilized=LDA_STABILIZED,
        lda_residual_scale=LDA_RESIDUAL_SCALE, fourier_feature_count=FOURIER_FEATURE_COUNT,
        fourier_scale=FOURIER_SCALE,
    )

# ============================================================
# Helmholtz residual (physical coordinates)
# ============================================================
def exact_solution(X):
    return (
        torch.sin(A1 * np.pi * X[:, 0:1]) *
        torch.sin(A2 * np.pi * X[:, 1:2])
    )


def forcing_term(X):
    coeff = -(A1 * np.pi) ** 2 - (A2 * np.pi) ** 2 + K ** 2
    return coeff * torch.sin(A1 * np.pi * X[:, 0:1]) * torch.sin(A2 * np.pi * X[:, 1:2])


def helmholtz_residual(u, X):
    du = grad(u, X, torch.ones_like(u), create_graph=True)[0]
    u_x = du[:, 0:1]
    u_y = du[:, 1:2]
    u_xx = grad(u_x, X, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = grad(u_y, X, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
    return u_xx + u_yy + (K ** 2) * u - forcing_term(X)

# ============================================================
# Data
# ============================================================
def sampler(sample_seed):
    dataset_path = GENERATED_DATA_DIR / f'seed_{sample_seed}_a1_{A1}_a2_{A2}_nf{NUM_F}_nb{NUM_B}.npz'

    def create_dataset():
        lhs_rng, boundary_rng = spawned_rngs(sample_seed, 2)
        X_f = -1.0 + 2.0 * latin_hypercube(2, NUM_F, lhs_rng)
        n_edge = NUM_B // 4
        remainder = NUM_B - 4 * n_edge
        edges = []
        y_lr = -1.0 + 2.0 * boundary_rng.random((n_edge, 1))
        edges.extend([np.hstack([-np.ones((n_edge, 1)), y_lr]),
                      np.hstack([np.ones((n_edge, 1)), y_lr])])
        x_tb = -1.0 + 2.0 * boundary_rng.random((n_edge, 1))
        edges.extend([np.hstack([x_tb, -np.ones((n_edge, 1))]),
                      np.hstack([x_tb, np.ones((n_edge, 1))])])
        if remainder > 0:
            rem = -1.0 + 2.0 * boundary_rng.random((remainder, 2))
            for i in range(remainder):
                edges.append(np.array([[-1, rem[i, 1]]]) if i % 2 == 0
                             else np.array([[rem[i, 0], -1]]))
        return {'X_f': X_f, 'X_b': np.vstack(edges)}

    arrays, dataset_sha256 = load_or_create_npz(dataset_path, create_dataset)
    return (
        torch.tensor(arrays['X_f'], dtype=torch.float32, device=device, requires_grad=True),
        torch.tensor(arrays['X_b'], dtype=torch.float32, device=device),
        dataset_path, dataset_sha256,
    )


def dynamic_sample_epoch(sample_seed, epoch):
    if NUM_B % 4:
        raise ValueError('HELMHOLTZ_NUM_B must be divisible by four')
    rng = epoch_rng(sample_seed, epoch, RESAMPLE_EVERY)
    X_f = -1.0 + 2.0 * fast_lhs(rng, NUM_F, 2)
    values = -1.0 + 2.0 * fast_lhs(rng, NUM_B, 1)
    n_edge = NUM_B // 4
    X_b = np.vstack([
        np.hstack([-np.ones((n_edge, 1)), values[:n_edge]]),
        np.hstack([np.ones((n_edge, 1)), values[n_edge:2 * n_edge]]),
        np.hstack([values[2 * n_edge:3 * n_edge], -np.ones((n_edge, 1))]),
        np.hstack([values[3 * n_edge:], np.ones((n_edge, 1))]),
    ])
    return (
        torch.tensor(X_f, dtype=torch.float32, device=device, requires_grad=True),
        torch.tensor(X_b, dtype=torch.float32, device=device),
    )


def dynamic_sampling_descriptor(sample_seed):
    path = GENERATED_DATA_DIR / f'seed_{sample_seed}_dynamic_a1_{A1}_a2_{A2}_nf{NUM_F}_nb{NUM_B}_r{RESAMPLE_EVERY}.json'
    return write_sampling_descriptor(path, {
        'algorithm': 'deterministic_randomized_latin_hypercube',
        'sample_seed': sample_seed, 'num_residual': NUM_F,
        'num_boundary_total': NUM_B, 'resample_every': RESAMPLE_EVERY,
        'domain': [[-1.0, 1.0], [-1.0, 1.0]],
    })

# ============================================================
# PCGrad
# ============================================================
def pcgrad(grads, order_generator):
    proj = [[g.clone() for g in task] for task in grads]
    n = len(proj)
    for i in range(n):
        for j in torch.randperm(n, generator=order_generator):
            j = j.item()
            if i == j:
                continue
            dot = sum((gi * gj).sum() for gi, gj in zip(proj[i], grads[j]))
            if dot < 0:
                norm = sum((gj ** 2).sum() for gj in grads[j])
                proj[i] = [gi - dot / norm * gj for gi, gj in zip(proj[i], grads[j])]
    return [sum(gs) for gs in zip(*proj)]



SMOOTH_SIGMA = 3.0
SLICE_X = 1.0
SLICE_EXACT_COLOR = '#2b2b2b'
SLICE_PRED_COLOR = '#d55e00'


def smooth_curve(values, sigma):
    if len(values) == 0 or sigma <= 0:
        return values
    return gaussian_filter1d(values, sigma=sigma, mode='nearest')


def plot_run_outputs(X, Y, U_exact, U_pred, fig_dir, prefix=''):
    os.makedirs(fig_dir, exist_ok=True)
    if prefix:
        heatmap_name = f'heatmap_exact_{prefix}.pdf'
        pred_name = f'heatmap_pred_{prefix}.pdf'
        error_name = f'heatmap_error_{prefix}.pdf'
        slice_name = f'slice_x1_{prefix}.pdf'
    else:
        heatmap_name = 'heatmap_exact.pdf'
        pred_name = 'heatmap_pred.pdf'
        error_name = 'heatmap_error.pdf'
        slice_name = 'slice_x1.pdf'
    exact_path = os.path.join(fig_dir, heatmap_name)
    pred_path = os.path.join(fig_dir, pred_name)
    error_path = os.path.join(fig_dir, error_name)

    fig, ax = plt.subplots(figsize=(5, 4))
    pcm = ax.pcolormesh(X, Y, U_exact, shading='gouraud')
    ax.set_title('Exact')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(pcm, ax=ax)
    fig.tight_layout()
    plt.savefig(exact_path, dpi=300, bbox_inches='tight', pad_inches=0.02, transparent=True)
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 4))
    pcm = ax.pcolormesh(X, Y, U_pred, shading='gouraud')
    ax.set_title('Prediction')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(pcm, ax=ax)
    fig.tight_layout()
    plt.savefig(pred_path, dpi=300, bbox_inches='tight', pad_inches=0.02, transparent=True)
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 4))
    pcm = ax.pcolormesh(X, Y, np.abs(U_exact - U_pred), shading='gouraud')
    ax.set_title('Abs Error')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(pcm, ax=ax)
    fig.tight_layout()
    plt.savefig(error_path, dpi=300, bbox_inches='tight', pad_inches=0.02, transparent=True)
    plt.close()

    x_values = X[0, :]
    y_values = Y[:, 0]
    idx = int(np.abs(x_values - SLICE_X).argmin())
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(y_values, U_exact[:, idx], color=SLICE_EXACT_COLOR, label='Exact')
    ax.plot(y_values, U_pred[:, idx], color=SLICE_PRED_COLOR, linestyle='--', label='Prediction')
    ax.set_title(f'x={x_values[idx]:.2f}')
    ax.set_xlabel('y')
    ax.set_ylabel('u(x, y)')
    ax.set_ylim(-0.5, 0.5)
    ax.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(fig_dir, slice_name), dpi=300, bbox_inches='tight', pad_inches=0.02, transparent=True)
    plt.close()

    return exact_path, pred_path, error_path, os.path.join(fig_dir, slice_name)


def plot_history_curves(hist_array, fig_dir, prefix=''):
    if hist_array.size == 0:
        return
    os.makedirs(fig_dir, exist_ok=True)
    loss_name = f'loss_{prefix}.pdf' if prefix else 'loss.pdf'
    error_name = f'error_{prefix}.pdf' if prefix else 'error.pdf'
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.plot(hist_array[:, 0], hist_array[:, 3], color='#1f78b4')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Total Loss')
    fig.tight_layout()
    plt.savefig(os.path.join(fig_dir, loss_name), dpi=300, bbox_inches='tight', pad_inches=0.02, transparent=True)
    plt.close()

    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.plot(hist_array[:, 0], hist_array[:, 4], color='#33a02c', label='L2')
    ax.plot(hist_array[:, 0], hist_array[:, 5], color='#e31a1c', label='L∞')
    ax.legend()
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Relative error')
    fig.tight_layout()
    plt.savefig(os.path.join(fig_dir, error_name), dpi=300, bbox_inches='tight', pad_inches=0.02, transparent=True)
    plt.close()


# ============================================================
# Train (possibly multiple runs)
# ============================================================
HISTORY_COLS = ['iter', 'pde', 'bc', 'total', 'l2', 'linf', 'mse', 'elapsed_sec']

if __name__ == '__main__' and os.getenv('RUN_ALL', '1') == '1':
    import subprocess
    import sys

    modes = ['pinn', 'pcgrad']
    nets = ['mlp', 'attention']
    base_env = os.environ.copy()
    base_env['RUN_ALL'] = '0'

    for mode in modes:
        for net in nets:
            print(f"=== Running MODE={mode} NET_ARCH={net} ===")
            env = base_env.copy()
            env['MODE'] = mode
            env['NET_ARCH'] = net
            subprocess.check_call([sys.executable, __file__], env=env)
    raise SystemExit(0)


def train_single_run(run_idx, seeds):
    print(f"\n=== Run {run_idx + 1}/{NUM_RUNS} | seeds={seeds} ===")
    seed_model_initialization(seeds['init_seed'])
    order_generator = optimizer_order_generator(seeds['optimizer_order_seed'])
    run_dir = os.path.join(RUNS_DIR, f'run_{run_idx + 1}')
    run_fig_dir = os.path.join(run_dir, 'figures')
    run_log_dir = os.path.join(run_dir, 'logs')
    run_ckpt_dir = os.path.join(run_dir, 'checkpoints')
    for path in (run_dir, run_fig_dir, run_log_dir, run_ckpt_dir):
        os.makedirs(path, exist_ok=True)

    if DYNAMIC_SAMPLING:
        dataset_path, dataset_sha256 = dynamic_sampling_descriptor(seeds['sample_seed'])
        X_f, X_b = dynamic_sample_epoch(seeds['sample_seed'], 0)
    else:
        X_f, X_b, dataset_path, dataset_sha256 = sampler(seeds['sample_seed'])
    net = build_model().to(device)
    warmup_cosine = LR_SCHEDULE.strip().lower() == 'linear_warmup_then_cosine'
    opt = Adam(net.parameters(), lr=LR_MIN if warmup_cosine else LR)
    scheduler = None if warmup_cosine else build_learning_rate_scheduler(opt, LR_SCHEDULE, EPOCHS, LR_MIN)
    recorder = DiagnosticsRecorder(net, run_log_dir, device, DIAGNOSTICS_EVERY)
    cost_profiler = CudaCostProfiler(device, EPOCHS)
    gradnorm_balancer = GradNormBalancer(2, GRADNORM_ALPHA, GRADNORM_LR) if MODE == 'gradnorm' else None
    ntk_generator = torch.Generator().manual_seed(seeds['optimizer_order_seed'] + 3000)
    ntk_weights = torch.ones(2, device=device)
    lra_balancer = GradientStatisticsLossBalancer(2, LRA_EMA) if MODE == 'lra' else None
    lra_weights = torch.ones(2, device=device)
    rba = ResidualBasedAttention(RBA_ETA, RBA_GAMMA) if MODE == 'rba' else None
    brdr = BalancedResidualDecayRate(BRDR_BETA_C, BRDR_BETA_W) if MODE == 'brdr' else None
    config = {'problem': 'helmholtz', 'mode': MODE, 'network': NET_ARCH,
              'lda_stabilized': LDA_STABILIZED if NET_ARCH == 'lda' else None,
              'lda_residual_scale': LDA_RESIDUAL_SCALE if NET_ARCH == 'lda' else None,
              'layers': [2, 50, 50, 50, 50, 1],
              'effective_layers': net.effective_layers, 'epochs': EPOCHS,
              'learning_rate': LR, 'lr_schedule': LR_SCHEDULE,
              'minimum_learning_rate': LR_MIN, 'warmup_epochs': WARMUP_EPOCHS,
              'record_every': RECORD_EVERY, 'dynamic_sampling': DYNAMIC_SAMPLING,
              'resample_every': RESAMPLE_EVERY if DYNAMIC_SAMPLING else None,
              'wall_time_budget_sec': WALL_TIME_BUDGET_SEC or None,
              'error_threshold_l2': ERROR_THRESHOLD_L2 or None,
              'wall_time_check_every': WALL_TIME_CHECK_EVERY if WALL_TIME_BUDGET_SEC else None,
              'num_residual': NUM_F, 'num_boundary': NUM_B,
              'gradnorm_alpha': GRADNORM_ALPHA if MODE == 'gradnorm' else None,
              'gradnorm_learning_rate': GRADNORM_LR if MODE == 'gradnorm' else None,
              'ntk_update_every': NTK_UPDATE_EVERY if MODE == 'ntk' else None,
              'ntk_max_outputs': NTK_MAX_OUTPUTS if MODE == 'ntk' else None,
              'lra_update_every': LRA_UPDATE_EVERY if MODE == 'lra' else None,
              'lra_ema': LRA_EMA if MODE == 'lra' else None,
              'rba_eta': RBA_ETA if MODE == 'rba' else None,
              'rba_gamma': RBA_GAMMA if MODE == 'rba' else None,
              'brdr_beta_c': BRDR_BETA_C if MODE == 'brdr' else None,
              'brdr_beta_w': BRDR_BETA_W if MODE == 'brdr' else None,
              'a1': A1, 'a2': A2, 'k': K, 'optimizer': 'Adam', 'dtype': 'float32'}
    write_run_metadata(os.path.join(run_log_dir, 'run_metadata.json'), config, seeds,
                       dataset_path, dataset_sha256)

    N_test = 200
    x = np.linspace(-1, 1, N_test)
    y = np.linspace(-1, 1, N_test)
    X, Y = np.meshgrid(x, y)
    XY_flat = np.hstack([X.flatten()[:, None], Y.flatten()[:, None]])
    XY_t = torch.tensor(XY_flat, dtype=torch.float32, device=device, requires_grad=True)
    u_exact = exact_solution(XY_t).detach().cpu().numpy()

    history = []
    start = time.time()
    threshold_time_sec = None
    threshold_iteration = None
    stopped_by_wall_time = False

    def synchronized_elapsed():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        return time.time() - start

    def evaluate_snapshot(ep, L_pde, L_bc, L):
        nonlocal threshold_time_sec, threshold_iteration
        with torch.no_grad():
            pred_snapshot = net(XY_t).cpu().numpy()
        elapsed_at_evaluation = synchronized_elapsed()
        l2 = np.linalg.norm(pred_snapshot - u_exact) / np.linalg.norm(u_exact)
        linf = np.max(np.abs(pred_snapshot - u_exact)) / np.max(np.abs(u_exact))
        mse = np.mean((pred_snapshot - u_exact) ** 2)
        history.append([
            ep, L_pde.item(), L_bc.item(), L.item(), l2, linf, mse,
            elapsed_at_evaluation,
        ])
        if ERROR_THRESHOLD_L2 > 0 and threshold_time_sec is None and l2 <= ERROR_THRESHOLD_L2:
            threshold_time_sec = elapsed_at_evaluation
            threshold_iteration = ep
        print(f"[{MODE.upper()}][Run {run_idx + 1}] {ep:6d} | "
              f"PDE {L_pde.item():.2e} BC {L_bc.item():.2e} | "
              f"L2 {l2:.2e} L∞ {linf:.2e} MSE {mse:.2e} | "
              f"time {elapsed_at_evaluation:.1f}s")

    def forward_losses(return_task_outputs=False):
        u_f = net(X_f)
        res = helmholtz_residual(u_f, X_f)
        L_pde = torch.mean(res ** 2)
        boundary_output = net(X_b)
        L_bc = torch.mean(boundary_output ** 2)
        if return_task_outputs:
            return L_pde, L_bc, [res, boundary_output]
        return L_pde, L_bc

    for ep in range(EPOCHS):
        if DYNAMIC_SAMPLING:
            X_f, X_b = dynamic_sample_epoch(seeds['sample_seed'], ep)
        if warmup_cosine:
            if WALL_TIME_BUDGET_SEC > 0 and ep >= WARMUP_EPOCHS:
                elapsed_fraction = min(1.0, max(0.0, (time.time() - start) / WALL_TIME_BUDGET_SEC))
                current_lr = LR_MIN + 0.5 * (LR - LR_MIN) * (
                    1.0 + np.cos(np.pi * elapsed_fraction)
                )
            else:
                current_lr = warmup_cosine_learning_rate(ep, EPOCHS, LR, LR_MIN, WARMUP_EPOCHS)
            for group in opt.param_groups:
                group['lr'] = current_lr
        cost_profiler.start_step(ep)
        ntk_traces = None
        if MODE == 'ntk' and ep % NTK_UPDATE_EVERY == 0:
            L_pde, L_bc, task_outputs = forward_losses(return_task_outputs=True)
            ntk_traces = hutchinson_ntk_traces(
                task_outputs, list(net.parameters()), ntk_generator, NTK_MAX_OUTPUTS
            )
            ntk_weights = ntk_loss_weights(ntk_traces).detach()
        elif MODE == 'rba':
            L_pde, L_bc, task_outputs = forward_losses(return_task_outputs=True)
            L_pde = rba.update_and_weight(task_outputs[0])
        elif MODE == 'brdr':
            L_pde, L_bc, task_outputs = forward_losses(return_task_outputs=True)
        else:
            L_pde, L_bc = forward_losses()
        losses = [L_pde, L_bc]
        if MODE == 'lra' and ep % LRA_UPDATE_EVERY == 0:
            lra_weights = lra_balancer.update(collect_task_gradients(losses, list(net.parameters())))
        L = (brdr.update_and_weight(task_outputs) if MODE == 'brdr'
             else sum(weight * loss for weight, loss in zip(lra_weights, losses)) if MODE == 'lra'
             else sum(losses) if MODE != 'ntk' else sum(weight * loss for weight, loss in zip(ntk_weights, losses)))
        task_names = ['pde', 'bc']
        params = list(net.parameters())
        named_params = list(net.named_parameters())
        diagnostic_grads = None
        projected_grads = None
        optimizer_weights = None
        record_diagnostics = WALL_TIME_BUDGET_SEC <= 0 and recorder.should_record(ep, EPOCHS)
        if MODE in ('sum', 'ntk', 'lra', 'rba', 'brdr') and record_diagnostics:
            diagnostic_grads = collect_task_gradients(losses, params)
        recorder.start_step()

        if MODE in ('sum', 'ntk', 'lra', 'rba', 'brdr'):
            opt.zero_grad()
            L.backward()
            if MODE == 'brdr':
                brdr.correct_gradients(L, params, opt.param_groups[0]['lr'])
            opt.step()
        elif MODE in ('pcgrad', 'mgda', 'gradnorm'):
            grads = []
            for loss in losses:
                opt.zero_grad()
                loss.backward(retain_graph=True)
                grads.append([
                    (p.grad.clone() if p.grad is not None else torch.zeros_like(p))
                    for p in params
                ])
            if MODE == 'pcgrad':
                final_grad = project_conflicting_gradients(grads, order_generator)
            elif MODE == 'mgda':
                final_grad, optimizer_weights = mgda_combined_gradients(grads)
            else:
                final_grad, optimizer_weights = gradnorm_balancer.update_and_combine(grads, losses)
            diagnostic_grads = grads
            projected_grads = final_grad
            opt.zero_grad()
            for p, g in zip(params, final_grad):
                p.grad = g
            opt.step()
        else:
            raise ValueError(f"Unknown MODE='{MODE}'")
        if scheduler is not None:
            scheduler.step()
        cost_profiler.end_step(ep)
        recorder.end_step()
        if record_diagnostics:
            recorder.record(ep, task_names, diagnostic_grads, named_params, projected_grads,
                            X_f[:min(1024, len(X_f))].detach(),
                            {
                                'learning_rate': opt.param_groups[0]['lr'],
                                'optimizer_weights': (
                                    optimizer_weights.detach().cpu().tolist()
                                    if optimizer_weights is not None else (
                                        ntk_weights.detach().cpu().tolist() if MODE == 'ntk' else None
                                    )
                                ),
                                'ntk_traces': (
                                    ntk_traces.detach().cpu().tolist() if ntk_traces is not None else None
                                ),
                            })

        recorded_this_epoch = ep % RECORD_EVERY == 0 or ep == EPOCHS - 1
        if recorded_this_epoch:
            evaluate_snapshot(ep, L_pde, L_bc, L)

        if WALL_TIME_BUDGET_SEC > 0 and (
            ep % WALL_TIME_CHECK_EVERY == 0 or recorded_this_epoch
        ):
            elapsed_now = synchronized_elapsed()
            if elapsed_now >= WALL_TIME_BUDGET_SEC:
                if not recorded_this_epoch:
                    evaluate_snapshot(ep, L_pde, L_bc, L)
                stopped_by_wall_time = True
                print(
                    f"Wall-time budget reached at iteration {ep}: "
                    f"{elapsed_now:.3f}s >= {WALL_TIME_BUDGET_SEC:.3f}s"
                )
                break

    elapsed = synchronized_elapsed()
    recorder.finalize(elapsed)
    cost_profiler.write(run_log_dir, net, {'problem': 'helmholtz', 'architecture': NET_ARCH, 'optimizer': MODE, 'a1': A1, 'a2': A2})

    with torch.no_grad():
        u_pred = net(XY_t).cpu().numpy()
    l2 = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)
    linf = np.max(np.abs(u_pred - u_exact)) / np.max(np.abs(u_exact))
    mse = np.mean((u_pred - u_exact) ** 2)

    U_pred = u_pred.reshape(X.shape)
    U_exact = u_exact.reshape(X.shape)

    heat_exact_path, heat_pred_path, heat_error_path, slice_path = plot_run_outputs(
        X, Y, U_exact, U_pred, run_fig_dir
    )

    hist_array = np.array(history, dtype=float)
    if hist_array.size == 0:
        hist_array = np.empty((0, len(HISTORY_COLS)))
    np.savetxt(
        os.path.join(run_log_dir, 'history.csv'),
        hist_array,
        delimiter=',',
        header=','.join(HISTORY_COLS),
        comments=''
    )
    plot_history_curves(hist_array, run_fig_dir)

    if WALL_TIME_BUDGET_SEC > 0 or ERROR_THRESHOLD_L2 > 0:
        wall_time_payload = {
            'protocol': 'equal-time-helmholtz-v1',
            'wall_time_budget_sec': WALL_TIME_BUDGET_SEC,
            'observed_training_time_sec': elapsed,
            'stopped_by_wall_time': stopped_by_wall_time,
            'completed_iterations': int(hist_array[-1, 0]) + 1,
            'evaluation_every_iterations': RECORD_EVERY,
            'wall_time_check_every_iterations': WALL_TIME_CHECK_EVERY,
            'relative_l2_threshold': ERROR_THRESHOLD_L2,
            'threshold_reached': threshold_time_sec is not None,
            'time_to_threshold_sec': threshold_time_sec,
            'threshold_iteration': threshold_iteration,
            'final_relative_l2': float(l2),
            'final_relative_linf': float(linf),
            'final_mse': float(mse),
        }
        Path(run_log_dir, 'wall_time_metrics.json').write_text(
            json.dumps(wall_time_payload, indent=2) + '\n', encoding='utf-8'
        )

    if rba is not None:
        np.save(os.path.join(run_log_dir, 'rba_final_attention.npy'), rba.weights.cpu().numpy())
    if brdr is not None:
        np.savez(os.path.join(run_log_dir, 'brdr_final_state.npz'),
                 scale=brdr.scale.cpu().numpy(), step=brdr.step,
                 **{f'weights_{index}': value.cpu().numpy() for index, value in enumerate(brdr.weights)},
                 **{f'moments_{index}': value.cpu().numpy() for index, value in enumerate(brdr.moments)})

    ckpt_path = os.path.join(run_ckpt_dir, 'model_final.pt')
    torch.save(net.state_dict(), ckpt_path)

    print(f"[Run {run_idx + 1}] Final L2 {l2:.3e} | L∞ {linf:.3e} | MSE {mse:.3e}")

    return {
        'history': hist_array,
        'u_pred': u_pred,
        'u_star': u_exact,
        'X': X,
        'Y': Y,
        'l2': l2,
        'linf': linf,
        'mse': mse,
        'elapsed': elapsed,
        'time_to_threshold_sec': threshold_time_sec,
        'threshold_iteration': threshold_iteration,
        'stopped_by_wall_time': stopped_by_wall_time,
        'seeds': seeds, 'dataset_sha256': dataset_sha256,
        'ckpt_path': ckpt_path,
        'fig_paths': (heat_exact_path, heat_pred_path, heat_error_path, slice_path),
        'run_dir': run_dir,
        'run_fig_dir': run_fig_dir
    }


run_results = []
if not 0 <= RUN_START_INDEX < NUM_RUNS:
    raise ValueError(f'RUN_START_INDEX must satisfy 0 <= start < {NUM_RUNS}')
for run_idx in range(RUN_START_INDEX, NUM_RUNS):
    seeds = seed_bundle(run_idx, SAMPLE_SEED_BASE, INIT_SEED_BASE, OPTIMIZER_SEED_BASE)
    run_results.append(train_single_run(run_idx, seeds))

if not run_results:
    raise RuntimeError("No runs were executed; check NUM_RUNS.")

# ============================================================
# Aggregate histories
# ============================================================
valid_histories = [res['history'] for res in run_results if res['history'].size]
if valid_histories:
    hist_stack = np.stack(valid_histories, axis=0)
    iters = valid_histories[0][:, 0]
    smooth_sigma = SMOOTH_SIGMA

    total_mean = hist_stack[:, :, 3].mean(axis=0)
    total_std = hist_stack[:, :, 3].std(axis=0)
    l2_mean = hist_stack[:, :, 4].mean(axis=0)
    l2_std = hist_stack[:, :, 4].std(axis=0)
    linf_mean = hist_stack[:, :, 5].mean(axis=0)
    linf_std = hist_stack[:, :, 5].std(axis=0)

    total_lower = np.clip(total_mean - total_std, 1e-12, None)
    total_upper = total_mean + total_std
    l2_lower = np.clip(l2_mean - l2_std, 1e-12, None)
    l2_upper = l2_mean + l2_std
    linf_lower = np.clip(linf_mean - linf_std, 1e-12, None)
    linf_upper = linf_mean + linf_std

    if smooth_sigma > 0:
        total_mean_plot = smooth_curve(total_mean, smooth_sigma)
        total_lower_plot = smooth_curve(total_lower, smooth_sigma)
        total_upper_plot = smooth_curve(total_upper, smooth_sigma)
        l2_mean_plot = smooth_curve(l2_mean, smooth_sigma)
        l2_lower_plot = smooth_curve(l2_lower, smooth_sigma)
        l2_upper_plot = smooth_curve(l2_upper, smooth_sigma)
        linf_mean_plot = smooth_curve(linf_mean, smooth_sigma)
        linf_lower_plot = smooth_curve(linf_lower, smooth_sigma)
        linf_upper_plot = smooth_curve(linf_upper, smooth_sigma)
    else:
        total_mean_plot = total_mean
        total_lower_plot = total_lower
        total_upper_plot = total_upper
        l2_mean_plot = l2_mean
        l2_lower_plot = l2_lower
        l2_upper_plot = l2_upper
        linf_mean_plot = linf_mean
        linf_lower_plot = linf_lower
        linf_upper_plot = linf_upper

    fig, ax = plt.subplots()
    ax.set_yscale('log')
    for hist in valid_histories:
        ax.plot(hist[:, 0], hist[:, 3], color='#a6bddb', alpha=0.3, linewidth=0.8)
    ax.plot(iters, total_mean_plot, color='#045a8d', linewidth=2, label='Mean total loss')
    ax.fill_between(iters, total_lower_plot, total_upper_plot, color='#045a8d', alpha=0.2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Total Loss')
    ax.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(AGG_FIG_DIR, 'loss.pdf'), dpi=300, bbox_inches='tight', pad_inches=0.02, transparent=True)
    plt.close()

    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.plot(iters, l2_mean_plot, color='#1b9e77', linewidth=2, label='Mean L2')
    ax.fill_between(iters, l2_lower_plot, l2_upper_plot, color='#1b9e77', alpha=0.2)
    ax.plot(iters, linf_mean_plot, color='#d95f02', linewidth=2, label='Mean L∞')
    ax.fill_between(iters, linf_lower_plot, linf_upper_plot, color='#d95f02', alpha=0.2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Relative error')
    ax.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(AGG_FIG_DIR, 'error.pdf'), dpi=300, bbox_inches='tight', pad_inches=0.02, transparent=True)
    plt.close()
else:
    print('No history recorded; skipped loss/error plots.')

# ============================================================
# Representative visualization (median final L2; not a selection rule)
# ============================================================
l2_values = np.array([res['l2'] for res in run_results])
representative_idx = int(np.argsort(l2_values)[len(l2_values) // 2])
representative_run = run_results[representative_idx]
representative_path = os.path.join(AGG_CKPT_DIR, 'model_representative.pt')
shutil.copyfile(representative_run['ckpt_path'], representative_path)

plot_run_outputs(
    representative_run['X'],
    representative_run['Y'],
    representative_run['u_star'].reshape(representative_run['X'].shape),
    representative_run['u_pred'].reshape(representative_run['X'].shape),
    AGG_FIG_DIR,
    'representative'
)

# ============================================================
# Final summary
# ============================================================
linf_values = np.array([res['linf'] for res in run_results])
mse_values = np.array([res['mse'] for res in run_results])
elapsed_total = sum(res['elapsed'] for res in run_results)

summary_path = os.path.join(AGG_LOG_DIR, 'summary.csv')
with open(summary_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['run', 'sample_seed', 'init_seed', 'optimizer_order_seed', 'dataset_sha256', 'final_l2', 'final_linf', 'final_mse', 'run_time_sec', 'total_elapsed_time_sec'])
    for idx, res in enumerate(run_results, start=1):
        seeds = res['seeds']
        writer.writerow([idx, seeds['sample_seed'], seeds['init_seed'], seeds['optimizer_order_seed'], res['dataset_sha256'], res['l2'], res['linf'], res['mse'], res['elapsed'], elapsed_total])
    writer.writerow([])
    writer.writerow(['representative_run_index', representative_idx + 1])
    writer.writerow(['representative_rule', 'median_final_l2_for_visualization_only'])
    writer.writerow(['total_elapsed_time_sec', elapsed_total])
    writer.writerow(['l2_mean', l2_values.mean()])
    writer.writerow(['l2_std', l2_values.std()])
    writer.writerow(['linf_mean', linf_values.mean()])
    writer.writerow(['linf_std', linf_values.std()])
    writer.writerow(['mse_mean', mse_values.mean()])
    writer.writerow(['mse_std', mse_values.std()])

print("\n===== Final Results =====")
print(f"Mode: {MODE}")
print(f"Runs executed this invocation: {len(run_results)} | Target total runs: {NUM_RUNS} | "
      f"Seed bundles: {[res['seeds'] for res in run_results]}")
print(f"Total training time: {elapsed_total:.2f}s")
print(f"Representative run for plots: #{representative_idx + 1} "
      f"(median final L2; no checkpoint selection)")
print(f"L2 mean ± std: {l2_values.mean():.3e} ± {l2_values.std():.3e}")
print(f"L∞ mean ± std: {linf_values.mean():.3e} ± {linf_values.std():.3e}")
print(f"MSE mean ± std: {mse_values.mean():.3e} ± {mse_values.std():.3e}")
