#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
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
from common.optimization import canonical_optimizer, project_conflicting_gradients
from common.diagnostics import DiagnosticsRecorder, collect_task_gradients
from common.cost_benchmark import CudaCostProfiler
from common.scheduling import build_learning_rate_scheduler
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
EPOCHS = int(os.getenv('EPOCHS', '20000'))
LR = float(os.getenv('LEARNING_RATE', '1e-3'))
LR_SCHEDULE = os.getenv('LR_SCHEDULE', 'constant')
LR_MIN = float(os.getenv('LR_MIN', '1e-5'))
WARMUP_EPOCHS = int(os.getenv('WARMUP_EPOCHS', '0'))
RECORD_EVERY = int(os.getenv('RECORD_EVERY', '100'))
DIAGNOSTICS_EVERY = int(os.getenv('DIAGNOSTICS_EVERY', str(RECORD_EVERY)))
NUM_RUNS = int(os.getenv('NUM_RUNS', '5'))
RUN_START_INDEX = int(os.getenv('RUN_START_INDEX', '0'))
SAMPLE_SEED_BASE = int(os.getenv('SAMPLE_SEED_BASE', '1234'))
INIT_SEED_BASE = int(os.getenv('INIT_SEED_BASE', '2234'))
OPTIMIZER_SEED_BASE = int(os.getenv('OPTIMIZER_SEED_BASE', '3234'))

NUM_F = int(os.getenv('CAVITY_NUM_F', '1000'))
NUM_B_TOP = int(os.getenv('CAVITY_NUM_TOP', '300'))
NUM_B_WALL = int(os.getenv('CAVITY_NUM_WALL', '300'))
DYNAMIC_SAMPLING = os.getenv('CAVITY_DYNAMIC_SAMPLING', '0').strip().lower() in ('1', 'true', 'yes')
RESAMPLE_EVERY = int(os.getenv('CAVITY_RESAMPLE_EVERY', '1'))
NO_PLOT = os.getenv('NO_PLOT', '0').strip().lower() in ('1', 'true', 'yes')

REYNOLDS = 100.0

SAVE_DIR = f'results_lid_driven_cavity_{RUN_TAG}'
RUNS_DIR = os.path.join(SAVE_DIR, 'runs')
AGG_DIR = os.path.join(SAVE_DIR, 'aggregate')
AGG_FIG_DIR = os.path.join(AGG_DIR, 'figures')
AGG_LOG_DIR = os.path.join(AGG_DIR, 'logs')
AGG_CKPT_DIR = os.path.join(AGG_DIR, 'checkpoints')
for path in (SAVE_DIR, RUNS_DIR, AGG_DIR, AGG_FIG_DIR, AGG_LOG_DIR, AGG_CKPT_DIR):
    os.makedirs(path, exist_ok=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'lid_driven_cavity')
GENERATED_DATA_DIR = Path(os.getenv(
    'CAVITY_GENERATED_DATA_DIR',
    str(REPO_ROOT / 'generated_data' / 'lid_driven_cavity'),
))

# ============================================================
# Device
# ============================================================
device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print(f'Using device: {device}')

# ============================================================
# Normalization (NN input only)
# ============================================================
def normalize_xy(X):
    x = X[:, 0:1]
    y = X[:, 1:2]
    x_hat = 2.0 * x - 1.0
    y_hat = 2.0 * y - 1.0
    return torch.cat([x_hat, y_hat], dim=1)

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
    layers = [2, 50, 50, 50, 50, 2]
    return build_shared_model(
        NET_ARCH, layers, lda_stabilized=LDA_STABILIZED,
        lda_residual_scale=LDA_RESIDUAL_SCALE,
    )

# ============================================================
# Helpers
# ============================================================
def psi_to_velocity(out, X, create_graph=True):
    psi = out[:, 0:1]
    psi_xy = grad(
        outputs=psi,
        inputs=X,
        grad_outputs=torch.ones_like(psi),
        create_graph=create_graph
    )[0]
    u = psi_xy[:, 1:2]
    v = -psi_xy[:, 0:1]
    return u, v


def navier_stokes_residual(out, X):
    psi = out[:, 0:1]
    p = out[:, 1:2]
    psi_xy = grad(
        outputs=psi,
        inputs=X,
        grad_outputs=torch.ones_like(psi),
        create_graph=True
    )[0]
    u = psi_xy[:, 1:2]
    v = -psi_xy[:, 0:1]

    du = grad(u, X, torch.ones_like(u), create_graph=True)[0]
    dv = grad(v, X, torch.ones_like(v), create_graph=True)[0]
    dp = grad(p, X, torch.ones_like(p), create_graph=True)[0]

    u_x = du[:, 0:1]
    u_y = du[:, 1:2]
    v_x = dv[:, 0:1]
    v_y = dv[:, 1:2]
    p_x = dp[:, 0:1]
    p_y = dp[:, 1:2]

    u_xx = grad(u_x, X, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = grad(u_y, X, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
    v_xx = grad(v_x, X, torch.ones_like(v_x), create_graph=True)[0][:, 0:1]
    v_yy = grad(v_y, X, torch.ones_like(v_y), create_graph=True)[0][:, 1:2]

    res_u = u * u_x + v * u_y + p_x - (u_xx + u_yy) / REYNOLDS
    res_v = u * v_x + v * v_y + p_y - (v_xx + v_yy) / REYNOLDS

    return res_u, res_v


def predict_velocity(net, coords, batch_size=4096):
    preds = []
    for chunk in torch.split(coords, batch_size):
        chunk = chunk.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            out = net(normalize_xy(chunk))
            u, v = psi_to_velocity(out, chunk, create_graph=False)
        preds.append(torch.cat([u, v], dim=1).detach().cpu())
    return torch.cat(preds, dim=0).numpy()

# ============================================================
# Data
# ============================================================
def sampler(sample_seed):
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    dataset_path = GENERATED_DATA_DIR / f'seed_{sample_seed}_nf{NUM_F}_ntop{NUM_B_TOP}_nwall{NUM_B_WALL}.npz'

    def create_dataset():
        lhs_rng, top_rng, wall_rng = spawned_rngs(sample_seed, 3)
        X_f = lb + (ub - lb) * latin_hypercube(2, NUM_F, lhs_rng)
        X_top = np.hstack([top_rng.random((NUM_B_TOP, 1)), np.ones((NUM_B_TOP, 1))])
        n_each = NUM_B_WALL // 3
        left = np.hstack([np.zeros((n_each, 1)), wall_rng.random((n_each, 1))])
        right = np.hstack([np.ones((n_each, 1)), wall_rng.random((n_each, 1))])
        n_bottom = NUM_B_WALL - 2 * n_each
        bottom = np.hstack([wall_rng.random((n_bottom, 1)), np.zeros((n_bottom, 1))])
        return {'X_f': X_f, 'X_top': X_top, 'X_wall': np.vstack([left, right, bottom])}

    arrays, dataset_sha256 = load_or_create_npz(dataset_path, create_dataset)

    return (
        torch.tensor(arrays['X_f'], dtype=torch.float32, device=device, requires_grad=True),
        torch.tensor(arrays['X_top'], dtype=torch.float32, device=device, requires_grad=True),
        torch.tensor(arrays['X_wall'], dtype=torch.float32, device=device, requires_grad=True),
        dataset_path, dataset_sha256,
    )


def dynamic_sample_epoch(sample_seed, epoch):
    rng = epoch_rng(sample_seed, epoch, RESAMPLE_EVERY)
    X_f = fast_lhs(rng, NUM_F, 2)
    top_x = fast_lhs(rng, NUM_B_TOP, 1)
    X_top = np.hstack([top_x, np.ones((NUM_B_TOP, 1), dtype=np.float32)])
    wall_values = fast_lhs(rng, NUM_B_WALL, 1)
    n_each = NUM_B_WALL // 3
    left = np.hstack([np.zeros((n_each, 1), dtype=np.float32), wall_values[:n_each]])
    right = np.hstack([np.ones((n_each, 1), dtype=np.float32), wall_values[n_each:2 * n_each]])
    n_bottom = NUM_B_WALL - 2 * n_each
    bottom = np.hstack([wall_values[2 * n_each:2 * n_each + n_bottom], np.zeros((n_bottom, 1), dtype=np.float32)])
    X_wall = np.vstack([left, right, bottom])
    return (
        torch.tensor(X_f, dtype=torch.float32, device=device, requires_grad=True),
        torch.tensor(X_top, dtype=torch.float32, device=device, requires_grad=True),
        torch.tensor(X_wall, dtype=torch.float32, device=device, requires_grad=True),
    )


def dynamic_sampling_descriptor(sample_seed):
    path = GENERATED_DATA_DIR / f'seed_{sample_seed}_dynamic_nf{NUM_F}_ntop{NUM_B_TOP}_nwall{NUM_B_WALL}_r{RESAMPLE_EVERY}.json'
    return write_sampling_descriptor(path, {
        'algorithm': 'deterministic_randomized_latin_hypercube',
        'sample_seed': sample_seed, 'num_residual': NUM_F,
        'num_top': NUM_B_TOP, 'num_wall': NUM_B_WALL,
        'resample_every': RESAMPLE_EVERY, 'domain': [[0.0, 1.0], [0.0, 1.0]],
    })

# ============================================================
# Reference data
# ============================================================
u_ref = np.genfromtxt(os.path.join(DATA_DIR, 'velocity_u.csv'), delimiter=',')
v_ref = np.genfromtxt(os.path.join(DATA_DIR, 'velocity_v.csv'), delimiter=',')
nx, ny = u_ref.shape
x = np.linspace(0.0, 1.0, nx)
y = np.linspace(0.0, 1.0, ny)
X, Y = np.meshgrid(x, y, indexing='ij')
X_star = np.hstack([X.flatten()[:, None], Y.flatten()[:, None]])
u_star = u_ref.flatten()[:, None]
v_star = v_ref.flatten()[:, None]
vel_star = np.sqrt(u_star ** 2 + v_star ** 2)
X_star_t = torch.tensor(X_star, dtype=torch.float32, device=device)

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



SLICE_Y = 0.8
EXACT_COLOR = '#2b2b2b'
PRED_COLOR = '#d55e00'
SMOOTH_SIGMA = 3.0


def smooth_curve(values, sigma):
    if len(values) == 0 or sigma <= 0:
        return values
    return gaussian_filter1d(values, sigma=sigma, mode='nearest')


def plot_run_outputs(X, Y, u_exact, v_exact, u_pred, v_pred, fig_dir, prefix=''):
    os.makedirs(fig_dir, exist_ok=True)
    vel_exact = np.sqrt(u_exact ** 2 + v_exact ** 2)
    vel_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)

    if prefix:
        mag_exact_name = f'velocity_exact_{prefix}.pdf'
        mag_pred_name = f'velocity_pred_{prefix}.pdf'
        mag_error_name = f'velocity_error_{prefix}.pdf'
        comp_name = f'velocity_components_{prefix}.pdf'
        slice_name = f'slice_y{SLICE_Y:.2f}_{prefix}.pdf'
    else:
        mag_exact_name = 'velocity_exact.pdf'
        mag_pred_name = 'velocity_pred.pdf'
        mag_error_name = 'velocity_error.pdf'
        comp_name = 'velocity_components.pdf'
        slice_name = f'slice_y{SLICE_Y:.2f}.pdf'

    mag_exact_path = os.path.join(fig_dir, mag_exact_name)
    mag_pred_path = os.path.join(fig_dir, mag_pred_name)
    mag_error_path = os.path.join(fig_dir, mag_error_name)
    comp_path = os.path.join(fig_dir, comp_name)
    slice_path = os.path.join(fig_dir, slice_name)

    fig, ax = plt.subplots(figsize=(5, 4))
    pcm = ax.pcolormesh(X, Y, vel_exact, shading='gouraud')
    ax.set_title('Exact |u|')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(pcm, ax=ax)
    fig.tight_layout()
    plt.savefig(mag_exact_path, dpi=300, bbox_inches='tight', pad_inches=0.02, transparent=True)
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 4))
    pcm = ax.pcolormesh(X, Y, vel_pred, shading='gouraud')
    ax.set_title('Prediction |u|')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(pcm, ax=ax)
    fig.tight_layout()
    plt.savefig(mag_pred_path, dpi=300, bbox_inches='tight', pad_inches=0.02, transparent=True)
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 4))
    pcm = ax.pcolormesh(X, Y, np.abs(vel_exact - vel_pred), shading='gouraud')
    ax.set_title('Abs Error |u|')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(pcm, ax=ax)
    fig.tight_layout()
    plt.savefig(mag_error_path, dpi=300, bbox_inches='tight', pad_inches=0.02, transparent=True)
    plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    pcm = axes[0, 0].pcolormesh(X, Y, u_exact, shading='gouraud')
    axes[0, 0].set_title('Exact u')
    fig.colorbar(pcm, ax=axes[0, 0])
    pcm = axes[0, 1].pcolormesh(X, Y, u_pred, shading='gouraud')
    axes[0, 1].set_title('Pred u')
    fig.colorbar(pcm, ax=axes[0, 1])
    pcm = axes[0, 2].pcolormesh(X, Y, np.abs(u_exact - u_pred), shading='gouraud')
    axes[0, 2].set_title('|u error|')
    fig.colorbar(pcm, ax=axes[0, 2])
    pcm = axes[1, 0].pcolormesh(X, Y, v_exact, shading='gouraud')
    axes[1, 0].set_title('Exact v')
    fig.colorbar(pcm, ax=axes[1, 0])
    pcm = axes[1, 1].pcolormesh(X, Y, v_pred, shading='gouraud')
    axes[1, 1].set_title('Pred v')
    fig.colorbar(pcm, ax=axes[1, 1])
    pcm = axes[1, 2].pcolormesh(X, Y, np.abs(v_exact - v_pred), shading='gouraud')
    axes[1, 2].set_title('|v error|')
    fig.colorbar(pcm, ax=axes[1, 2])
    fig.tight_layout()
    plt.savefig(comp_path, dpi=300, bbox_inches='tight', pad_inches=0.02, transparent=True)
    plt.close()

    y_values = Y[0, :]
    x_values = X[:, 0]
    idx = int(np.abs(y_values - SLICE_Y).argmin())
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(x_values, vel_exact[:, idx], color=EXACT_COLOR, label='Exact')
    ax.plot(x_values, vel_pred[:, idx], color=PRED_COLOR, linestyle='--', label='Prediction')
    ax.set_title(f'y={y_values[idx]:.2f}')
    ax.set_xlabel('x')
    ax.set_ylabel('|u(x, y)|')
    ax.legend()
    fig.tight_layout()
    plt.savefig(slice_path, dpi=300, bbox_inches='tight', pad_inches=0.02, transparent=True)
    plt.close()

    return mag_exact_path, mag_pred_path, mag_error_path, comp_path, slice_path


def plot_history_curves(hist_array, fig_dir, prefix=''):
    if hist_array.size == 0:
        return
    os.makedirs(fig_dir, exist_ok=True)
    loss_name = f'loss_{prefix}.pdf' if prefix else 'loss.pdf'
    error_name = f'error_{prefix}.pdf' if prefix else 'error.pdf'
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.plot(hist_array[:, 0], hist_array[:, 4], color='#1f78b4')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Total Loss')
    fig.tight_layout()
    plt.savefig(os.path.join(fig_dir, loss_name), dpi=300, bbox_inches='tight', pad_inches=0.02, transparent=True)
    plt.close()

    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.plot(hist_array[:, 0], hist_array[:, 5], color='#33a02c', label='L2')
    ax.plot(hist_array[:, 0], hist_array[:, 6], color='#e31a1c', label='L∞')
    ax.legend()
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Relative error')
    fig.tight_layout()
    plt.savefig(os.path.join(fig_dir, error_name), dpi=300, bbox_inches='tight', pad_inches=0.02, transparent=True)
    plt.close()


# ============================================================
# Train (possibly multiple runs)
# ============================================================
HISTORY_COLS = ['iter', 'pde', 'bc_top', 'bc_wall', 'total', 'l2', 'linf', 'mse']

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
        X_f, X_top, X_wall = dynamic_sample_epoch(seeds['sample_seed'], 0)
    else:
        X_f, X_top, X_wall, dataset_path, dataset_sha256 = sampler(seeds['sample_seed'])
    net = build_model().to(device)
    warmup_cosine = LR_SCHEDULE.strip().lower() == 'linear_warmup_then_cosine'
    opt = Adam(net.parameters(), lr=LR_MIN if warmup_cosine else LR)
    scheduler = None if warmup_cosine else build_learning_rate_scheduler(opt, LR_SCHEDULE, EPOCHS, LR_MIN)
    recorder = DiagnosticsRecorder(net, run_log_dir, device, DIAGNOSTICS_EVERY)
    cost_profiler = CudaCostProfiler(device, EPOCHS)
    config = {'problem': 'lid_driven_cavity', 'mode': MODE, 'network': NET_ARCH,
              'lda_stabilized': LDA_STABILIZED if NET_ARCH == 'lda' else None,
              'lda_residual_scale': LDA_RESIDUAL_SCALE if NET_ARCH == 'lda' else None,
              'layers': [2, 50, 50, 50, 50, 2], 'epochs': EPOCHS,
              'learning_rate': LR, 'lr_schedule': LR_SCHEDULE,
              'minimum_learning_rate': LR_MIN, 'warmup_epochs': WARMUP_EPOCHS,
              'record_every': RECORD_EVERY, 'dynamic_sampling': DYNAMIC_SAMPLING,
              'resample_every': RESAMPLE_EVERY if DYNAMIC_SAMPLING else None,
              'num_residual': NUM_F, 'num_top': NUM_B_TOP, 'num_wall': NUM_B_WALL,
              'reynolds': REYNOLDS, 'optimizer': 'Adam', 'dtype': 'float32'}
    write_run_metadata(os.path.join(run_log_dir, 'run_metadata.json'), config, seeds,
                       dataset_path, dataset_sha256)

    history = []
    start = time.time()

    def forward_losses():
        out_f = net(normalize_xy(X_f))
        res_u, res_v = navier_stokes_residual(out_f, X_f)
        L_pde = torch.mean(res_u ** 2 + res_v ** 2)

        out_top = net(normalize_xy(X_top))
        u_top, v_top = psi_to_velocity(out_top, X_top)
        L_bc_top = torch.mean((u_top - 1.0) ** 2 + v_top ** 2)

        out_wall = net(normalize_xy(X_wall))
        u_wall, v_wall = psi_to_velocity(out_wall, X_wall)
        L_bc_wall = torch.mean(u_wall ** 2 + v_wall ** 2)

        return L_pde, L_bc_top, L_bc_wall

    for ep in range(EPOCHS):
        if DYNAMIC_SAMPLING:
            X_f, X_top, X_wall = dynamic_sample_epoch(seeds['sample_seed'], ep)
        cost_profiler.start_step(ep)
        if warmup_cosine:
            current_lr = warmup_cosine_learning_rate(ep, EPOCHS, LR, LR_MIN, WARMUP_EPOCHS)
            for group in opt.param_groups:
                group['lr'] = current_lr
        L_pde, L_bc_top, L_bc_wall = forward_losses()
        L = L_pde + L_bc_top + L_bc_wall
        losses = [L_pde, L_bc_top, L_bc_wall]
        task_names = ['pde', 'bc_top', 'bc_wall']
        params = list(net.parameters())
        named_params = list(net.named_parameters())
        diagnostic_grads = None
        projected_grads = None
        if MODE == 'sum' and recorder.should_record(ep, EPOCHS):
            diagnostic_grads = collect_task_gradients(losses, params)
        recorder.start_step()

        if MODE == 'sum':
            opt.zero_grad()
            L.backward()
            opt.step()
        elif MODE == 'pcgrad':
            grads = []
            for loss in losses:
                opt.zero_grad()
                loss.backward(retain_graph=True)
                grads.append([
                    (p.grad.clone() if p.grad is not None else torch.zeros_like(p))
                    for p in params
                ])
            final_grad = project_conflicting_gradients(grads, order_generator)
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
        if recorder.should_record(ep, EPOCHS):
            recorder.record(ep, task_names, diagnostic_grads, named_params, projected_grads,
                            normalize_xy(X_f[:min(1024, len(X_f))]).detach(),
                            {'learning_rate': opt.param_groups[0]['lr']})

        if ep % RECORD_EVERY == 0 or ep == EPOCHS - 1:
            preds = predict_velocity(net, X_star_t)
            vel_pred = np.sqrt(preds[:, 0:1] ** 2 + preds[:, 1:2] ** 2)
            l2 = np.linalg.norm(vel_pred - vel_star) / np.linalg.norm(vel_star)
            linf = np.max(np.abs(vel_pred - vel_star)) / np.max(np.abs(vel_star))
            mse = np.mean((vel_pred - vel_star) ** 2)
            history.append([
                ep,
                L_pde.item(),
                L_bc_top.item(),
                L_bc_wall.item(),
                L.item(),
                l2,
                linf,
                mse
            ])
            print(f"[{MODE.upper()}][Run {run_idx + 1}] {ep:6d} | "
                  f"PDE {L_pde.item():.2e} BC_top {L_bc_top.item():.2e} "
                  f"BC_wall {L_bc_wall.item():.2e} | L2 {l2:.2e} L∞ {linf:.2e} MSE {mse:.2e}")

    elapsed = time.time() - start
    recorder.finalize(elapsed)
    cost_profiler.write(run_log_dir, net, {'problem': 'lid_driven_cavity', 'architecture': NET_ARCH, 'optimizer': MODE})

    preds = predict_velocity(net, X_star_t)
    vel_pred = np.sqrt(preds[:, 0:1] ** 2 + preds[:, 1:2] ** 2)
    l2 = np.linalg.norm(vel_pred - vel_star) / np.linalg.norm(vel_star)
    linf = np.max(np.abs(vel_pred - vel_star)) / np.max(np.abs(vel_star))
    mse = np.mean((vel_pred - vel_star) ** 2)

    u_pred = preds[:, 0:1].reshape(X.shape)
    v_pred = preds[:, 1:2].reshape(X.shape)

    if NO_PLOT:
        heat_exact_path = heat_pred_path = heat_error_path = comp_path = slice_path = None
    else:
        heat_exact_path, heat_pred_path, heat_error_path, comp_path, slice_path = plot_run_outputs(
            X, Y, u_star.reshape(X.shape), v_star.reshape(X.shape), u_pred, v_pred, run_fig_dir
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
    if not NO_PLOT:
        plot_history_curves(hist_array, run_fig_dir)

    ckpt_path = os.path.join(run_ckpt_dir, 'model_final.pt')
    torch.save(net.state_dict(), ckpt_path)

    print(f"[Run {run_idx + 1}] Final L2 {l2:.3e} | L∞ {linf:.3e} | MSE {mse:.3e}")

    return {
        'history': hist_array,
        'l2': l2,
        'linf': linf,
        'mse': mse,
        'elapsed': elapsed,
        'seeds': seeds,
        'dataset_sha256': dataset_sha256,
        'ckpt_path': ckpt_path,
        'u_pred': u_pred,
        'v_pred': v_pred,
        'fig_paths': (heat_exact_path, heat_pred_path, heat_error_path, comp_path, slice_path),
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

    total_mean = hist_stack[:, :, 4].mean(axis=0)
    total_std = hist_stack[:, :, 4].std(axis=0)
    l2_mean = hist_stack[:, :, 5].mean(axis=0)
    l2_std = hist_stack[:, :, 5].std(axis=0)
    linf_mean = hist_stack[:, :, 6].mean(axis=0)
    linf_std = hist_stack[:, :, 6].std(axis=0)

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
        ax.plot(hist[:, 0], hist[:, 4], color='#a6bddb', alpha=0.3, linewidth=0.8)
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

if not NO_PLOT:
    plot_run_outputs(
        X, Y, u_star.reshape(X.shape), v_star.reshape(X.shape),
        representative_run['u_pred'], representative_run['v_pred'], AGG_FIG_DIR, 'representative'
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
