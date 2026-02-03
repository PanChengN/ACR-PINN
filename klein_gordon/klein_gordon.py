#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import os
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
from torch.optim import Adam
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from pyDOE import lhs
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.dpi'] = 300

# ============================================================
# Config
# ============================================================
MODE = os.getenv('MODE', 'pinn')     # 'pinn' | 'pcgrad'
NET_ARCH = os.getenv('NET_ARCH', 'mlp')    # 'mlp' | 'attention'
RUN_TAG = f'{MODE}_{NET_ARCH}'
EPOCHS = 40000
LR = 1e-3
RECORD_EVERY = 1
BASE_SEED = 1234
NUM_RUNS = 5

NUM_F = 10000
NUM_B = 200
NUM_IC = 200

ALPHA = -1.0
BETA = 0.0
GAMMA = 1.0
K_POWER = 3

SAVE_DIR = f'results_klein_gordon_{RUN_TAG}'
RUNS_DIR = os.path.join(SAVE_DIR, 'runs')
AGG_DIR = os.path.join(SAVE_DIR, 'aggregate')
AGG_FIG_DIR = os.path.join(AGG_DIR, 'figures')
AGG_LOG_DIR = os.path.join(AGG_DIR, 'logs')
AGG_CKPT_DIR = os.path.join(AGG_DIR, 'checkpoints')
for path in (SAVE_DIR, RUNS_DIR, AGG_DIR, AGG_FIG_DIR, AGG_LOG_DIR, AGG_CKPT_DIR):
    os.makedirs(path, exist_ok=True)


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
def normalize_xt(X):
    x = X[:, 0:1]
    t = X[:, 1:2]
    x_hat = 2.0 * x - 1.0
    t_hat = 2.0 * t - 1.0
    return torch.cat([x_hat, t_hat], dim=1)

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
    if NET_ARCH == 'mlp':
        return MLP(layers)
    if NET_ARCH == 'attention':
        return NetAttentionDynamic(layers)
    raise ValueError(f"Unknown NET_ARCH='{NET_ARCH}'")

# ============================================================
# Exact / forcing
# ============================================================
def exact_solution(X):
    x = X[:, 0:1]
    t = X[:, 1:2]
    return x * torch.cos(5 * np.pi * t) + (x * t) ** 3


def exact_ut(X):
    x = X[:, 0:1]
    t = X[:, 1:2]
    return -5 * np.pi * x * torch.sin(5 * np.pi * t) + 3 * (x ** 3) * (t ** 2)


def forcing_term(X):
    x = X[:, 0:1]
    t = X[:, 1:2]
    u_tt = -(5 * np.pi) ** 2 * x * torch.cos(5 * np.pi * t) + 6 * (x ** 3) * t
    u_xx = 6 * x * (t ** 3)
    u = exact_solution(X)
    return u_tt + ALPHA * u_xx + BETA * u + GAMMA * (u ** K_POWER)

# ============================================================
# Klein-Gordon residual (physical coordinates)
# ============================================================
def klein_gordon_residual(u, X):
    du = grad(u, X, torch.ones_like(u), create_graph=True)[0]
    u_x = du[:, 0:1]
    u_t = du[:, 1:2]
    u_xx = grad(u_x, X, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_tt = grad(u_t, X, torch.ones_like(u_t), create_graph=True)[0][:, 1:2]
    return u_tt + ALPHA * u_xx + BETA * u + GAMMA * (u ** K_POWER) - forcing_term(X)

# ============================================================
# Data
# ============================================================
def sampler():
    X_f = lhs(2, NUM_F)
    X_f[:, 0:1] = np.clip(X_f[:, 0:1], 0, 1)
    X_f[:, 1:2] = np.clip(X_f[:, 1:2], 0, 1)

    n_bc = NUM_B // 2
    t_bc = np.random.rand(n_bc, 1)
    X_bc = np.vstack([
        np.hstack([np.zeros((n_bc, 1)), t_bc]),
        np.hstack([np.ones((NUM_B - n_bc, 1)), np.random.rand(NUM_B - n_bc, 1)])
    ])

    x_ic = np.random.rand(NUM_IC, 1)
    X_ic = np.hstack([x_ic, np.zeros((NUM_IC, 1))])

    return (
        torch.tensor(X_f, dtype=torch.float32, device=device, requires_grad=True),
        torch.tensor(X_bc, dtype=torch.float32, device=device),
        torch.tensor(X_ic, dtype=torch.float32, device=device, requires_grad=True),
    )

# ============================================================
# PCGrad
# ============================================================
def pcgrad(grads):
    proj = [[g.clone() for g in task] for task in grads]
    n = len(proj)
    for i in range(n):
        for j in torch.randperm(n):
            j = j.item()
            if i == j:
                continue
            dot = sum((gi * gj).sum() for gi, gj in zip(proj[i], grads[j]))
            if dot < 0:
                norm = sum((gj ** 2).sum() for gj in grads[j])
                proj[i] = [gi - dot / norm * gj for gi, gj in zip(proj[i], grads[j])]
    return [sum(gs) for gs in zip(*proj)]



SLICE_TIMES = [0.0, 0.5, 1.0]
EXACT_COLOR = '#2b2b2b'
PRED_COLOR = '#d55e00'
SMOOTH_SIGMA = 3.0


def smooth_curve(values, sigma):
    if len(values) == 0 or sigma <= 0:
        return values
    return gaussian_filter1d(values, sigma=sigma, mode='nearest')


def plot_run_outputs(X, T, Exact, x, t, X_star, u_pred, fig_dir, prefix=''):
    os.makedirs(fig_dir, exist_ok=True)
    U_pred = u_pred.reshape(X.shape)
    if prefix:
        heatmap_name = f'heatmap_exact_{prefix}.pdf'
        pred_name = f'heatmap_pred_{prefix}.pdf'
        error_name = f'heatmap_error_{prefix}.pdf'
        slice_prefix = f'time_slice_{prefix}_t'
    else:
        heatmap_name = 'heatmap_exact.pdf'
        pred_name = 'heatmap_pred.pdf'
        error_name = 'heatmap_error.pdf'
        slice_prefix = 'time_slice_t'
    exact_path = os.path.join(fig_dir, heatmap_name)
    pred_path = os.path.join(fig_dir, pred_name)
    error_path = os.path.join(fig_dir, error_name)

    fig, ax = plt.subplots(figsize=(5, 4))
    pcm = ax.pcolormesh(X, T, Exact, shading='gouraud')
    ax.set_title('Exact')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    fig.colorbar(pcm, ax=ax)
    fig.tight_layout()
    plt.savefig(exact_path, dpi=300, bbox_inches='tight', pad_inches=0.02, transparent=True)
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 4))
    pcm = ax.pcolormesh(X, T, U_pred, shading='gouraud')
    ax.set_title('Prediction')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    fig.colorbar(pcm, ax=ax)
    fig.tight_layout()
    plt.savefig(pred_path, dpi=300, bbox_inches='tight', pad_inches=0.02, transparent=True)
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 4))
    pcm = ax.pcolormesh(X, T, np.abs(Exact - U_pred), shading='gouraud')
    ax.set_title('Abs Error')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    fig.colorbar(pcm, ax=ax)
    fig.tight_layout()
    plt.savefig(error_path, dpi=300, bbox_inches='tight', pad_inches=0.02, transparent=True)
    plt.close()

    t_values = t.flatten()
    x_values = x.flatten()
    slice_paths = []
    for target in SLICE_TIMES:
        idx = int(np.abs(t_values - target).argmin())
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(x_values, Exact[idx, :], color=EXACT_COLOR, label='Exact')
        ax.plot(x_values, U_pred[idx, :], color=PRED_COLOR, linestyle='--', label='Prediction')
        ax.set_title(f't={t_values[idx]:.2f}')
        ax.set_xlabel('x')
        ax.set_ylabel('u(x, t)')
        ax.legend()
        fig.tight_layout()
        slice_name = f'{slice_prefix}{t_values[idx]:.2f}.pdf'
        slice_path = os.path.join(fig_dir, slice_name)
        plt.savefig(slice_path, dpi=300, bbox_inches='tight', pad_inches=0.02, transparent=True)
        plt.close()
        slice_paths.append(slice_path)
    return exact_path, pred_path, error_path, slice_paths


def plot_history_curves(hist_array, fig_dir, prefix=''):
    if hist_array.size == 0:
        return
    os.makedirs(fig_dir, exist_ok=True)
    loss_name = f'loss_{prefix}.pdf' if prefix else 'loss.pdf'
    error_name = f'error_{prefix}.pdf' if prefix else 'error.pdf'
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.plot(hist_array[:, 0], hist_array[:, 5], color='#1f78b4')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Total Loss')
    fig.tight_layout()
    plt.savefig(os.path.join(fig_dir, loss_name), dpi=300, bbox_inches='tight', pad_inches=0.02, transparent=True)
    plt.close()

    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.plot(hist_array[:, 0], hist_array[:, 6], color='#33a02c', label='L2')
    ax.plot(hist_array[:, 0], hist_array[:, 7], color='#e31a1c', label='L∞')
    ax.legend()
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Relative error')
    fig.tight_layout()
    plt.savefig(os.path.join(fig_dir, error_name), dpi=300, bbox_inches='tight', pad_inches=0.02, transparent=True)
    plt.close()


# ============================================================
# Train (possibly multiple runs)
# ============================================================
HISTORY_COLS = ['iter', 'pde', 'bc', 'ic', 'ic_t', 'total', 'l2', 'linf']

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


def train_single_run(run_idx, seed):
    print(f"\n=== Run {run_idx + 1}/{NUM_RUNS} | seed={seed} ===")
    set_seeds(seed)
    run_dir = os.path.join(RUNS_DIR, f'run_{run_idx + 1}')
    run_fig_dir = os.path.join(run_dir, 'figures')
    run_log_dir = os.path.join(run_dir, 'logs')
    run_ckpt_dir = os.path.join(run_dir, 'checkpoints')
    for path in (run_dir, run_fig_dir, run_log_dir, run_ckpt_dir):
        os.makedirs(path, exist_ok=True)

    X_f, X_bc, X_ic = sampler()
    X_bc_norm = normalize_xt(X_bc)

    with torch.no_grad():
        u_bc = exact_solution(X_bc)
        u_ic = exact_solution(X_ic)
        ut_ic = exact_ut(X_ic)

    net = build_model().to(device)
    opt = Adam(net.parameters(), lr=LR)

    N_test = 200
    x = np.linspace(0, 1, N_test)
    t = np.linspace(0, 1, N_test)
    X, T = np.meshgrid(x, t)
    X_star = np.hstack([X.flatten()[:, None], T.flatten()[:, None]])
    u_star = exact_solution(torch.tensor(X_star, dtype=torch.float32, device=device)).detach().cpu().numpy()
    X_star_t = torch.tensor(X_star, dtype=torch.float32, device=device)
    X_star_norm = normalize_xt(X_star_t)

    history = []
    start = time.time()

    def forward_losses():
        X_f_norm = normalize_xt(X_f)
        u_f = net(X_f_norm)
        res = klein_gordon_residual(u_f, X_f)
        L_pde = torch.mean(res ** 2)

        u_b_pred = net(X_bc_norm)
        L_bc = torch.mean((u_b_pred - u_bc) ** 2)

        u_ic_pred = net(normalize_xt(X_ic))
        L_ic = torch.mean((u_ic_pred - u_ic) ** 2)
        u_t_pred = grad(u_ic_pred, X_ic, torch.ones_like(u_ic_pred), create_graph=True)[0][:, 1:2]
        L_ic_t = torch.mean((u_t_pred - ut_ic) ** 2)

        return L_pde, L_bc, L_ic, L_ic_t

    def capture_state():
        return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    best_l2 = float('inf')
    best_linf = float('inf')
    best_state = None
    best_pred = None

    for ep in range(EPOCHS):
        L_pde, L_bc, L_ic, L_ic_t = forward_losses()
        L = L_pde + L_bc + L_ic + L_ic_t

        if MODE == 'pinn':
            opt.zero_grad()
            L.backward()
            opt.step()
        elif MODE == 'pcgrad':
            params = list(net.parameters())
            grads = []
            for loss in [L_pde, L_bc, L_ic, L_ic_t]:
                opt.zero_grad()
                loss.backward(retain_graph=True)
                grads.append([
                    (p.grad.clone() if p.grad is not None else torch.zeros_like(p))
                    for p in params
                ])
            final_grad = pcgrad(grads)
            opt.zero_grad()
            for p, g in zip(params, final_grad):
                p.grad = g
            opt.step()
        else:
            raise ValueError(f"Unknown MODE='{MODE}'")

        if ep % RECORD_EVERY == 0:
            with torch.no_grad():
                pred_snapshot = net(X_star_norm).cpu().numpy()
            l2 = np.linalg.norm(pred_snapshot - u_star) / np.linalg.norm(u_star)
            linf = np.max(np.abs(pred_snapshot - u_star)) / np.max(np.abs(u_star))
            history.append([ep, L_pde.item(), L_bc.item(), L_ic.item(), L_ic_t.item(), L.item(), l2, linf])
            if l2 < best_l2:
                best_l2 = l2
                best_linf = linf
                best_state = capture_state()
                best_pred = pred_snapshot.copy()

            print(f"[{MODE.upper()}][Run {run_idx + 1}] {ep:6d} | "
                  f"PDE {L_pde.item():.2e} BC {L_bc.item():.2e} "
                  f"IC {L_ic.item():.2e} IC_t {L_ic_t.item():.2e} | "
                  f"L2 {l2:.2e} L∞ {linf:.2e}")

    elapsed = time.time() - start

    if best_state is not None:
        net.load_state_dict(best_state)
        u_pred = best_pred
        l2 = best_l2
        linf = best_linf
    else:
        with torch.no_grad():
            u_pred = net(X_star_norm).cpu().numpy()
        l2 = np.linalg.norm(u_pred - u_star) / np.linalg.norm(u_star)
        linf = np.max(np.abs(u_pred - u_star)) / np.max(np.abs(u_star))

    heat_exact_path, heat_pred_path, heat_error_path, slice_paths = plot_run_outputs(
        X, T, u_star.reshape(X.shape), x, t, X_star, u_pred, run_fig_dir
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

    ckpt_path = os.path.join(run_ckpt_dir, 'model.pt')
    torch.save(net.state_dict(), ckpt_path)

    print(f"[Run {run_idx + 1}] Best L2 {l2:.3e} | L∞ {linf:.3e}")

    return {
        'history': hist_array,
        'u_pred': u_pred,
        'u_star': u_star,
        'X_star': X_star,
        'X': X,
        'T': T,
        'x': x,
        't': t,
        'l2': l2,
        'linf': linf,
        'elapsed': elapsed,
        'seed': seed,
        'ckpt_path': ckpt_path,
        'fig_paths': (heat_exact_path, heat_pred_path, heat_error_path, slice_paths),
        'run_dir': run_dir,
        'run_fig_dir': run_fig_dir
    }


run_results = []
for run_idx in range(NUM_RUNS):
    seed = BASE_SEED + run_idx
    run_results.append(train_single_run(run_idx, seed))

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

    total_mean = hist_stack[:, :, 5].mean(axis=0)
    total_std = hist_stack[:, :, 5].std(axis=0)
    l2_mean = hist_stack[:, :, 6].mean(axis=0)
    l2_std = hist_stack[:, :, 6].std(axis=0)
    linf_mean = hist_stack[:, :, 7].mean(axis=0)
    linf_std = hist_stack[:, :, 7].std(axis=0)

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
        ax.plot(hist[:, 0], hist[:, 5], color='#a6bddb', alpha=0.3, linewidth=0.8)
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
# Final evaluation & plots (best run)
# ============================================================
l2_values = np.array([res['l2'] for res in run_results])
best_idx = int(np.argmin(l2_values))
best_run = run_results[best_idx]
best_path = os.path.join(AGG_CKPT_DIR, 'model_best.pt')
shutil.copyfile(best_run['ckpt_path'], best_path)

plot_run_outputs(
    best_run['X'],
    best_run['T'],
    best_run['u_star'].reshape(best_run['X'].shape),
    best_run['x'],
    best_run['t'],
    best_run['X_star'],
    best_run['u_pred'],
    AGG_FIG_DIR,
    'best'
)

# ============================================================
# Final summary
# ============================================================
linf_values = np.array([res['linf'] for res in run_results])
elapsed_total = sum(res['elapsed'] for res in run_results)

summary_path = os.path.join(AGG_LOG_DIR, 'summary.csv')
with open(summary_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['run', 'seed', 'l2', 'linf', 'run_time_sec', 'total_elapsed_time_sec'])
    for idx, res in enumerate(run_results, start=1):
        writer.writerow([idx, res['seed'], res['l2'], res['linf'], res['elapsed'], elapsed_total])
    writer.writerow([])
    writer.writerow(['best_run_index', best_idx + 1])
    writer.writerow(['best_run_seed', best_run['seed']])
    writer.writerow(['best_l2', best_run['l2']])
    writer.writerow(['best_linf', best_run['linf']])
    writer.writerow(['total_elapsed_time_sec', elapsed_total])
    writer.writerow(['l2_mean', l2_values.mean()])
    writer.writerow(['l2_std', l2_values.std()])
    writer.writerow(['linf_mean', linf_values.mean()])
    writer.writerow(['linf_std', linf_values.std()])

print("\n===== Final Results =====")
print(f"Mode: {MODE}")
print(f"Runs: {NUM_RUNS} | Seeds: {[BASE_SEED + i for i in range(NUM_RUNS)]}")
print(f"Total training time: {elapsed_total:.2f}s")
print(f"Best run: #{best_idx + 1} (seed={best_run['seed']}) "
      f"L2 {best_run['l2']:.3e} | L∞ {best_run['linf']:.3e}")
print(f"L2 mean ± std: {l2_values.mean():.3e} ± {l2_values.std():.3e}")
print(f"L∞ mean ± std: {linf_values.mean():.3e} ± {linf_values.std():.3e}")
