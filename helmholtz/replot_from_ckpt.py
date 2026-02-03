#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.dpi'] = 300

# ============================================================
# Defaults (match helmholtz.py)
# ============================================================
DEFAULT_A1 = 4.0
DEFAULT_A2 = 4.0
K = 1.0

SLICE_XS = [1.0, 0.75]
EXACT_COLOR = '#2b2b2b'
PRED_COLOR = '#d55e00'
ERROR_COLOR = '#377eb8'


class MLP(nn.Module):
    def __init__(self, layers):
        super().__init__()
        net = []
        for i in range(len(layers) - 2):
            lin = nn.Linear(layers[i], layers[i + 1])
            nn.init.xavier_normal_(lin.weight, gain=5 / 3)
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


def build_model(net_arch):
    layers = [2, 50, 50, 50, 50, 1]
    if net_arch == 'mlp':
        return MLP(layers)
    if net_arch == 'attention':
        return NetAttentionDynamic(layers)
    raise ValueError(f"Unknown NET_ARCH='{net_arch}'")


def exact_solution(X, a1, a2):
    return (
        torch.sin(a1 * np.pi * X[:, 0:1]) *
        torch.sin(a2 * np.pi * X[:, 1:2])
    )


def infer_net_arch(ckpt_path):
    lowered = ckpt_path.lower()
    if 'attention' in lowered:
        return 'attention'
    if 'mlp' in lowered:
        return 'mlp'
    return None


def infer_a_params(ckpt_path):
    lowered = ckpt_path.lower()
    a1 = None
    a2 = None
    for token in lowered.replace('-', '_').split('_'):
        if token.startswith('a1'):
            try:
                a1 = float(token.replace('a1', ''))
            except ValueError:
                pass
        elif token.startswith('a2'):
            try:
                a2 = float(token.replace('a2', ''))
            except ValueError:
                pass
    if a1 is None or a2 is None:
        # Try pattern like a1_4.0_a2_4.0
        parts = lowered.replace('-', '_').split('_')
        for i, part in enumerate(parts[:-1]):
            if part == 'a1':
                try:
                    a1 = float(parts[i + 1])
                except ValueError:
                    pass
            if part == 'a2':
                try:
                    a2 = float(parts[i + 1])
                except ValueError:
                    pass
    if a1 is None or a2 is None:
        print(f'Warning: could not parse a1/a2 from {ckpt_path}, using defaults.')
    return (a1 if a1 is not None else DEFAULT_A1,
            a2 if a2 is not None else DEFAULT_A2)


def find_checkpoints(root_dir):
    ckpts = []
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if name.endswith('.pt'):
                ckpts.append(os.path.join(dirpath, name))
    return sorted(ckpts)


def resolve_out_dir(ckpt_path, base_out_dir, multiple_ckpts):
    if base_out_dir:
        if multiple_ckpts:
            stem = os.path.splitext(os.path.basename(ckpt_path))[0]
            return os.path.join(base_out_dir, stem)
        return base_out_dir
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path)) or '.'
    stem = os.path.splitext(os.path.basename(ckpt_path))[0]
    return os.path.join(ckpt_dir, f'replot_outputs_{stem}')


def plot_slices(X, Y, U_exact, U_pred, fig_dir, prefix='',
                y_limits_by_target=None, err_limits_by_target=None):
    os.makedirs(fig_dir, exist_ok=True)
    x_values = X[0, :]
    y_values = Y[:, 0]
    y_limits_by_target = y_limits_by_target or {}
    err_limits_by_target = err_limits_by_target or {}
    for target in SLICE_XS:
        idx = int(np.abs(x_values - target).argmin())
        if target in y_limits_by_target:
            min_u, max_u = y_limits_by_target[target]
        else:
            exact_slice = U_exact[:, idx:idx + 1]
            pred_slice = U_pred[:, idx:idx + 1]
            min_u = float(np.min([exact_slice.min(), pred_slice.min()]))
            max_u = float(np.max([exact_slice.max(), pred_slice.max()]))
        if target in err_limits_by_target:
            min_err, max_err = err_limits_by_target[target]
        else:
            err_slice = np.abs(U_exact[:, idx:idx + 1] - U_pred[:, idx:idx + 1])
            min_err = float(err_slice.min())
            max_err = float(err_slice.max())
        tag = f'{x_values[idx]:.2f}'
        name = f'slice_x{tag}_{prefix}.pdf' if prefix else f'slice_x{tag}.pdf'
        err_name = f'slice_error_x{tag}_{prefix}.pdf' if prefix else f'slice_error_x{tag}.pdf'

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(y_values, U_exact[:, idx], color=EXACT_COLOR, label='Exact')
        ax.plot(y_values, U_pred[:, idx], color=PRED_COLOR, linestyle='--', label='Prediction')
        ax.set_title(f'x={x_values[idx]:.2f}')
        ax.set_xlabel('y')
        ax.set_ylabel('u(x, y)')
        ax.set_ylim(min_u, max_u)
        ax.legend()
        fig.tight_layout()
        plt.savefig(os.path.join(fig_dir, name), dpi=300, bbox_inches='tight',
                    pad_inches=0.02, transparent=True)
        plt.close()

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(y_values, np.abs(U_exact[:, idx] - U_pred[:, idx]), color=ERROR_COLOR)
        ax.set_title(f'|Error| at x={x_values[idx]:.2f}')
        ax.set_xlabel('y')
        ax.set_ylabel(r'$|u_{\mathrm{exact}} - u_{\mathrm{pred}}|$')
        ax.set_ylim(min_err, max_err)
        fig.tight_layout()
        plt.savefig(os.path.join(fig_dir, err_name), dpi=300, bbox_inches='tight',
                    pad_inches=0.02, transparent=True)
        plt.close()


def compute_slice_limits(X, results, targets):
    x_values = X[0, :]
    limits = {}
    err_limits = {}
    for target in targets:
        idx = int(np.abs(x_values - target).argmin())
        min_u = np.inf
        max_u = -np.inf
        min_err = np.inf
        max_err = -np.inf
        for _, _, U_exact, U_pred in results:
            exact_slice = U_exact[:, idx:idx + 1]
            pred_slice = U_pred[:, idx:idx + 1]
            min_u = float(min(min_u, exact_slice.min(), pred_slice.min()))
            max_u = float(max(max_u, exact_slice.max(), pred_slice.max()))
            err_slice = np.abs(U_exact[:, idx:idx + 1] - U_pred[:, idx:idx + 1])
            min_err = float(min(min_err, err_slice.min()))
            max_err = float(max(max_err, err_slice.max()))
        limits[target] = (min_u, max_u)
        err_limits[target] = (min_err, max_err)
    return limits, err_limits


def plot_heatmaps(X, Y, U_exact, U_pred, fig_dir, prefix=''):
    os.makedirs(fig_dir, exist_ok=True)
    if prefix:
        heatmap_name = f'heatmap_exact_{prefix}.pdf'
        pred_name = f'heatmap_pred_{prefix}.pdf'
        error_name = f'heatmap_error_{prefix}.pdf'
    else:
        heatmap_name = 'heatmap_exact.pdf'
        pred_name = 'heatmap_pred.pdf'
        error_name = 'heatmap_error.pdf'

    fig, ax = plt.subplots(figsize=(5, 4))
    pcm = ax.pcolormesh(X, Y, U_exact, shading='gouraud')
    ax.set_title('Exact')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(pcm, ax=ax)
    fig.tight_layout()
    plt.savefig(os.path.join(fig_dir, heatmap_name), dpi=300, bbox_inches='tight',
                pad_inches=0.02, transparent=True)
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 4))
    pcm = ax.pcolormesh(X, Y, U_pred, shading='gouraud')
    ax.set_title('Prediction')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(pcm, ax=ax)
    fig.tight_layout()
    plt.savefig(os.path.join(fig_dir, pred_name), dpi=300, bbox_inches='tight',
                pad_inches=0.02, transparent=True)
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 4))
    abs_err = np.abs(U_exact - U_pred)
    pcm = ax.pcolormesh(X, Y, abs_err, shading='gouraud')
    ax.set_title('Abs Error')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(pcm, ax=ax)
    fig.tight_layout()
    plt.savefig(os.path.join(fig_dir, error_name), dpi=300, bbox_inches='tight',
                pad_inches=0.02, transparent=True)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Replot Helmholtz slice results from checkpoints.')
    parser.add_argument('--ckpt', nargs='*', default=None, help='Path(s) to model.pt')
    parser.add_argument('--net-arch', default=None, choices=['mlp', 'attention'],
                        help='Network architecture used for the checkpoint (auto-infer if omitted)')
    parser.add_argument('--out-dir', default=None,
                        help='Directory to save plots (defaults to a new folder in ckpt directory)')
    parser.add_argument('--n-test', type=int, default=200, help='Grid size for plotting')
    args = parser.parse_args()

    ckpts = args.ckpt or find_checkpoints(os.getcwd())
    if not ckpts:
        raise FileNotFoundError('No .pt checkpoints found in current directory.')

    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f'Using device: {device}')

    x = np.linspace(-1, 1, args.n_test)
    y = np.linspace(-1, 1, args.n_test)
    X, Y = np.meshgrid(x, y)
    XY_flat = np.hstack([X.flatten()[:, None], Y.flatten()[:, None]])
    XY_t = torch.tensor(XY_flat, dtype=torch.float32, device=device)
    multiple_ckpts = len(ckpts) > 1
    results = []
    for ckpt_path in ckpts:
        a1, a2 = infer_a_params(ckpt_path)
        net_arch = args.net_arch or infer_net_arch(ckpt_path)
        if net_arch not in ('mlp', 'attention'):
            raise ValueError(f"Could not infer net arch for: {ckpt_path}")
        out_dir = resolve_out_dir(ckpt_path, args.out_dir, multiple_ckpts)

        net = build_model(net_arch).to(device)
        state = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(state)
        net.eval()

        U_exact = exact_solution(XY_t, a1, a2).detach().cpu().numpy().reshape(X.shape)
        with torch.no_grad():
            U_pred = net(XY_t).cpu().numpy().reshape(X.shape)

        results.append((ckpt_path, out_dir, U_exact, U_pred))

    slice_limits, err_limits = compute_slice_limits(X, results, targets=[0.75])
    slice_limits[0.75] = (-0.2, 0.2)
    err_limits.pop(0.75, None)
    for ckpt_path, out_dir, U_exact, U_pred in results:
        plot_heatmaps(X, Y, U_exact, U_pred, out_dir)
        plot_slices(X, Y, U_exact, U_pred, out_dir,
                    y_limits_by_target=slice_limits,
                    err_limits_by_target=err_limits)
        print(f'Plots saved to: {out_dir}')


if __name__ == '__main__':
    main()
