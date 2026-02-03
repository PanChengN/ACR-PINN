#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.io

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.dpi'] = 300

# ============================================================
# Defaults (match burgers.py)
# ============================================================
SLICE_TIMES = [0.0, 0.5, 1.0]
EXACT_COLOR = '#2b2b2b'
PRED_COLOR = '#d55e00'
ERROR_COLOR = '#377eb8'


def normalize_xt(X):
    x = X[:, 0:1]            # x in [-1,1]
    t = X[:, 1:2]            # t in [0,1]
    t_hat = 2.0 * t - 1.0    # -> [-1,1]
    return torch.cat([x, t_hat], dim=1)


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


def infer_net_arch(ckpt_path):
    lowered = ckpt_path.lower()
    if 'attention' in lowered:
        return 'attention'
    if 'mlp' in lowered:
        return 'mlp'
    return None


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


def load_burgers_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    data = scipy.io.loadmat(os.path.join(data_dir, 'burgers_shock.mat'))
    x = data['x'].flatten()[:, None]
    t = data['t'].flatten()[:, None]
    exact = np.real(data['usol'])
    return x, t, exact


def plot_heatmaps(X, T, u_exact, u_pred, fig_dir, prefix=''):
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
    pcm = ax.pcolormesh(X, T, u_exact, shading='gouraud')
    ax.set_title('Exact')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    fig.colorbar(pcm, ax=ax)
    fig.tight_layout()
    plt.savefig(os.path.join(fig_dir, heatmap_name), dpi=300, bbox_inches='tight',
                pad_inches=0.02, transparent=True)
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 4))
    pcm = ax.pcolormesh(X, T, u_pred, shading='gouraud')
    ax.set_title('Prediction')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    fig.colorbar(pcm, ax=ax)
    fig.tight_layout()
    plt.savefig(os.path.join(fig_dir, pred_name), dpi=300, bbox_inches='tight',
                pad_inches=0.02, transparent=True)
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 4))
    pcm = ax.pcolormesh(X, T, np.abs(u_exact - u_pred), shading='gouraud')
    ax.set_title('Abs Error')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    fig.colorbar(pcm, ax=ax)
    fig.tight_layout()
    plt.savefig(os.path.join(fig_dir, error_name), dpi=300, bbox_inches='tight',
                pad_inches=0.02, transparent=True)
    plt.close()


def plot_slices(x, t, u_exact, u_pred, fig_dir, prefix=''):
    os.makedirs(fig_dir, exist_ok=True)
    x_values = x.flatten()
    t_values = t.flatten()
    for target in SLICE_TIMES:
        idx = int(np.abs(t_values - target).argmin())
        tag = f'{t_values[idx]:.2f}'
        slice_name = f'time_slice_t{tag}_{prefix}.pdf' if prefix else f'time_slice_t{tag}.pdf'
        err_name = f'time_slice_error_t{tag}_{prefix}.pdf' if prefix else f'time_slice_error_t{tag}.pdf'

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(x_values, u_exact[idx, :], color=EXACT_COLOR, label='Exact')
        ax.plot(x_values, u_pred[idx, :], color=PRED_COLOR, linestyle='--', label='Prediction')
        ax.set_title(f't={t_values[idx]:.2f}')
        ax.set_xlabel('x')
        ax.set_ylabel('u(x, t)')
        ax.legend()
        fig.tight_layout()
        plt.savefig(os.path.join(fig_dir, slice_name), dpi=300, bbox_inches='tight',
                    pad_inches=0.02, transparent=True)
        plt.close()

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(x_values, np.abs(u_exact[idx, :] - u_pred[idx, :]), color=ERROR_COLOR)
        ax.set_title(f'|Error| at t={t_values[idx]:.2f}')
        ax.set_xlabel('x')
        ax.set_ylabel(r'$|u_{\mathrm{exact}} - u_{\mathrm{pred}}|$')
        fig.tight_layout()
        plt.savefig(os.path.join(fig_dir, err_name), dpi=300, bbox_inches='tight',
                    pad_inches=0.02, transparent=True)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Replot Burgers results from saved checkpoints.')
    parser.add_argument('--ckpt', nargs='*', default=None, help='Path(s) to model.pt')
    parser.add_argument('--net-arch', default=None, choices=['mlp', 'attention'],
                        help='Network architecture used for the checkpoint (auto-infer if omitted)')
    parser.add_argument('--out-dir', default=None,
                        help='Directory to save plots (defaults to a new folder in ckpt directory)')
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

    x, t, exact = load_burgers_data()
    X, T = np.meshgrid(x, t)
    X_star = np.hstack([X.flatten()[:, None], T.flatten()[:, None]])
    X_star_t = torch.tensor(X_star, dtype=torch.float32, device=device)
    X_star_norm = normalize_xt(X_star_t)

    multiple_ckpts = len(ckpts) > 1
    for ckpt_path in ckpts:
        net_arch = args.net_arch or infer_net_arch(ckpt_path)
        if net_arch not in ('mlp', 'attention'):
            raise ValueError(f"Could not infer net arch for: {ckpt_path}")
        out_dir = resolve_out_dir(ckpt_path, args.out_dir, multiple_ckpts)

        net = build_model(net_arch).to(device)
        state = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(state)
        net.eval()

        with torch.no_grad():
            u_pred = net(X_star_norm).cpu().numpy().reshape(X.shape)

        u_exact = exact.T
        plot_heatmaps(X, T, u_exact, u_pred, out_dir)
        plot_slices(x, t, u_exact, u_pred, out_dir)
        print(f'Plots saved to: {out_dir}')


if __name__ == '__main__':
    main()
