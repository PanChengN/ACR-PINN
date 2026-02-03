#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.dpi'] = 300

# ============================================================
# Defaults (match Lid_driven_Cavity.py)
# ============================================================
SLICE_YS = [0.8, 0.75, 0.5]
EXACT_COLOR = '#2b2b2b'
PRED_COLOR = '#d55e00'
ERROR_COLOR = '#377eb8'


def normalize_xy(X):
    x = X[:, 0:1]
    y = X[:, 1:2]
    x_hat = 2.0 * x - 1.0
    y_hat = 2.0 * y - 1.0
    return torch.cat([x_hat, y_hat], dim=1)


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


def build_model(net_arch, layers=None):
    layers = layers or [2, 50, 50, 50, 2]
    if net_arch == 'mlp':
        return MLP(layers)
    if net_arch == 'attention':
        return NetAttentionDynamic(layers)
    raise ValueError(f"Unknown NET_ARCH='{net_arch}'")


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


def predict_velocity(net, coords, batch_size=4096):
    preds = []
    for chunk in torch.split(coords, batch_size):
        chunk = chunk.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            out = net(normalize_xy(chunk))
            u, v = psi_to_velocity(out, chunk, create_graph=False)
        preds.append(torch.cat([u, v], dim=1).detach().cpu())
    return torch.cat(preds, dim=0).numpy()


def infer_net_arch(ckpt_path):
    lowered = ckpt_path.lower()
    if 'attention' in lowered:
        return 'attention'
    if 'mlp' in lowered:
        return 'mlp'
    return None


def infer_net_arch_from_state(state):
    if any(k.startswith('gates.') or k.startswith('encoder1.') or k.startswith('encoder2.')
           for k in state.keys()):
        return 'attention'
    if any(k.startswith('net.') for k in state.keys()):
        return 'mlp'
    return None


def infer_layers_from_state(state, net_arch):
    if net_arch == 'mlp':
        weight_keys = [k for k in state.keys() if k.startswith('net.') and k.endswith('.weight')]
        indices = []
        for key in weight_keys:
            try:
                indices.append(int(key.split('.')[1]))
            except (IndexError, ValueError):
                continue
        indices = sorted(indices)
        if not indices:
            return None
        layers = [state[f'net.{indices[0]}.weight'].shape[1]]
        for idx in indices:
            layers.append(state[f'net.{idx}.weight'].shape[0])
        return layers
    if net_arch == 'attention':
        weight_keys = [k for k in state.keys() if k.startswith('linear.') and k.endswith('.weight')]
        indices = []
        for key in weight_keys:
            try:
                indices.append(int(key.split('.')[1]))
            except (IndexError, ValueError):
                continue
        indices = sorted(indices)
        if not indices:
            return None
        layers = [state[f'linear.{indices[0]}.weight'].shape[1]]
        for idx in indices:
            layers.append(state[f'linear.{idx}.weight'].shape[0])
        return layers
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


def load_reference_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'Lid-driven-Cavity')
    u_ref = np.genfromtxt(os.path.join(data_dir, 'reference_u.csv'), delimiter=',')
    v_ref = np.genfromtxt(os.path.join(data_dir, 'reference_v.csv'), delimiter=',')
    nx, ny = u_ref.shape
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    return x, y, X, Y, u_ref, v_ref


def plot_heatmaps(X, Y, vel_exact, vel_pred, fig_dir, prefix=''):
    os.makedirs(fig_dir, exist_ok=True)
    if prefix:
        exact_name = f'velocity_exact_{prefix}.pdf'
        pred_name = f'velocity_pred_{prefix}.pdf'
        error_name = f'velocity_error_{prefix}.pdf'
    else:
        exact_name = 'velocity_exact.pdf'
        pred_name = 'velocity_pred.pdf'
        error_name = 'velocity_error.pdf'

    fig, ax = plt.subplots(figsize=(5, 4))
    pcm = ax.pcolormesh(X, Y, vel_exact, shading='gouraud')
    ax.set_title('Exact |u|')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(pcm, ax=ax)
    fig.tight_layout()
    plt.savefig(os.path.join(fig_dir, exact_name), dpi=300, bbox_inches='tight',
                pad_inches=0.02, transparent=True)
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 4))
    pcm = ax.pcolormesh(X, Y, vel_pred, shading='gouraud')
    ax.set_title('Prediction |u|')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(pcm, ax=ax)
    fig.tight_layout()
    plt.savefig(os.path.join(fig_dir, pred_name), dpi=300, bbox_inches='tight',
                pad_inches=0.02, transparent=True)
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 4))
    pcm = ax.pcolormesh(X, Y, np.abs(vel_exact - vel_pred), shading='gouraud')
    ax.set_title('Abs Error |u|')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(pcm, ax=ax)
    fig.tight_layout()
    plt.savefig(os.path.join(fig_dir, error_name), dpi=300, bbox_inches='tight',
                pad_inches=0.02, transparent=True)
    plt.close()


def plot_slices(X, Y, vel_exact, vel_pred, fig_dir, prefix=''):
    os.makedirs(fig_dir, exist_ok=True)
    y_values = Y[0, :]
    x_values = X[:, 0]
    for target in SLICE_YS:
        idx = int(np.abs(y_values - target).argmin())
        tag = f'{y_values[idx]:.2f}'
        slice_name = f'slice_y{tag}_{prefix}.pdf' if prefix else f'slice_y{tag}.pdf'
        err_name = f'slice_error_y{tag}_{prefix}.pdf' if prefix else f'slice_error_y{tag}.pdf'

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(x_values, vel_exact[:, idx], color=EXACT_COLOR, label='Exact')
        ax.plot(x_values, vel_pred[:, idx], color=PRED_COLOR, linestyle='--', label='Prediction')
        ax.set_title(f'y={y_values[idx]:.2f}')
        ax.set_xlabel('x')
        ax.set_ylabel('|u(x, y)|')
        ax.legend()
        fig.tight_layout()
        plt.savefig(os.path.join(fig_dir, slice_name), dpi=300, bbox_inches='tight',
                    pad_inches=0.02, transparent=True)
        plt.close()

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(x_values, np.abs(vel_exact[:, idx] - vel_pred[:, idx]), color=ERROR_COLOR)
        ax.set_title(f'|Error| at y={y_values[idx]:.2f}')
        ax.set_xlabel('x')
        ax.set_ylabel(r'$|u_{\mathrm{exact}} - u_{\mathrm{pred}}|$')
        fig.tight_layout()
        plt.savefig(os.path.join(fig_dir, err_name), dpi=300, bbox_inches='tight',
                    pad_inches=0.02, transparent=True)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Replot Lid-driven Cavity results from checkpoints.')
    parser.add_argument('--ckpt', nargs='*', default=None, help='Path(s) to model.pt')
    parser.add_argument('--net-arch', default=None, choices=['mlp', 'attention'],
                        help='Network architecture used for the checkpoint (auto-infer if omitted)')
    parser.add_argument('--out-dir', default=None,
                        help='Directory to save plots (defaults to a new folder in ckpt directory)')
    parser.add_argument('--batch-size', type=int, default=4096, help='Batch size for prediction')
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

    x, y, X, Y, u_ref, v_ref = load_reference_data()
    coords = np.hstack([X.flatten()[:, None], Y.flatten()[:, None]])
    coords_t = torch.tensor(coords, dtype=torch.float32, device=device)

    vel_exact = np.sqrt(u_ref ** 2 + v_ref ** 2)

    multiple_ckpts = len(ckpts) > 1
    for ckpt_path in ckpts:
        state = torch.load(ckpt_path, map_location=device)
        net_arch = args.net_arch or infer_net_arch(ckpt_path) or infer_net_arch_from_state(state)
        if net_arch not in ('mlp', 'attention'):
            raise ValueError(f"Could not infer net arch for: {ckpt_path}")
        out_dir = resolve_out_dir(ckpt_path, args.out_dir, multiple_ckpts)

        layers = infer_layers_from_state(state, net_arch)
        if layers is None:
            raise ValueError(f"Could not infer network layers for: {ckpt_path}")
        net = build_model(net_arch, layers=layers).to(device)
        net.load_state_dict(state)
        net.eval()

        uv_pred = predict_velocity(net, coords_t, batch_size=args.batch_size)
        u_pred = uv_pred[:, 0].reshape(X.shape)
        v_pred = uv_pred[:, 1].reshape(X.shape)
        vel_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)

        plot_heatmaps(X, Y, vel_exact, vel_pred, out_dir)
        plot_slices(X, Y, vel_exact, vel_pred, out_dir)
        print(f'Plots saved to: {out_dir}')


if __name__ == '__main__':
    main()
