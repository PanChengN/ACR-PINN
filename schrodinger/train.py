#!/usr/bin/env python
"""Train ACR-PINN variants for the one-dimensional nonlinear Schrödinger equation."""

import csv
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
from torch.autograd import grad
from torch.optim import Adam

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.diagnostics import DiagnosticsRecorder, collect_task_gradients
from common.cost_benchmark import CudaCostProfiler
from common.models import build_model as build_shared_model, canonical_architecture
from common.optimization import canonical_optimizer, project_conflicting_gradients
from common.reproducibility import (
    file_sha256,
    optimizer_order_generator,
    seed_bundle,
    seed_model_initialization,
    stable_config_sha256,
    write_run_metadata,
)


MODE = canonical_optimizer(os.getenv("MODE", "sum"))
NET_ARCH = canonical_architecture(os.getenv("NET_ARCH", "mlp"))
LDA_STABILIZED = os.getenv("LDA_STABILIZED", "1").strip().lower() in ("1", "true", "yes")
LDA_RESIDUAL_SCALE = float(os.getenv("LDA_RESIDUAL_SCALE", "0.1"))
RUN_TAG = os.getenv("RESULT_TAG", f"{MODE}_{NET_ARCH}")
EPOCHS = int(os.getenv("EPOCHS", "100000"))
LR = float(os.getenv("LEARNING_RATE", "1e-3"))
LR_MIN = float(os.getenv("LR_MIN", "1e-4"))
WARMUP_EPOCHS = int(os.getenv("WARMUP_EPOCHS", "100"))
RECORD_EVERY = int(os.getenv("RECORD_EVERY", "1000"))
DIAGNOSTICS_EVERY = int(os.getenv("DIAGNOSTICS_EVERY", str(RECORD_EVERY)))
NUM_RUNS = int(os.getenv("NUM_RUNS", "1"))
RUN_START_INDEX = int(os.getenv("RUN_START_INDEX", "0"))
SAMPLE_SEED_BASE = int(os.getenv("SAMPLE_SEED_BASE", "1234"))
INIT_SEED_BASE = int(os.getenv("INIT_SEED_BASE", "2234"))
OPTIMIZER_SEED_BASE = int(os.getenv("OPTIMIZER_SEED_BASE", "3234"))
NUM_F = int(os.getenv("SCHRODINGER_NUM_F", "20000"))
NUM_B = int(os.getenv("SCHRODINGER_NUM_B", "500"))
NUM_IC = int(os.getenv("SCHRODINGER_NUM_IC", "500"))
RESAMPLE_EVERY = int(os.getenv("SCHRODINGER_RESAMPLE_EVERY", "1"))
FIXED_SAMPLING = os.getenv("SCHRODINGER_FIXED_SAMPLING", "0").strip().lower() in ("1", "true", "yes")
NO_PLOT = os.getenv("NO_PLOT", "0") == "1"
REFERENCE_PATH = Path(os.getenv(
    "SCHRODINGER_REFERENCE", str(REPO_ROOT / "data" / "nonlinear_schrodinger.mat")
))

X_MIN, X_MAX = -5.0, 5.0
T_MIN, T_MAX = 0.0, math.pi / 2.0
SAVE_DIR = Path(f"results_schrodinger_{RUN_TAG}")
RUNS_DIR = SAVE_DIR / "runs"
AGG_LOG_DIR = SAVE_DIR / "aggregate" / "logs"
AGG_CKPT_DIR = SAVE_DIR / "aggregate" / "checkpoints"
AGG_FIG_DIR = SAVE_DIR / "aggregate" / "figures"
GENERATED_DATA_DIR = Path(os.getenv(
    "SCHRODINGER_GENERATED_DATA_DIR",
    str(REPO_ROOT / "generated_data" / "schrodinger"),
))

device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print(f"Using device: {device}")


def normalize_xt(points):
    x = 2.0 * (points[:, 0:1] - X_MIN) / (X_MAX - X_MIN) - 1.0
    t = 2.0 * (points[:, 1:2] - T_MIN) / (T_MAX - T_MIN) - 1.0
    return torch.cat([x, t], dim=1)


def build_model():
    return build_shared_model(
        NET_ARCH, [2, 50, 50, 50, 50, 2],
        lda_stabilized=LDA_STABILIZED,
        lda_residual_scale=LDA_RESIDUAL_SCALE,
    )


def initial_solution(x):
    return torch.cat([2.0 / torch.cosh(x), torch.zeros_like(x)], dim=1)


def pde_residual(model, points):
    output = model(normalize_xt(points))
    u, v = output[:, 0:1], output[:, 1:2]
    du = grad(u, points, torch.ones_like(u), create_graph=True)[0]
    dv = grad(v, points, torch.ones_like(v), create_graph=True)[0]
    u_x, u_t = du[:, 0:1], du[:, 1:2]
    v_x, v_t = dv[:, 0:1], dv[:, 1:2]
    u_xx = grad(u_x, points, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    v_xx = grad(v_x, points, torch.ones_like(v_x), create_graph=True)[0][:, 0:1]
    amplitude_squared = u.square() + v.square()
    real_residual = -v_t + 0.5 * u_xx + amplitude_squared * u
    imaginary_residual = u_t + 0.5 * v_xx + amplitude_squared * v
    return real_residual, imaginary_residual


def _fast_lhs(rng, samples, dimensions):
    """Randomized LHS equivalent, efficient enough for per-iteration resampling."""
    values = np.empty((samples, dimensions), dtype=np.float32)
    for dimension in range(dimensions):
        permutation = rng.permutation(samples)
        values[:, dimension] = (permutation + rng.random(samples)) / samples
    return values


def sample_epoch(sample_seed, epoch):
    """Generate the deterministic paired LHS batch for one optimizer step."""
    effective_epoch = epoch // RESAMPLE_EVERY
    rng = np.random.default_rng(np.random.SeedSequence([sample_seed, effective_epoch]))
    interior_unit = _fast_lhs(rng, NUM_F, 2)
    boundary_unit = _fast_lhs(rng, NUM_B, 1)
    initial_unit = _fast_lhs(rng, NUM_IC, 1)
    interior = np.column_stack([
        X_MIN + (X_MAX - X_MIN) * interior_unit[:, 0],
        T_MIN + (T_MAX - T_MIN) * interior_unit[:, 1],
    ])
    boundary_t = T_MIN + (T_MAX - T_MIN) * boundary_unit
    initial_x = X_MIN + (X_MAX - X_MIN) * initial_unit
    return (
        torch.tensor(interior, dtype=torch.float32, device=device, requires_grad=True),
        torch.tensor(boundary_t, dtype=torch.float32, device=device),
        torch.tensor(initial_x, dtype=torch.float32, device=device),
    )


def sampling_descriptor(sample_seed):
    path = GENERATED_DATA_DIR / f"seed_{sample_seed}_sampling_descriptor.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "algorithm": "deterministic_randomized_latin_hypercube",
        "sample_seed": sample_seed,
        "sampling_mode": "fixed" if FIXED_SAMPLING else "dynamic",
        "resample_every": None if FIXED_SAMPLING else RESAMPLE_EVERY,
        "num_residual": NUM_F,
        "num_boundary": NUM_B,
        "num_initial": NUM_IC,
        "domain": [[X_MIN, X_MAX], [T_MIN, T_MAX]],
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != serialized:
        raise RuntimeError(f"Sampling descriptor collision: {path}")
    path.write_text(serialized, encoding="utf-8")
    return path, file_sha256(path)


def learning_rate_for_epoch(epoch):
    """100-step warm-up followed by 1e-3 to 1e-4 cosine decay."""
    if WARMUP_EPOCHS > 0 and epoch < WARMUP_EPOCHS:
        return LR_MIN + (LR - LR_MIN) * (epoch + 1) / WARMUP_EPOCHS
    cosine_steps = max(1, EPOCHS - WARMUP_EPOCHS - 1)
    progress = min(1.0, max(0.0, (epoch - WARMUP_EPOCHS) / cosine_steps))
    return LR_MIN + 0.5 * (LR - LR_MIN) * (1.0 + math.cos(math.pi * progress))


def load_reference():
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(f"NLS reference data not found: {REFERENCE_PATH}")
    data = loadmat(REFERENCE_PATH)
    x = np.asarray(data["x"]).reshape(-1).astype(np.float32)
    t = np.asarray(data["tt"]).reshape(-1).astype(np.float32)
    solution = np.asarray(data["uu"])
    if solution.shape == (t.size, x.size):
        solution = solution.T
    if solution.shape != (x.size, t.size):
        raise ValueError(f"Unexpected reference-solution shape {solution.shape}")
    x_grid, t_grid = np.meshgrid(x, t, indexing="ij")
    points = np.column_stack([x_grid.reshape(-1), t_grid.reshape(-1)]).astype(np.float32)
    values = np.column_stack([solution.real.reshape(-1), solution.imag.reshape(-1)]).astype(np.float32)
    return x, t, solution, points, values


def predict_batched(model, points, batch_size=65536):
    chunks = []
    with torch.no_grad():
        for start in range(0, len(points), batch_size):
            batch = torch.tensor(points[start:start + batch_size], dtype=torch.float32, device=device)
            chunks.append(model(normalize_xt(batch)).cpu().numpy())
    return np.vstack(chunks)


def metrics(prediction, reference):
    pred_complex = prediction[:, 0] + 1j * prediction[:, 1]
    ref_complex = reference[:, 0] + 1j * reference[:, 1]
    difference = pred_complex - ref_complex
    return {
        "l2": float(np.linalg.norm(difference) / np.linalg.norm(ref_complex)),
        "linf": float(np.max(np.abs(difference)) / np.max(np.abs(ref_complex))),
        "mse": float(np.mean(np.abs(difference) ** 2)),
    }


HISTORY_COLUMNS = ["iter", "pde", "boundary", "initial", "total", "l2", "linf", "mse"]


def train_single_run(run_index, seeds, reference):
    print(f"\n=== Run {run_index + 1}/{NUM_RUNS} | seeds={seeds} ===")
    seed_model_initialization(seeds["init_seed"])
    order_generator = optimizer_order_generator(seeds["optimizer_order_seed"])
    run_dir = RUNS_DIR / f"run_{run_index + 1}"
    log_dir = run_dir / "logs"
    checkpoint_dir = run_dir / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    descriptor_path, descriptor_sha256 = sampling_descriptor(seeds["sample_seed"])

    model = build_model().to(device)
    optimizer = Adam(model.parameters(), lr=LR_MIN)
    recorder = DiagnosticsRecorder(model, log_dir, device, DIAGNOSTICS_EVERY)
    cost_profiler = CudaCostProfiler(device, EPOCHS)
    config = {
        "problem": "schrodinger",
        "protocol_source": "configs/schrodinger.json",
        "mode": MODE,
        "network": NET_ARCH,
        "lda_stabilized": LDA_STABILIZED if NET_ARCH == "lda" else None,
        "lda_residual_scale": LDA_RESIDUAL_SCALE if NET_ARCH == "lda" else None,
        "layers": [2, 50, 50, 50, 50, 2],
        "activation": "tanh",
        "epochs": EPOCHS,
        "learning_rate_initial": LR,
        "learning_rate_final": LR_MIN,
        "warmup_epochs": WARMUP_EPOCHS,
        "schedule": "linear_warmup_then_cosine",
        "num_residual": NUM_F,
        "num_boundary": NUM_B,
        "num_initial": NUM_IC,
        "fixed_sampling": FIXED_SAMPLING,
        "resample_every": None if FIXED_SAMPLING else RESAMPLE_EVERY,
        "loss_tasks": ["pde", "periodic_boundary", "initial"],
        "supervised_interior_points": 0,
        "reference_path": str(REFERENCE_PATH.resolve()),
        "reference_sha256": file_sha256(REFERENCE_PATH),
        "dtype": "float32",
    }
    write_run_metadata(log_dir / "run_metadata.json", config, seeds, descriptor_path, descriptor_sha256)
    fixed_batch = sample_epoch(seeds["sample_seed"], 0) if FIXED_SAMPLING else None
    x_ref, t_ref, solution_grid, reference_points, reference_values = reference
    history = []
    start_time = time.time()

    for epoch in range(EPOCHS):
        X_f, t_b, x_i = fixed_batch if fixed_batch is not None else sample_epoch(seeds["sample_seed"], epoch)
        cost_profiler.start_step(epoch)
        X_left = torch.cat([torch.full_like(t_b, X_MIN), t_b], dim=1).requires_grad_(True)
        X_right = torch.cat([torch.full_like(t_b, X_MAX), t_b], dim=1).requires_grad_(True)
        X_initial = torch.cat([x_i, torch.zeros_like(x_i)], dim=1)

        residual_u, residual_v = pde_residual(model, X_f)
        pde_loss = torch.mean(residual_u.square() + residual_v.square())
        left, right = model(normalize_xt(X_left)), model(normalize_xt(X_right))
        value_loss = torch.mean((left - right).square())
        derivative_loss = 0.0
        for component in range(2):
            left_x = grad(left[:, component:component + 1], X_left, torch.ones_like(left[:, component:component + 1]), create_graph=True)[0][:, 0:1]
            right_x = grad(right[:, component:component + 1], X_right, torch.ones_like(right[:, component:component + 1]), create_graph=True)[0][:, 0:1]
            derivative_loss = derivative_loss + torch.mean((left_x - right_x).square())
        boundary_loss = value_loss + derivative_loss
        initial_loss = torch.mean((model(normalize_xt(X_initial)) - initial_solution(x_i)).square())
        task_losses = [pde_loss, boundary_loss, initial_loss]
        total_loss = sum(task_losses)
        task_names = ["pde", "boundary", "initial"]
        parameters = list(model.parameters())
        named_parameters = list(model.named_parameters())
        diagnostic_gradients = None
        projected_gradients = None
        if MODE == "sum" and recorder.should_record(epoch, EPOCHS):
            diagnostic_gradients = collect_task_gradients(task_losses, parameters)

        current_lr = learning_rate_for_epoch(epoch)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        recorder.start_step()
        if MODE == "sum":
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        elif MODE == "pcgrad":
            task_gradients = collect_task_gradients(task_losses, parameters)
            projected_gradients = project_conflicting_gradients(task_gradients, order_generator)
            diagnostic_gradients = task_gradients
            optimizer.zero_grad()
            for parameter, gradient_value in zip(parameters, projected_gradients):
                parameter.grad = gradient_value
            optimizer.step()
        else:
            raise ValueError(f"Unsupported optimizer strategy: {MODE}")
        cost_profiler.end_step(epoch)
        recorder.end_step()

        if recorder.should_record(epoch, EPOCHS):
            recorder.record(
                epoch, task_names, diagnostic_gradients, named_parameters, projected_gradients,
                normalize_xt(X_f[:min(1024, len(X_f))]).detach(),
                {"learning_rate": current_lr, "sampling_epoch": 0 if FIXED_SAMPLING else epoch // RESAMPLE_EVERY},
            )
        if epoch % RECORD_EVERY == 0 or epoch == EPOCHS - 1:
            result = metrics(predict_batched(model, reference_points), reference_values)
            history.append([
                epoch, float(pde_loss.detach()), float(boundary_loss.detach()),
                float(initial_loss.detach()), float(total_loss.detach()),
                result["l2"], result["linf"], result["mse"],
            ])
            print(
                f"[{MODE.upper()}][Run {run_index + 1}] {epoch:6d} | "
                f"PDE {pde_loss.item():.2e} BC {boundary_loss.item():.2e} "
                f"IC {initial_loss.item():.2e} | L2 {result['l2']:.2e} MSE {result['mse']:.2e}"
            )

    elapsed = time.time() - start_time
    recorder.finalize(elapsed)
    cost_profiler.write(log_dir, model, {"problem": "schrodinger", "architecture": NET_ARCH, "optimizer": MODE})
    final_prediction = predict_batched(model, reference_points)
    final_metrics = metrics(final_prediction, reference_values)
    history_array = np.asarray(history, dtype=float)
    np.savetxt(log_dir / "history.csv", history_array, delimiter=",", header=",".join(HISTORY_COLUMNS), comments="")
    checkpoint_path = checkpoint_dir / 'model_final.pt'
    torch.save(model.state_dict(), checkpoint_path)
    print(f"[Run {run_index + 1}] Final metrics: {final_metrics}")
    return {
        "run_index": run_index,
        "seeds": seeds,
        "dataset_sha256": descriptor_sha256,
        "metrics": final_metrics,
        "elapsed": elapsed,
        "checkpoint": checkpoint_path,
        "prediction": final_prediction,
        "reference": (x_ref, t_ref, solution_grid),
    }


def save_plots(result):
    if NO_PLOT:
        return
    AGG_FIG_DIR.mkdir(parents=True, exist_ok=True)
    x, t, solution = result["reference"]
    prediction = result["prediction"]
    magnitude_exact = np.abs(solution)
    magnitude_prediction = np.abs(prediction[:, 0] + 1j * prediction[:, 1]).reshape(solution.shape)
    for name, field in {
        "magnitude_exact": magnitude_exact,
        "magnitude_prediction": magnitude_prediction,
        "magnitude_absolute_error": np.abs(magnitude_prediction - magnitude_exact),
    }.items():
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        image = ax.pcolormesh(t, x, field, shading="auto")
        ax.set_xlabel("t")
        ax.set_ylabel("x")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(AGG_FIG_DIR / f"{name}.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main():
    if not 0 <= RUN_START_INDEX < NUM_RUNS:
        raise ValueError(f"RUN_START_INDEX must satisfy 0 <= start < {NUM_RUNS}")
    for path in (RUNS_DIR, AGG_LOG_DIR, AGG_CKPT_DIR, AGG_FIG_DIR):
        path.mkdir(parents=True, exist_ok=True)
    reference = load_reference()
    results = []
    for run_index in range(RUN_START_INDEX, NUM_RUNS):
        seeds = seed_bundle(run_index, SAMPLE_SEED_BASE, INIT_SEED_BASE, OPTIMIZER_SEED_BASE)
        results.append(train_single_run(run_index, seeds, reference))
    l2_values = np.asarray([result["metrics"]["l2"] for result in results])
    representative = results[int(np.argsort(l2_values)[len(results) // 2])]
    shutil.copyfile(representative["checkpoint"], AGG_CKPT_DIR / "model_representative.pt")
    save_plots(representative)
    with (AGG_LOG_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "sample_seed", "init_seed", "optimizer_order_seed", "sampling_descriptor_sha256", "final_l2", "final_linf", "final_mse", "run_time_sec"])
        for result in results:
            seeds, values = result["seeds"], result["metrics"]
            writer.writerow([result["run_index"] + 1, seeds["sample_seed"], seeds["init_seed"], seeds["optimizer_order_seed"], result["dataset_sha256"], values["l2"], values["linf"], values["mse"], result["elapsed"]])
        writer.writerow([])
        writer.writerow(["representative_run_index", representative["run_index"] + 1])
        writer.writerow(["representative_rule", "median_final_l2_for_visualization_only"])
    print("\n===== Final Results =====")
    print(f"Mode: {MODE} | Network: {NET_ARCH}")
    print(f"Metrics: {representative['metrics']}")


if __name__ == "__main__":
    main()
