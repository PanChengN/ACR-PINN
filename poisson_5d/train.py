#!/usr/bin/env python
"""Train ACR-PINN variants for the five-dimensional Poisson equation."""

import csv
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
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
    load_or_create_npz,
    optimizer_order_generator,
    seed_bundle,
    seed_model_initialization,
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
DIMENSION = int(os.getenv("POISSON_5D_DIMENSION", "5"))
NUM_F = int(os.getenv("POISSON_5D_NUM_F", "20000"))
NUM_B = int(os.getenv("POISSON_5D_NUM_B", "5000"))
NUM_TEST = int(os.getenv("POISSON_5D_NUM_TEST", "100000"))
RESAMPLE_EVERY = int(os.getenv("POISSON_5D_RESAMPLE_EVERY", "1"))
FIXED_SAMPLING = os.getenv("POISSON_5D_FIXED_SAMPLING", "0").strip().lower() in ("1", "true", "yes")

SAVE_DIR = Path(f"results_poisson_5d_{RUN_TAG}")
RUNS_DIR = SAVE_DIR / "runs"
AGG_LOG_DIR = SAVE_DIR / "aggregate" / "logs"
AGG_CKPT_DIR = SAVE_DIR / "aggregate" / "checkpoints"
GENERATED_DATA_DIR = Path(os.getenv(
    "POISSON_5D_GENERATED_DATA_DIR",
    str(REPO_ROOT / "generated_data" / "poisson_5d"),
))

device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print(f"Using device: {device}")


def build_model():
    return build_shared_model(
        NET_ARCH, [DIMENSION, 50, 50, 50, 50, 1],
        lda_stabilized=LDA_STABILIZED,
        lda_residual_scale=LDA_RESIDUAL_SCALE,
    )


def exact_solution(points):
    return torch.sin(0.5 * torch.pi * points).sum(dim=1, keepdim=True)


def forcing_term(points):
    return 0.25 * torch.pi**2 * torch.sin(0.5 * torch.pi * points).sum(dim=1, keepdim=True)


def pde_residual(model, points):
    prediction = model(points)
    first = grad(prediction, points, torch.ones_like(prediction), create_graph=True)[0]
    laplacian = torch.zeros_like(prediction)
    for dimension in range(DIMENSION):
        derivative = first[:, dimension:dimension + 1]
        second = grad(derivative, points, torch.ones_like(derivative), create_graph=True)[0]
        laplacian = laplacian + second[:, dimension:dimension + 1]
    return -laplacian - forcing_term(points)


def _fast_lhs(rng, samples, dimensions):
    values = np.empty((samples, dimensions), dtype=np.float32)
    for dimension in range(dimensions):
        values[:, dimension] = (rng.permutation(samples) + rng.random(samples)) / samples
    return values


def sample_epoch(sample_seed, epoch):
    effective_epoch = epoch // RESAMPLE_EVERY
    rng = np.random.default_rng(np.random.SeedSequence([sample_seed, effective_epoch]))
    interior = _fast_lhs(rng, NUM_F, DIMENSION)
    boundary = _fast_lhs(rng, NUM_B, DIMENSION)
    faces = rng.permutation(np.arange(NUM_B) % (2 * DIMENSION))
    boundary[np.arange(NUM_B), faces // 2] = faces % 2
    return (
        torch.tensor(interior, dtype=torch.float32, device=device, requires_grad=True),
        torch.tensor(boundary, dtype=torch.float32, device=device),
    )


def sampling_descriptor(sample_seed):
    path = GENERATED_DATA_DIR / f"seed_{sample_seed}_sampling_descriptor.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "algorithm": "deterministic_randomized_latin_hypercube",
        "boundary_algorithm": "balanced_uniform_hypercube_faces",
        "sample_seed": sample_seed,
        "dimension": DIMENSION,
        "num_residual": NUM_F,
        "num_boundary": NUM_B,
        "sampling_mode": "fixed" if FIXED_SAMPLING else "dynamic",
        "resample_every": None if FIXED_SAMPLING else RESAMPLE_EVERY,
        "domain": [[0.0, 1.0]] * DIMENSION,
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != serialized:
        raise RuntimeError(f"Sampling descriptor collision: {path}")
    path.write_text(serialized, encoding="utf-8")
    return path, file_sha256(path)


def evaluation_set():
    path = GENERATED_DATA_DIR / f"evaluation_seed_9259_n{NUM_TEST}.npz"

    def create():
        rng = np.random.default_rng(9259)
        return {"points": _fast_lhs(rng, NUM_TEST, DIMENSION)}

    arrays, sha256 = load_or_create_npz(path, create)
    return arrays["points"], path, sha256


def learning_rate_for_epoch(epoch):
    if WARMUP_EPOCHS > 0 and epoch < WARMUP_EPOCHS:
        return LR_MIN + (LR - LR_MIN) * (epoch + 1) / WARMUP_EPOCHS
    cosine_steps = max(1, EPOCHS - WARMUP_EPOCHS - 1)
    progress = min(1.0, max(0.0, (epoch - WARMUP_EPOCHS) / cosine_steps))
    return LR_MIN + 0.5 * (LR - LR_MIN) * (1.0 + math.cos(math.pi * progress))


def predict_batched(model, points, batch_size=65536):
    chunks = []
    with torch.no_grad():
        for start in range(0, len(points), batch_size):
            batch = torch.tensor(points[start:start + batch_size], dtype=torch.float32, device=device)
            chunks.append(model(batch).cpu().numpy())
    return np.vstack(chunks)


def metrics(model, test_points):
    prediction = predict_batched(model, test_points)
    reference = exact_solution(torch.tensor(test_points, dtype=torch.float32)).numpy()
    difference = prediction - reference
    return {
        "l2": float(np.linalg.norm(difference) / np.linalg.norm(reference)),
        "linf": float(np.max(np.abs(difference)) / np.max(np.abs(reference))),
        "mse": float(np.mean(difference**2)),
    }


HISTORY_COLUMNS = ["iter", "pde", "boundary", "total", "l2", "linf", "mse"]


def train_single_run(run_index, seeds, test_points, evaluation_sha256):
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
        "problem": "poisson_5d",
        "protocol_source": "configs/poisson_5d.json",
        "mode": MODE,
        "network": NET_ARCH,
        "lda_stabilized": LDA_STABILIZED if NET_ARCH == "lda" else None,
        "lda_residual_scale": LDA_RESIDUAL_SCALE if NET_ARCH == "lda" else None,
        "layers": [DIMENSION, 50, 50, 50, 50, 1],
        "activation": "tanh",
        "dimension": DIMENSION,
        "epochs": EPOCHS,
        "learning_rate_initial": LR,
        "learning_rate_final": LR_MIN,
        "warmup_epochs": WARMUP_EPOCHS,
        "schedule": "linear_warmup_then_cosine",
        "num_residual": NUM_F,
        "num_boundary": NUM_B,
        "num_test": NUM_TEST,
        "fixed_sampling": FIXED_SAMPLING,
        "resample_every": None if FIXED_SAMPLING else RESAMPLE_EVERY,
        "loss_tasks": ["pde", "boundary"],
        "supervised_interior_points": 0,
        "evaluation_sha256": evaluation_sha256,
        "dtype": "float32",
    }
    write_run_metadata(log_dir / "run_metadata.json", config, seeds, descriptor_path, descriptor_sha256)
    fixed_batch = sample_epoch(seeds["sample_seed"], 0) if FIXED_SAMPLING else None
    history = []
    start_time = time.time()

    for epoch in range(EPOCHS):
        interior, boundary = fixed_batch if fixed_batch is not None else sample_epoch(seeds["sample_seed"], epoch)
        cost_profiler.start_step(epoch)
        pde_loss = torch.mean(pde_residual(model, interior).square())
        boundary_loss = torch.mean((model(boundary) - exact_solution(boundary)).square())
        task_losses = [pde_loss, boundary_loss]
        total_loss = sum(task_losses)
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
                epoch, ["pde", "boundary"], diagnostic_gradients, named_parameters,
                projected_gradients, interior[:min(1024, len(interior))].detach(),
                {"learning_rate": current_lr, "sampling_epoch": 0 if FIXED_SAMPLING else epoch // RESAMPLE_EVERY},
            )
        if epoch % RECORD_EVERY == 0 or epoch == EPOCHS - 1:
            result = metrics(model, test_points)
            history.append([
                epoch, float(pde_loss.detach()), float(boundary_loss.detach()),
                float(total_loss.detach()), result["l2"], result["linf"], result["mse"],
            ])
            print(
                f"[{MODE.upper()}][Run {run_index + 1}] {epoch:6d} | "
                f"PDE {pde_loss.item():.2e} BC {boundary_loss.item():.2e} | "
                f"L2 {result['l2']:.2e} MSE {result['mse']:.2e}"
            )

    elapsed = time.time() - start_time
    recorder.finalize(elapsed)
    cost_profiler.write(log_dir, model, {"problem": "poisson_5d", "architecture": NET_ARCH, "optimizer": MODE})
    final_metrics = metrics(model, test_points)
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
    }


def main():
    if not 0 <= RUN_START_INDEX < NUM_RUNS:
        raise ValueError(f"RUN_START_INDEX must satisfy 0 <= start < {NUM_RUNS}")
    for path in (RUNS_DIR, AGG_LOG_DIR, AGG_CKPT_DIR):
        path.mkdir(parents=True, exist_ok=True)
    test_points, _, evaluation_sha256 = evaluation_set()
    results = []
    for run_index in range(RUN_START_INDEX, NUM_RUNS):
        seeds = seed_bundle(run_index, SAMPLE_SEED_BASE, INIT_SEED_BASE, OPTIMIZER_SEED_BASE)
        results.append(train_single_run(run_index, seeds, test_points, evaluation_sha256))
    l2_values = np.asarray([result["metrics"]["l2"] for result in results])
    representative = results[int(np.argsort(l2_values)[len(results) // 2])]
    shutil.copyfile(representative["checkpoint"], AGG_CKPT_DIR / "model_representative.pt")
    with (AGG_LOG_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "sample_seed", "init_seed", "optimizer_order_seed", "sampling_descriptor_sha256", "evaluation_sha256", "final_l2", "final_linf", "final_mse", "run_time_sec"])
        for result in results:
            seeds, values = result["seeds"], result["metrics"]
            writer.writerow([result["run_index"] + 1, seeds["sample_seed"], seeds["init_seed"], seeds["optimizer_order_seed"], result["dataset_sha256"], evaluation_sha256, values["l2"], values["linf"], values["mse"], result["elapsed"]])
    print("\n===== Final Results =====")
    print(f"Mode: {MODE} | Network: {NET_ARCH} | Metrics: {representative['metrics']}")


if __name__ == "__main__":
    main()
