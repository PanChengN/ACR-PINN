import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "poisson_5d" / "train.py"
CONFIG = REPO_ROOT / "configs" / "poisson_5d.json"
MANIFEST = REPO_ROOT / "configs" / "poisson_5d_manifest.json"


def load_module():
    spec = importlib.util.spec_from_file_location("poisson_5d_benchmark", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_article_configuration_is_frozen():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    problem = config["problems"]["poisson_5d"]
    assert problem["layers"] == [5, 50, 50, 50, 50, 1]
    assert problem["epochs"] == 100000
    assert problem["samples"] == {"residual": 20000, "boundary": 5000}
    assert problem["environment"]["POISSON_5D_RESAMPLE_EVERY"] == 1
    assert config["training_policy"]["warmup_epochs"] == 100
    assert config["training_policy"]["lr_min"] == 1e-4


def test_manifest_contains_paired_four_method_matrix():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    jobs = {(job["architecture"], job["optimizer"]) for job in manifest["jobs"]}
    assert jobs == {("mlp", "sum"), ("mlp", "pcgrad"), ("lda", "sum"), ("lda", "pcgrad")}


def test_manufactured_solution_satisfies_pde():
    module = load_module()
    points = torch.rand(16, 5, requires_grad=True)
    solution = module.exact_solution(points)
    first = torch.autograd.grad(solution, points, torch.ones_like(solution), create_graph=True)[0]
    laplacian = torch.zeros_like(solution)
    for dimension in range(5):
        second = torch.autograd.grad(first[:, dimension:dimension + 1], points, torch.ones_like(first[:, dimension:dimension + 1]), create_graph=True)[0]
        laplacian += second[:, dimension:dimension + 1]
    assert torch.max(torch.abs(-laplacian - module.forcing_term(points))).item() < 1e-5


def test_sampling_is_deterministic_resampled_and_on_boundary(monkeypatch):
    monkeypatch.setenv("POISSON_5D_NUM_F", "40")
    monkeypatch.setenv("POISSON_5D_NUM_B", "20")
    module = load_module()
    first_f, first_b = module.sample_epoch(1269, 0)
    repeated_f, repeated_b = module.sample_epoch(1269, 0)
    next_f, _ = module.sample_epoch(1269, 1)
    assert torch.equal(first_f, repeated_f)
    assert torch.equal(first_b, repeated_b)
    assert not torch.equal(first_f, next_f)
    on_face = ((first_b == 0) | (first_b == 1)).sum(dim=1)
    assert torch.all(on_face >= 1)
    face_counts = []
    for dimension in range(5):
        face_counts.extend([(first_b[:, dimension] == side).sum().item() for side in (0, 1)])
    assert face_counts == [2] * 10


def test_no_interior_supervision_or_test_selection():
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"supervised_interior_points": 0' in source
    assert "data_loss" not in source
    assert "model_best.pt" not in source
    assert "'model_final.pt'" in source
    assert '"loss_tasks": ["pde", "boundary"]' in source


def test_learning_rate_endpoints():
    module = load_module()
    assert np.isclose(module.learning_rate_for_epoch(99), 1e-3)
    assert np.isclose(module.learning_rate_for_epoch(module.EPOCHS - 1), 1e-4)
