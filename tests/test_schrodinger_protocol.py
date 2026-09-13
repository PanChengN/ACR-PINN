import importlib.util
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "schrodinger" / "train.py"
CONFIG = REPO_ROOT / "configs" / "schrodinger.json"
MANIFEST = REPO_ROOT / "configs" / "schrodinger_manifest.json"


def load_module(monkeypatch):
    monkeypatch.setenv("NO_PLOT", "1")
    spec = importlib.util.spec_from_file_location("schrodinger_benchmark", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_article_configuration_is_frozen():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    problem = config["problems"]["schrodinger"]
    policy = config["training_policy"]
    assert problem["layers"] == [2, 50, 50, 50, 50, 2]
    assert problem["epochs"] == 100000
    assert problem["samples"] == {"residual": 20000, "boundary": 500, "initial": 500}
    assert problem["environment"]["SCHRODINGER_RESAMPLE_EVERY"] == 1
    assert policy == {
        "learning_rate": 0.001,
        "lr_schedule": "linear_warmup_then_cosine",
        "lr_min": 0.0001,
        "warmup_epochs": 100,
        "checkpoint_policy": "fixed_final_iteration",
    }


def test_manifest_contains_paired_four_method_matrix():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    jobs = {(job["architecture"], job["optimizer"]) for job in manifest["jobs"]}
    assert jobs == {("mlp", "sum"), ("mlp", "pcgrad"), ("lda", "sum"), ("lda", "pcgrad")}


def test_sampling_is_deterministic_and_resampled(monkeypatch):
    module = load_module(monkeypatch)
    first = module.sample_epoch(1259, 0)[0].detach().cpu().numpy()
    repeated = module.sample_epoch(1259, 0)[0].detach().cpu().numpy()
    next_epoch = module.sample_epoch(1259, 1)[0].detach().cpu().numpy()
    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, next_epoch)
    assert first.shape == (20000, 2)


def test_no_interior_supervision_or_test_checkpoint_selection():
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"supervised_interior_points": 0' in source
    assert "data_loss" not in source
    assert "model_best.pt" not in source
    assert "'model_final.pt'" in source
    for task in ("pde", "boundary", "initial"):
        assert f'"{task}"' in source


def test_warmup_and_cosine_endpoints(monkeypatch):
    module = load_module(monkeypatch)
    assert np.isclose(module.learning_rate_for_epoch(99), 1e-3)
    assert np.isclose(module.learning_rate_for_epoch(module.EPOCHS - 1), 1e-4)
