import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / 'burgers' / 'train.py'
CONFIG = REPO_ROOT / 'configs' / 'burgers.json'
MANIFEST = REPO_ROOT / 'configs' / 'burgers_manifest.json'


def load_module():
    spec = importlib.util.spec_from_file_location('burgers_protocol', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_configuration_is_frozen():
    config = json.loads(CONFIG.read_text(encoding='utf-8'))
    problem = config['problems']['burgers']
    assert problem['layers'] == [2, 50, 50, 50, 50, 1]
    assert problem['epochs'] == 40000
    assert problem['samples'] == {
        'residual': 20000, 'boundary_total': 500, 'initial': 500,
    }
    assert problem['environment']['BURGERS_RESAMPLE_EVERY'] == 1
    assert config['seed_policy']['paired_runs'] == 5
    assert config['seed_policy']['master_seed_base'] == 19018
    assert config['training_policy']['lr_min'] == 1e-4


def test_manifest_contains_four_method_matrix():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    jobs = {(job['architecture'], job['optimizer']) for job in manifest['jobs']}
    assert jobs == {('mlp', 'sum'), ('mlp', 'pcgrad'), ('lda', 'sum'), ('lda', 'pcgrad')}


def test_dynamic_sampling_is_paired_deterministic_and_changes(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('BURGERS_NUM_F', '40')
    monkeypatch.setenv('BURGERS_NUM_B', '20')
    monkeypatch.setenv('BURGERS_NUM_IC', '10')
    monkeypatch.setenv('BURGERS_DYNAMIC_SAMPLING', '1')
    monkeypatch.setenv('BURGERS_BOUNDARY_COUNT_IS_TOTAL', '1')
    module = load_module()
    first = module.dynamic_sample_epoch(1291, 0)
    repeated = module.dynamic_sample_epoch(1291, 0)
    following = module.dynamic_sample_epoch(1291, 1)
    assert all(torch.equal(left, right) for left, right in zip(first, repeated))
    assert not torch.equal(first[0], following[0])
    assert first[0].shape == (40, 2)
    assert first[1].shape == (10, 2)
    assert first[2].shape == (10, 2)
    assert first[3].shape == (10, 2)
    assert torch.allclose(first[4], -torch.sin(torch.pi * first[3][:, 0:1]))


def test_warmup_cosine_endpoints(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('EPOCHS', '30000')
    monkeypatch.setenv('LEARNING_RATE', '1e-3')
    monkeypatch.setenv('LR_MIN', '1e-4')
    monkeypatch.setenv('WARMUP_EPOCHS', '100')
    module = load_module()
    assert np.isclose(module.warmup_cosine_learning_rate(99), 1e-3)
    assert np.isclose(module.warmup_cosine_learning_rate(module.EPOCHS - 1), 1e-4)


def test_fixed_final_checkpoint_policy_is_preserved():
    source = SCRIPT.read_text(encoding='utf-8')
    assert 'model_best.pt' not in source
    assert "'model_final.pt'" in source
