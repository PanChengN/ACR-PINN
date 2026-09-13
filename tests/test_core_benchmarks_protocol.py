import json
from pathlib import Path

import numpy as np

from common.experiment_protocol import epoch_rng, fast_lhs, warmup_cosine_learning_rate


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / 'configs' / 'core_benchmarks.json'
MANIFEST = REPO_ROOT / 'configs' / 'core_benchmarks_manifest.json'


def test_protocol_is_frozen_for_core_benchmarks():
    config = json.loads(CONFIG.read_text(encoding='utf-8'))
    assert config['training_policy'] == {
        'learning_rate': 1e-3,
        'lr_schedule': 'linear_warmup_then_cosine',
        'lr_min': 1e-4,
        'warmup_epochs': 100,
        'checkpoint_policy': 'fixed_final_iteration',
    }
    assert config['seed_policy']['paired_runs'] == 5
    assert config['seed_policy']['master_seed_base'] == 19018
    assert set(config['problems']) == {
        'helmholtz_1_4', 'klein_gordon', 'lid_driven_cavity'
    }
    for problem in config['problems'].values():
        assert problem['epochs'] == 100000
        assert problem['record_every'] == 1000
        assert any(key.endswith('RESAMPLE_EVERY') and value == 1
                   for key, value in problem['environment'].items())


def test_manifest_contains_twelve_paired_jobs():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    assert len(manifest['jobs']) == 12
    assert len({(job['problem'], job['architecture'], job['optimizer'])
                for job in manifest['jobs']}) == 12


def test_sampling_and_schedule_are_deterministic():
    first = fast_lhs(epoch_rng(1301, 0), 32, 2)
    repeated = fast_lhs(epoch_rng(1301, 0), 32, 2)
    following = fast_lhs(epoch_rng(1301, 1), 32, 2)
    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, following)
    assert np.isclose(warmup_cosine_learning_rate(99, 40000, 1e-3, 1e-4, 100), 1e-3)
    assert np.isclose(warmup_cosine_learning_rate(39999, 40000, 1e-3, 1e-4, 100), 1e-4)


def test_all_training_scripts_record_mse_and_fixed_final_model():
    scripts = [
        REPO_ROOT / 'burgers' / 'train.py',
        REPO_ROOT / 'helmholtz' / 'train.py',
        REPO_ROOT / 'klein_gordon' / 'train.py',
        REPO_ROOT / 'lid_driven_cavity' / 'train.py',
    ]
    for script in scripts:
        source = script.read_text(encoding='utf-8')
        assert "'mse'" in source
        assert "'model_final.pt'" in source
        assert 'model_best.pt' not in source
