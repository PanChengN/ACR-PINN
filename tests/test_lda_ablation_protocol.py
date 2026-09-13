import json
from pathlib import Path

import numpy as np

from experiments.analyze_gate_behavior import INTERVENTIONS, relative_metrics


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / 'configs' / 'lda_ablation.json'
MANIFEST = REPO_ROOT / 'configs' / 'lda_ablation_manifest.json'


def test_ablation_matrix_covers_three_pdes_seven_architectures_and_five_seeds():
    config = json.loads(CONFIG.read_text())
    assert set(config['problems']) == {'burgers', 'helmholtz_4_4', 'klein_gordon'}
    assert config['seed_policy']['paired_runs'] == 5
    assert config['optimizer_strategies'] == ['sum']
    assert config['architectures'] == [
        'mlp', 'mlp_param_matched', 'direct_coordinate',
        'single_encoder_residual', 'fixed_average', 'static_gate', 'lda',
    ]
    assert config['problems']['burgers']['epochs'] == 40000
    assert config['problems']['helmholtz_4_4']['epochs'] == 100000
    assert config['problems']['klein_gordon']['epochs'] == 100000
    assert config['problems']['burgers']['samples'] == {
        'residual': 20000, 'boundary_total': 500, 'initial': 500,
    }
    assert config['problems']['helmholtz_4_4']['samples']['residual'] == 20000


def test_ablation_manifest_launches_only_five_control_variants():
    manifest = json.loads(MANIFEST.read_text())
    assert manifest['config'] == 'configs/lda_ablation.json'
    jobs = manifest['jobs']
    assert len(jobs) == 15
    assert {job['problem'] for job in jobs} == {
        'burgers', 'helmholtz_4_4', 'klein_gordon',
    }
    assert {job['architecture'] for job in jobs} == {
        'mlp_param_matched', 'direct_coordinate', 'single_encoder_residual',
        'fixed_average', 'static_gate',
    }
    assert {job['optimizer'] for job in jobs} == {'sum'}


def test_gate_analysis_interventions_and_metrics_are_frozen():
    assert INTERVENTIONS == (
        'native', 'uniform', 'swap', 'sample_permute', 'layer_mean',
    )
    reference = np.array([[1.0], [2.0], [3.0]])
    metrics = relative_metrics(reference.copy(), reference)
    assert metrics == {'relative_l2': 0.0, 'relative_linf': 0.0, 'mse': 0.0}
