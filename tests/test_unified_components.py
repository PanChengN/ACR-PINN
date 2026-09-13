import json
from pathlib import Path

import torch

from common.models import (
    DirectCoordinateInjection,
    FixedAverageResidual,
    FourierFeatureMLP,
    LDA,
    MLP,
    ModifiedMLP,
    SeparateSpaceTimeLDA,
    SingleEncoderResidual,
    StaticFeatureGateResidual,
    build_model,
    canonical_architecture,
    parameter_count,
    parameter_matched_mlp_layers,
)
from common.optimization import canonical_optimizer, project_conflicting_gradients
from common.reproducibility import optimizer_order_generator, seed_model_initialization


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_architecture_aliases_and_output_shapes():
    assert canonical_architecture('attention') == 'lda'
    assert canonical_architecture('separate_space_time_lda') == 'separate_st_lda'
    assert isinstance(build_model('mlp', [2, 8, 1]), MLP)
    assert isinstance(build_model('lda', [2, 8, 1]), LDA)
    assert isinstance(build_model('modified_mlp', [2, 8, 1]), ModifiedMLP)
    assert isinstance(build_model('fourier_features', [2, 8, 1]), FourierFeatureMLP)
    inputs = torch.zeros(7, 2)
    assert build_model('mlp', [2, 8, 1])(inputs).shape == (7, 1)
    assert build_model('lda', [2, 8, 1])(inputs).shape == (7, 1)


def test_baseline_architecture_baselines_are_seeded_and_differentiable():
    layers = [2, 8, 8, 1]
    inputs = torch.randn(7, 2, requires_grad=True)
    for architecture in ('modified_mlp', 'fourier_mlp'):
        seed_model_initialization(19018)
        first = build_model(architecture, layers, fourier_feature_count=6).state_dict()
        seed_model_initialization(19018)
        second = build_model(architecture, layers, fourier_feature_count=6).state_dict()
        assert all(torch.equal(first[key], second[key]) for key in first)
        model = build_model(architecture, layers, fourier_feature_count=6)
        output = model(inputs)
        assert output.shape == (7, 1)
        gradients = torch.autograd.grad(output.sum(), tuple(model.parameters()), allow_unused=True)
        assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)


def test_separate_space_time_lda_is_gated_and_parameter_fair():
    layers = [2, 50, 50, 50, 50, 1]
    joint = build_model('lda', layers)
    separate = build_model('separate_st_lda', layers)
    assert isinstance(separate, SeparateSpaceTimeLDA)
    assert separate(torch.randn(7, 2)).shape == (7, 1)
    assert abs(parameter_count(separate) - parameter_count(joint)) / parameter_count(joint) < 0.01
    _, gates = separate.forward_with_gates(torch.randn(7, 2))
    assert len(gates) == 4
    assert all(gate.shape == (7, 2, 50) for gate in gates)


def test_lda_control_architectures_have_expected_types_shapes_and_gradients():
    expected_types = {
        'mlp_param_matched': MLP,
        'direct_coordinate': DirectCoordinateInjection,
        'single_encoder_residual': SingleEncoderResidual,
        'fixed_average': FixedAverageResidual,
        'static_gate': StaticFeatureGateResidual,
    }
    inputs = torch.randn(7, 2, requires_grad=True)
    for architecture, expected_type in expected_types.items():
        model = build_model(architecture, [2, 8, 8, 1])
        assert isinstance(model, expected_type)
        output = model(inputs)
        assert output.shape == (7, 1)
        gradients = torch.autograd.grad(output.sum(), tuple(model.parameters()), allow_unused=True)
        assert all(gradient is not None for gradient in gradients)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_lda_control_parameter_counts_match_documented_values():
    layers = [2, 50, 50, 50, 50, 1]
    expected = {
        'mlp': 7851,
        'mlp_param_matched': 59781,
        'direct_coordinate': 8251,
        'single_encoder_residual': 8455,
        'fixed_average': 9055,
        'static_gate': 9455,
        'lda': 59655,
    }
    assert parameter_matched_mlp_layers(layers) == [2, 140, 140, 140, 140, 1]
    models = {name: build_model(name, layers) for name in expected}
    assert {name: parameter_count(model) for name, model in models.items()} == expected
    assert models['mlp_param_matched'].effective_layers == [2, 140, 140, 140, 140, 1]
    assert models['lda'].effective_layers == layers
    assert abs(expected['mlp_param_matched'] - expected['lda']) / expected['lda'] < 0.01


def test_parameter_matched_width_selection_does_not_change_seeded_initialization():
    layers = [2, 50, 50, 50, 50, 1]
    seed_model_initialization(7788)
    first = build_model('mlp_param_matched', layers).state_dict()
    seed_model_initialization(7788)
    second = build_model('mlp_param_matched', layers).state_dict()
    assert all(torch.equal(first[key], second[key]) for key in first)


def test_shared_model_initialization_is_reproducible():
    seed_model_initialization(2234)
    first = build_model('lda', [2, 8, 1]).state_dict()
    seed_model_initialization(2234)
    second = build_model('lda', [2, 8, 1]).state_dict()
    assert all(torch.equal(first[key], second[key]) for key in first)


def test_stabilized_lda_starts_from_symmetric_gates_and_small_residuals():
    model = build_model('lda', [2, 8, 8, 1])
    _, gates = model.forward_with_gates(torch.randn(5, 2))
    for gate in gates:
        assert torch.allclose(gate[:, 0, :], torch.full_like(gate[:, 0, :], 0.5))
        assert torch.allclose(gate[:, 1, :], torch.full_like(gate[:, 1, :], 0.5))
    assert torch.allclose(model.residual_scales.detach(), torch.full((2,), 0.1))


def test_lda_gate_interventions_are_well_defined_and_deterministic():
    model = build_model('lda', [2, 8, 8, 1])
    inputs = torch.randn(11, 2)
    with torch.no_grad():
        for gate in model.gates:
            gate[-1].weight.normal_(mean=0.0, std=0.2)
    native = model(inputs)
    assert torch.allclose(native, model.forward_with_gate_intervention(inputs, 'native'))
    for mode in ('uniform', 'swap'):
        output = model.forward_with_gate_intervention(inputs, mode)
        assert output.shape == native.shape
        assert torch.isfinite(output).all()
    first_generator = torch.Generator().manual_seed(991)
    second_generator = torch.Generator().manual_seed(991)
    first = model.forward_with_gate_intervention(
        inputs, 'sample_permute', generator=first_generator
    )
    second = model.forward_with_gate_intervention(
        inputs, 'sample_permute', generator=second_generator
    )
    assert torch.allclose(first, second)
    _, gates = model.forward_with_gates(inputs)
    layer_means = [gate.mean(dim=0) for gate in gates]
    averaged = model.forward_with_gate_intervention(inputs, 'layer_mean', layer_means)
    assert averaged.shape == native.shape


def test_lda_baseline_switch_keeps_unit_residual_injection():
    model = build_model('lda', [2, 8, 1], lda_stabilized=False)
    assert not isinstance(model.residual_scales, torch.nn.Parameter)
    assert torch.equal(model.residual_scales, torch.ones(1))


def test_optimizer_aliases_and_pcgrad_projection():
    assert canonical_optimizer('pinn') == 'sum'
    assert canonical_optimizer('pcgrad') == 'pcgrad'
    task_gradients = [[torch.tensor([1.0, 0.0])], [torch.tensor([-1.0, 1.0])]]
    projected = project_conflicting_gradients(task_gradients, optimizer_order_generator(3234))
    assert len(projected) == 1
    assert torch.isfinite(projected[0]).all()


def test_core_configuration_has_all_declared_problems():
    config = json.loads((REPO_ROOT / 'configs' / 'core_benchmarks.json').read_text())
    assert set(config['problems']) == {
        'helmholtz_1_4', 'klein_gordon', 'lid_driven_cavity'
    }
    assert config['architectures'] == ['mlp', 'lda']
    assert config['optimizer_strategies'] == ['sum', 'pcgrad']
    assert config['problems']['lid_driven_cavity']['layers'] == [2, 50, 50, 50, 50, 2]


def test_all_training_scripts_use_shared_runtime_components():
    scripts = [
        'burgers/train.py',
        'helmholtz/train.py',
        'klein_gordon/train.py',
        'lid_driven_cavity/train.py',
        'poisson_5d/train.py',
        'schrodinger/train.py',
    ]
    for relative_path in scripts:
        source = (REPO_ROOT / relative_path).read_text()
        assert 'return build_shared_model(' in source
        assert 'project_conflicting_gradients(' in source
