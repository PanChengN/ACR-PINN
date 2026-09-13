import torch

from common.optimization import (
    GradNormBalancer,
    BalancedResidualDecayRate,
    ResidualBasedAttention,
    canonical_optimizer,
    hutchinson_ntk_traces,
    mgda_combined_gradients,
    ntk_loss_weights,
)


def _conflicting_gradients():
    return [
        [torch.tensor([1.0, 0.0])],
        [torch.tensor([-1.0, 1.0])],
        [torch.tensor([0.5, 0.5])],
    ]


def test_mgda_returns_a_finite_simplex_weighted_direction():
    direction, weights = mgda_combined_gradients(_conflicting_gradients())
    assert len(direction) == 1
    assert torch.isfinite(direction[0]).all()
    assert torch.isfinite(weights).all()
    assert torch.all(weights >= 0)
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)


def test_gradnorm_keeps_positive_weights_with_fixed_mean_one():
    balancer = GradNormBalancer(3)
    _, weights = balancer.update_and_combine(
        _conflicting_gradients(),
        [torch.tensor(2.0), torch.tensor(1.0), torch.tensor(3.0)],
    )
    assert torch.all(weights > 0)
    assert torch.allclose(weights.mean(), torch.tensor(1.0), atol=1e-6)


def test_hutchinson_ntk_trace_and_trace_ratio_weights_are_finite_and_seeded():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    outputs = [parameter[0:1], parameter[1:2], parameter.sum().reshape(1)]
    first = hutchinson_ntk_traces(outputs, [parameter], torch.Generator().manual_seed(7))
    second = hutchinson_ntk_traces(outputs, [parameter], torch.Generator().manual_seed(7))
    assert torch.allclose(first, second)
    weights = ntk_loss_weights(first)
    assert torch.isfinite(first).all()
    assert torch.isfinite(weights).all()
    assert torch.all(weights > 0)


def test_new_optimizer_aliases_are_explicit():
    for name in ('mgda', 'gradnorm', 'ntk', 'ntk_weighting', 'rba', 'brdr'):
        assert canonical_optimizer(name) == ('ntk' if name == 'ntk_weighting' else name)


def test_rba_uses_persistent_detached_squared_attention_weights():
    rba = ResidualBasedAttention(eta=0.001, gamma=0.999)
    residual = torch.tensor([[2.0], [-1.0]], requires_grad=True)
    first = rba.update_and_weight(residual)
    assert torch.allclose(rba.weights, torch.tensor([[0.001], [0.0005]]))
    assert torch.allclose(first, torch.mean((rba.weights * residual) ** 2))
    second = rba.update_and_weight(residual)
    assert torch.allclose(rba.weights, torch.tensor([[0.001999], [0.0009995]]))
    second.backward()
    assert residual.grad is not None


def test_brdr_normalizes_pointwise_weights_and_corrects_gradients():
    brdr = BalancedResidualDecayRate(beta_c=0.5, beta_w=0.5)
    residual_a = torch.tensor([[2.0], [1.0]], requires_grad=True)
    residual_b = torch.tensor([[1.0]], requires_grad=True)
    loss = brdr.update_and_weight([residual_a, residual_b])
    assert brdr.step == 1
    assert torch.allclose(torch.cat([weight.reshape(-1) for weight in brdr.weights]).mean(), torch.tensor(1.0))
    loss.backward()
    before = residual_a.grad.detach().clone()
    scale = brdr.correct_gradients(loss, [residual_a, residual_b], learning_rate=0.1)
    assert torch.isfinite(scale)
    assert torch.isfinite(residual_a.grad).all()
    assert not torch.allclose(before, residual_a.grad)
