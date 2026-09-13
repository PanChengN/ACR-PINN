import json

import torch

from common.diagnostics import (
    DiagnosticsRecorder,
    collect_task_gradients,
    gate_statistics,
    gradient_statistics,
)
from common.models import build_model
from common.optimization import project_conflicting_gradients
from common.reproducibility import optimizer_order_generator


def test_gradient_diagnostics_fields_and_ranges():
    model = build_model('mlp', [2, 4, 1])
    inputs = torch.randn(8, 2)
    output = model(inputs)
    losses = [(output ** 2).mean(), ((output - 1.0) ** 2).mean()]
    parameters = list(model.parameters())
    gradients = collect_task_gradients(losses, parameters)
    projected = project_conflicting_gradients(gradients, optimizer_order_generator(3))
    stats = gradient_statistics(['pde', 'bc'], gradients, list(model.named_parameters()), projected)
    assert set(stats) == {
        'gradient_norms', 'pairwise_cosines', 'conflict_rate',
        'projection_ratio', 'layerwise_gradient_norms'
    }
    assert -1.0 <= stats['pairwise_cosines'][0]['cosine'] <= 1.0
    assert 0.0 <= stats['conflict_rate'] <= 1.0
    assert stats['projection_ratio'] >= 0.0


def test_gate_statistics_are_present_only_for_lda():
    inputs = torch.randn(16, 2)
    lda_stats = gate_statistics(build_model('lda', [2, 4, 4, 1]), inputs)
    assert len(lda_stats) == 2
    assert all(0.0 <= row['saturation_rate'] <= 1.0 for row in lda_stats)
    assert gate_statistics(build_model('mlp', [2, 4, 1]), inputs) == []


def test_cost_summary_is_machine_readable(tmp_path):
    model = build_model('mlp', [2, 4, 1])
    recorder = DiagnosticsRecorder(model, tmp_path, torch.device('cpu'), every=1)
    recorder.start_step()
    _ = model(torch.zeros(2, 2))
    recorder.end_step()
    summary = recorder.finalize(0.1)
    saved = json.loads((tmp_path / 'cost_summary.json').read_text())
    assert saved == summary
    assert summary['trainable_parameters'] == sum(p.numel() for p in model.parameters())
    assert summary['optimization_steps_measured'] == 1
