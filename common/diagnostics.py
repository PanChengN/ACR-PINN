import json
import math
import time
from pathlib import Path

import numpy as np
import torch


def collect_task_gradients(losses, parameters):
    task_gradients = []
    for loss in losses:
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        task_gradients.append([
            gradient.detach().clone() if gradient is not None else torch.zeros_like(parameter)
            for gradient, parameter in zip(gradients, parameters)
        ])
    return task_gradients


def _flatten(gradients):
    return torch.cat([gradient.reshape(-1) for gradient in gradients])


def gradient_statistics(task_names, task_gradients, named_parameters, projected_gradients=None):
    flat = [_flatten(gradients) for gradients in task_gradients]
    norms = {name: float(torch.linalg.vector_norm(vector).item()) for name, vector in zip(task_names, flat)}
    pairwise = []
    conflicts = 0
    for first in range(len(flat)):
        for second in range(first + 1, len(flat)):
            denominator = torch.linalg.vector_norm(flat[first]) * torch.linalg.vector_norm(flat[second])
            cosine = float((torch.dot(flat[first], flat[second]) / denominator).item()) if denominator > 0 else 0.0
            conflicts += int(cosine < 0)
            pairwise.append({'first': task_names[first], 'second': task_names[second], 'cosine': cosine})
    raw_sum = [sum(per_parameter) for per_parameter in zip(*task_gradients)]
    if projected_gradients is None:
        projection_ratio = 0.0
    else:
        raw_flat, projected_flat = _flatten(raw_sum), _flatten(projected_gradients)
        denominator = torch.linalg.vector_norm(raw_flat)
        projection_ratio = float((torch.linalg.vector_norm(projected_flat - raw_flat) / denominator).item()) if denominator > 0 else 0.0
    layerwise = {}
    for parameter_index, (parameter_name, _) in enumerate(named_parameters):
        layer = parameter_name.rsplit('.', 1)[0]
        layerwise.setdefault(layer, {})
        for task_name, gradients in zip(task_names, task_gradients):
            value = float(torch.linalg.vector_norm(gradients[parameter_index]).item())
            layerwise[layer][task_name] = math.sqrt(layerwise[layer].get(task_name, 0.0) ** 2 + value ** 2)
    return {
        'gradient_norms': norms,
        'pairwise_cosines': pairwise,
        'conflict_rate': conflicts / len(pairwise) if pairwise else 0.0,
        'projection_ratio': projection_ratio,
        'layerwise_gradient_norms': layerwise,
    }


def gate_statistics(model, inputs):
    if not hasattr(model, 'forward_with_gates'):
        return []
    with torch.no_grad():
        _, gates = model.forward_with_gates(inputs)
    result = []
    for layer, weights in enumerate(gates):
        clipped = weights.clamp_min(1e-12)
        entropy = -(clipped * clipped.log()).sum(dim=1)
        result.append({
            'layer': layer,
            'mean_branch_1': float(weights[:, 0, :].mean().item()),
            'mean_branch_2': float(weights[:, 1, :].mean().item()),
            'std': float(weights.std().item()),
            'entropy_mean': float(entropy.mean().item()),
            'saturation_rate': float(((weights < 0.05) | (weights > 0.95)).float().mean().item()),
        })
    return result


class DiagnosticsRecorder:
    def __init__(self, model, log_dir, device, every):
        self.model = model
        self.device = device
        self.every = every
        self.path = Path(log_dir) / 'diagnostics.jsonl'
        self.cost_path = Path(log_dir) / 'cost_summary.json'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text('', encoding='utf-8')
        self.step_times = []
        self.parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)

    def should_record(self, iteration, total_iterations):
        return iteration % self.every == 0 or iteration == total_iterations - 1

    def start_step(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        self._step_start = time.perf_counter()

    def end_step(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        self.step_times.append(time.perf_counter() - self._step_start)

    def record(self, iteration, task_names, task_gradients, named_parameters,
               projected_gradients=None, gate_inputs=None, extra_statistics=None):
        payload = {'iteration': iteration}
        payload.update(gradient_statistics(
            task_names, task_gradients, named_parameters, projected_gradients
        ))
        payload['gates'] = gate_statistics(self.model, gate_inputs) if gate_inputs is not None else []
        if extra_statistics:
            payload.update(extra_statistics)
        with self.path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(payload, sort_keys=True) + '\n')

    def finalize(self, total_training_time_sec):
        peak = torch.cuda.max_memory_allocated(self.device) if self.device.type == 'cuda' else 0
        payload = {
            'trainable_parameters': self.parameter_count,
            'optimization_step_time_mean_sec': float(np.mean(self.step_times)),
            'optimization_step_time_std_sec': float(np.std(self.step_times)),
            'optimization_steps_measured': len(self.step_times),
            'total_training_time_sec': float(total_training_time_sec),
            'peak_memory_allocated_bytes': int(peak),
            'device': str(self.device),
        }
        with self.cost_path.open('w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write('\n')
        return payload
