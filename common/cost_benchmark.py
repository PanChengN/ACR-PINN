"""Utilities for controlled CUDA runtime and memory measurements."""

import json
import os
import platform
import time
from pathlib import Path

import torch


class CudaCostProfiler:
    """Measure only optimizer-update work, excluding sampling and diagnostics."""

    def __init__(self, device, epochs):
        self.enabled = os.getenv('COST_BENCHMARK', '0').strip().lower() in ('1', 'true', 'yes')
        self.device = device
        self.warmup_steps = int(os.getenv('COST_WARMUP_STEPS', '100'))
        self.timed_steps = int(os.getenv('COST_TIMED_STEPS', '200'))
        self.step_times_sec = []
        self._started_at = None
        if not self.enabled:
            return
        if device.type != 'cuda':
            raise RuntimeError('Controlled cost measurement requires a CUDA device.')
        if epochs != self.warmup_steps + self.timed_steps:
            raise ValueError('EPOCHS must equal COST_WARMUP_STEPS + COST_TIMED_STEPS.')

    def start_step(self, epoch):
        if not self.enabled:
            return
        if epoch == self.warmup_steps:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        if epoch >= self.warmup_steps:
            torch.cuda.synchronize(self.device)
            self._started_at = time.perf_counter()

    def end_step(self, epoch):
        if self.enabled and epoch >= self.warmup_steps:
            torch.cuda.synchronize(self.device)
            self.step_times_sec.append(time.perf_counter() - self._started_at)

    def write(self, run_log_dir, model, metadata):
        if not self.enabled:
            return
        if len(self.step_times_sec) != self.timed_steps:
            raise RuntimeError(f'Expected {self.timed_steps} timed steps, got {len(self.step_times_sec)}.')
        values = torch.tensor(self.step_times_sec, dtype=torch.float64)
        payload = {
            'protocol': 'controlled-cost-v1',
            'measurement_scope': 'optimizer update only; dynamic sampling, diagnostics, and evaluation excluded',
            'warmup_steps': self.warmup_steps,
            'timed_steps': self.timed_steps,
            'step_times_sec': self.step_times_sec,
            'step_time_sec': {
                'mean': float(values.mean()), 'median': float(values.median()),
                'std': float(values.std(unbiased=True)),
                'q05': float(torch.quantile(values, 0.05)), 'q95': float(torch.quantile(values, 0.95)),
            },
            'peak_memory_allocated_mib': torch.cuda.max_memory_allocated(self.device) / (1024 ** 2),
            'parameter_count': sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            'device': torch.cuda.get_device_name(self.device),
            'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
            'python': platform.python_version(), **metadata,
        }
        Path(run_log_dir, 'cost_measurement.json').write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
