"""Shared deterministic sampling and learning-rate helpers for article-aligned runs."""

import json
import math

import numpy as np

from common.reproducibility import file_sha256


def fast_lhs(rng, samples, dimensions):
    values = np.empty((samples, dimensions), dtype=np.float32)
    for dimension in range(dimensions):
        values[:, dimension] = (rng.permutation(samples) + rng.random(samples)) / samples
    return values


def epoch_rng(sample_seed, epoch, resample_every=1):
    if resample_every < 1:
        raise ValueError('resample_every must be at least one')
    return np.random.default_rng(
        np.random.SeedSequence([sample_seed, epoch // resample_every])
    )


def warmup_cosine_learning_rate(epoch, total_epochs, initial_lr, final_lr, warmup_epochs):
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return final_lr + (initial_lr - final_lr) * (epoch + 1) / warmup_epochs
    cosine_steps = max(1, total_epochs - warmup_epochs - 1)
    progress = min(1.0, max(0.0, (epoch - warmup_epochs) / cosine_steps))
    return final_lr + 0.5 * (initial_lr - final_lr) * (1.0 + math.cos(math.pi * progress))


def write_sampling_descriptor(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + '\n'
    if path.exists() and path.read_text(encoding='utf-8') != serialized:
        raise RuntimeError(f'Sampling descriptor collision: {path}')
    path.write_text(serialized, encoding='utf-8')
    return path, file_sha256(path)
