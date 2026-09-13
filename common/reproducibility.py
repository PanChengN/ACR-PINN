import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
from scipy.stats import qmc


def seed_bundle(run_idx, sample_base=20018, init_base=19018, optimizer_base=21018):
    """Return the master seed and its independent streams for one paired run.

    The initialization stream is the master seed itself. Sampling and optimizer
    ordering use deterministic offsets so consuming randomness in one component
    cannot perturb either of the others.
    """
    master_seed = int(init_base + run_idx)
    return {
        'master_seed': master_seed,
        'sample_seed': int(sample_base + run_idx),
        'init_seed': master_seed,
        'optimizer_order_seed': int(optimizer_base + run_idx),
    }


def seed_model_initialization(seed):
    """Seed model initialization without controlling dataset generation."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def optimizer_order_generator(seed):
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)
    return generator


def spawned_rngs(seed, count):
    """Create independent NumPy generators from one sample seed."""
    return [np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(count)]


def latin_hypercube(dim, samples, rng):
    """Generate a plain scrambled LHS in [0, 1)^dim using an explicit RNG."""
    try:
        engine = qmc.LatinHypercube(
            d=dim,
            scramble=True,
            optimization=None,
            rng=rng,
        )
    except TypeError as error:
        if "unexpected keyword argument 'rng'" not in str(error):
            raise
        # SciPy < 1.15 predates the SPEC-007 ``rng`` keyword but accepts
        # the same explicit numpy Generator through its legacy ``seed`` name.
        engine = qmc.LatinHypercube(
            d=dim,
            scramble=True,
            optimization=None,
            seed=rng,
        )
    return engine.random(n=samples)


def load_or_create_npz(path, create_fn):
    """Persist deterministic datasets so all paired methods read identical points."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        arrays = create_fn()
        temp_path = path.with_name(f'.{path.stem}.{os.getpid()}.tmp.npz')
        np.savez_compressed(temp_path, **arrays)
        os.replace(temp_path, path)
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    return arrays, file_sha256(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def stable_config_sha256(config):
    payload = json.dumps(config, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def git_revision(repo_root):
    """Return the source revision and dirty flag without making Git mandatory."""
    try:
        commit = subprocess.check_output(
            ['git', '-C', str(repo_root), 'rev-parse', 'HEAD'],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(subprocess.check_output(
            ['git', '-C', str(repo_root), 'status', '--porcelain'],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip())
        return {'commit': commit, 'dirty': dirty}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {'commit': None, 'dirty': None}


def write_run_metadata(path, config, seeds, dataset_path, dataset_sha256):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'config': config,
        'config_sha256': stable_config_sha256(config),
        'seeds': seeds,
        'dataset_path': str(Path(dataset_path).resolve()),
        'dataset_sha256': dataset_sha256,
        'code_revision': git_revision(Path(__file__).resolve().parents[1]),
        'checkpoint_policy': 'fixed_final_iteration',
        'test_metrics_used_for_selection': False,
    }
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')
    return payload
