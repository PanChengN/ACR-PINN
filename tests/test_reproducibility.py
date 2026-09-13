from pathlib import Path

import numpy as np

from common.reproducibility import (
    latin_hypercube,
    load_or_create_npz,
    git_revision,
    seed_bundle,
    spawned_rngs,
    stable_config_sha256,
)


MASTER_SEEDS = [19018, 19019, 19020, 19021, 19022]


def test_seed_bundle_separates_random_sources():
    seeds = seed_bundle(0)
    assert seeds['master_seed'] == seeds['init_seed']
    assert len({seeds['sample_seed'], seeds['init_seed'], seeds['optimizer_order_seed']}) == 3
    assert seeds == {
        'master_seed': 19018,
        'sample_seed': 20018,
        'init_seed': 19018,
        'optimizer_order_seed': 21018,
    }


def test_five_paired_master_seeds_have_stable_independent_offsets():
    bundles = [seed_bundle(index) for index in range(5)]
    assert [bundle['master_seed'] for bundle in bundles] == MASTER_SEEDS
    for bundle in bundles:
        assert bundle['init_seed'] == bundle['master_seed']
        assert bundle['sample_seed'] == bundle['master_seed'] + 1000
        assert bundle['optimizer_order_seed'] == bundle['master_seed'] + 2000


def test_lhs_is_reproducible_with_explicit_rng():
    a = latin_hypercube(2, 32, spawned_rngs(1234, 1)[0])
    b = latin_hypercube(2, 32, spawned_rngs(1234, 1)[0])
    c = latin_hypercube(2, 32, spawned_rngs(1235, 1)[0])
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_persisted_dataset_is_reused(tmp_path: Path):
    dataset_path = tmp_path / 'points.npz'

    def create():
        return {'X_f': latin_hypercube(2, 16, spawned_rngs(7, 1)[0])}

    first, first_hash = load_or_create_npz(dataset_path, create)
    second, second_hash = load_or_create_npz(
        dataset_path,
        lambda: {'X_f': np.full((16, 2), -1.0)},
    )
    assert np.array_equal(first['X_f'], second['X_f'])
    assert first_hash == second_hash


def test_config_hash_is_order_independent():
    assert stable_config_sha256({'a': 1, 'b': 2}) == stable_config_sha256({'b': 2, 'a': 1})


def test_git_revision_has_auditable_shape(tmp_path: Path):
    revision = git_revision(tmp_path)
    assert set(revision) == {'commit', 'dirty'}
