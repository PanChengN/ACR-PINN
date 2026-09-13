from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_SCRIPTS = [
    REPO_ROOT / 'burgers' / 'train.py',
    REPO_ROOT / 'helmholtz' / 'train.py',
    REPO_ROOT / 'klein_gordon' / 'train.py',
    REPO_ROOT / 'lid_driven_cavity' / 'train.py',
    REPO_ROOT / 'schrodinger' / 'train.py',
    REPO_ROOT / 'poisson_5d' / 'train.py',
]


@pytest.mark.parametrize('script_path', TRAINING_SCRIPTS)
def test_primary_checkpoint_cannot_be_selected_by_test_error(script_path: Path):
    source = script_path.read_text(encoding='utf-8')
    forbidden = ('best_state', 'best_pred', 'model_best.pt', 'np.argmin(l2_values)')
    assert not any(token in source for token in forbidden)
    assert "'model_final.pt'" in source
    assert 'fixed final L2' not in source.lower()  # guard accidental hard-coded result text


@pytest.mark.parametrize('script_path', TRAINING_SCRIPTS)
def test_reproducibility_metadata_is_written_for_every_run(script_path: Path):
    source = script_path.read_text(encoding='utf-8')
    assert 'seed_bundle(' in source
    assert 'write_run_metadata(' in source
    assert 'dataset_sha256' in source
    assert 'optimizer_order_generator(' in source
