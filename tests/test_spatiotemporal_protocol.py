import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / 'configs' / 'spatiotemporal.json'
MANIFEST = REPO_ROOT / 'configs' / 'spatiotemporal_manifest.json'


def test_spatiotemporal_freezes_joint_separate_longtime_and_paired_seeds():
    config = json.loads(CONFIG.read_text(encoding='utf-8'))
    assert config['architectures'] == ['lda', 'separate_st_lda']
    assert config['optimizer_strategies'] == ['sum', 'pcgrad']
    assert config['seed_policy']['paired_runs'] == 5
    assert set(config['problems']) == {
        'klein_gordon_t1', 'klein_gordon_t2', 'klein_gordon_t4',
    }
    assert [config['problems'][name]['environment']['KLEIN_GORDON_TIME_MAX']
            for name in ('klein_gordon_t1', 'klein_gordon_t2', 'klein_gordon_t4')] == [1.0, 2.0, 4.0]
    assert all(problem['epochs'] == 100000 for problem in config['problems'].values())
    assert all(problem['environment']['KLEIN_GORDON_DYNAMIC_SAMPLING'] == 1
               for problem in config['problems'].values())


def test_spatiotemporal_manifest_covers_all_conditions():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    conditions = {(job['problem'], job['architecture'], job['optimizer'])
                  for job in manifest['jobs']}
    expected = {
        (problem, architecture, optimizer)
        for problem in ('klein_gordon_t1', 'klein_gordon_t2', 'klein_gordon_t4')
        for architecture in ('mlp', 'lda', 'separate_st_lda')
        for optimizer in ('sum', 'pcgrad')
    }
    assert conditions == expected
