#!/usr/bin/env python
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / 'configs' / 'core_benchmarks_manifest.json'


def command_for(job, config_path, gpu):
    command = [
        sys.executable, str(REPO_ROOT / 'experiments' / 'run_experiment.py'),
        '--config', str(config_path), '--problem', job['problem'],
        '--architecture', job['architecture'], '--optimizer', job['optimizer'],
        '--cuda-device', str(gpu),
    ]
    optional_arguments = {
        'start_run': '--start-run',
        'epochs': '--epochs',
        'num_runs': '--num-runs',
        'record_every': '--record-every',
        'result_tag': '--result-tag',
        'master_seed_base': '--master-seed-base',
        'sample_seed_base': '--sample-seed-base',
        'init_seed_base': '--init-seed-base',
        'optimizer_order_seed_base': '--optimizer-order-seed-base',
    }
    for key, flag in optional_arguments.items():
        if key in job:
            command.extend([flag, str(job[key])])
    return command


def run_job(job, command, log_dir):
    name = f"{job['problem']}__{job['architecture']}__{job['optimizer']}"
    log_path = log_dir / f'{name}.log'
    with log_path.open('w', encoding='utf-8') as handle:
        subprocess.run(command, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
    return name, log_path


def run_lane(items, log_dir):
    """Run one strictly sequential job queue for a single GPU."""
    for item in items:
        name, log_path = run_job(item['job'], item['command'], log_dir)
        print(f'completed {name}: {log_path}', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Safely launch a frozen experiment matrix.')
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--confirm', default='')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    config_path = REPO_ROOT / manifest['config']
    gpus = [item.strip() for item in args.gpus.split(',') if item.strip()]
    planned = [
        {'job': job, 'gpu': gpus[index % len(gpus)],
         'command': command_for(job, config_path, gpus[index % len(gpus)])}
        for index, job in enumerate(manifest['jobs'])
    ]
    print(json.dumps({'manifest_id': manifest['manifest_id'], 'jobs': planned}, indent=2))
    if not args.execute:
        return
    confirmation = manifest['manifest_id']
    if args.confirm != confirmation:
        raise SystemExit(f'Execution refused: pass --confirm {confirmation}')
    log_dir = REPO_ROOT / 'formal_logs' / manifest['manifest_id']
    log_dir.mkdir(parents=True, exist_ok=True)
    lanes = {gpu: [] for gpu in gpus}
    for item in planned:
        lanes[item['gpu']].append(item)
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(run_lane, lanes[gpu], log_dir) for gpu in gpus]
        for future in as_completed(futures):
            future.result()


if __name__ == '__main__':
    main()
