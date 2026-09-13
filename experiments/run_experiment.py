#!/usr/bin/env python
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / 'configs' / 'core_benchmarks.json'


def parse_args():
    parser = argparse.ArgumentParser(description='Run one configured ACR-PINN experiment.')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--problem', required=True)
    parser.add_argument('--architecture', choices=(
        'mlp', 'mlp_param_matched', 'direct_coordinate',
        'single_encoder_residual', 'fixed_average', 'static_gate', 'lda', 'separate_st_lda',
        'modified_mlp', 'fourier_mlp',
    ), required=True)
    parser.add_argument(
        '--optimizer',
        choices=('sum', 'pcgrad', 'mgda', 'gradnorm', 'ntk', 'lra', 'rba', 'brdr'),
        required=True,
    )
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--num-runs', type=int)
    parser.add_argument('--start-run', type=int, default=0,
                        help='Zero-based run index to resume from; NUM_RUNS remains the total target.')
    parser.add_argument('--record-every', type=int)
    parser.add_argument('--result-tag')
    # These overrides are intentionally per-process.  They are needed for
    # factorial robustness studies where the sample and initialization stream
    # must vary independently, while existing paired manifests retain their
    # configuration-defined seed policy unchanged.
    parser.add_argument('--master-seed-base', type=int)
    parser.add_argument('--sample-seed-base', type=int)
    parser.add_argument('--init-seed-base', type=int)
    parser.add_argument('--optimizer-order-seed-base', type=int)
    parser.add_argument('--cuda-device', default='0')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def build_command(args):
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if args.problem not in config['problems']:
        raise ValueError(f"Unknown problem '{args.problem}'. Choices: {sorted(config['problems'])}")
    problem = config['problems'][args.problem]
    seed_policy = config['seed_policy']
    if 'master_seed_base' in seed_policy:
        master_seed_base = int(seed_policy['master_seed_base'])
        init_seed_base = master_seed_base + int(seed_policy.get('init_seed_offset', 0))
        sample_seed_base = master_seed_base + int(seed_policy.get('sample_seed_offset', 1000))
        optimizer_seed_base = master_seed_base + int(
            seed_policy.get('optimizer_order_seed_offset', 2000)
        )
    else:
        # Support configurations that provide explicit component seed bases.
        sample_seed_base = int(seed_policy['sample_seed_base'])
        init_seed_base = int(seed_policy['init_seed_base'])
        optimizer_seed_base = int(seed_policy['optimizer_order_seed_base'])
        master_seed_base = init_seed_base
    if args.master_seed_base is not None:
        master_seed_base = args.master_seed_base
    if args.sample_seed_base is not None:
        sample_seed_base = args.sample_seed_base
    if args.init_seed_base is not None:
        init_seed_base = args.init_seed_base
    if args.optimizer_order_seed_base is not None:
        optimizer_seed_base = args.optimizer_order_seed_base
    training_policy = config.get('training_policy', {})
    result_tag = args.result_tag or f"{config.get('result_prefix', '')}{args.optimizer}_{args.architecture}"
    env = os.environ.copy()
    env.update({
        'RUN_ALL': '0',
        'MODE': args.optimizer,
        'NET_ARCH': args.architecture,
        'EPOCHS': str(args.epochs if args.epochs is not None else problem['epochs']),
        'NUM_RUNS': str(args.num_runs if args.num_runs is not None else seed_policy['paired_runs']),
        'RUN_START_INDEX': str(args.start_run),
        'RECORD_EVERY': str(args.record_every if args.record_every is not None else problem['record_every']),
        'MASTER_SEED_BASE': str(master_seed_base),
        'SAMPLE_SEED_BASE': str(sample_seed_base),
        'INIT_SEED_BASE': str(init_seed_base),
        'OPTIMIZER_SEED_BASE': str(optimizer_seed_base),
        'LEARNING_RATE': str(training_policy.get('learning_rate', 1e-3)),
        'LR_SCHEDULE': str(training_policy.get('lr_schedule', 'constant')),
        'LR_MIN': str(training_policy.get('lr_min', 1e-5)),
        'WARMUP_EPOCHS': str(training_policy.get('warmup_epochs', 0)),
        'RESULT_TAG': result_tag,
        'CUDA_VISIBLE_DEVICES': args.cuda_device,
        'MPLBACKEND': 'Agg',
    })
    env.update({key: str(value) for key, value in problem.get('environment', {}).items()})
    command = [sys.executable, str(REPO_ROOT / problem['script'])]
    return command, env, config, problem


def main():
    args = parse_args()
    command, env, _, _ = build_command(args)
    preview = {
        'command': command,
        'problem': args.problem,
        'architecture': args.architecture,
        'optimizer': args.optimizer,
        'epochs': int(env['EPOCHS']),
        'num_runs': int(env['NUM_RUNS']),
        'start_run': int(env['RUN_START_INDEX']),
        'record_every': int(env['RECORD_EVERY']),
        'result_tag': env['RESULT_TAG'],
        'learning_rate': float(env['LEARNING_RATE']),
        'lr_schedule': env['LR_SCHEDULE'],
        'lr_min': float(env['LR_MIN']),
        'master_seeds': [
            int(env['MASTER_SEED_BASE']) + run_idx
            for run_idx in range(int(env['NUM_RUNS']))
        ],
        'derived_seed_bases': {
            'sample': int(env['SAMPLE_SEED_BASE']),
            'init': int(env['INIT_SEED_BASE']),
            'optimizer_order': int(env['OPTIMIZER_SEED_BASE']),
        },
    }
    print(json.dumps(preview, indent=2))
    if not args.dry_run:
        subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
