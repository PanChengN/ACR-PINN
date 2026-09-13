#!/usr/bin/env python
"""Refuse controlled timing when the selected CUDA GPU is in use."""

import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    command = ['nvidia-smi', f'--id={args.gpu}', '--query-compute-apps=pid,process_name,used_memory',
               '--format=csv,noheader,nounits']
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    processes = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if processes:
        raise SystemExit('Timing refused: target GPU is not exclusive:\n' + '\n'.join(processes))
    print(f'Preflight PASS: GPU {args.gpu} has no compute processes.')


if __name__ == '__main__':
    main()
