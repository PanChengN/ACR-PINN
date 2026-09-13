#!/usr/bin/env python
"""Select the representative-baseline Fourier scale from completed scale-selection outputs."""

import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TAGS = {
    'burgers': {1.0: 'baseline_ff_burgers_scale1', 5.0: 'baseline_ff_burgers_scale5'},
    'helmholtz_4_4': {1.0: 'baseline_ff_h44_scale1', 5.0: 'baseline_ff_h44_scale5'},
    'klein_gordon': {1.0: 'baseline_ff_kg_scale1', 5.0: 'baseline_ff_kg_scale5'},
}


def result_directory(root, problem, tag):
    if problem == 'burgers':
        return root / f'results_{tag}'
    if problem == 'helmholtz_4_4':
        return root / f'results_helmholtz_a1_4.0_a2_4.0_{tag}'
    if problem == 'klein_gordon':
        return root / f'results_klein_gordon_{tag}'
    raise ValueError(f'Unknown scale-selection problem: {problem}')


def read_final_l2(root, problem, tag):
    path = result_directory(root, problem, tag) / 'aggregate' / 'logs' / 'summary.csv'
    if not path.exists():
        raise FileNotFoundError(f'Missing completed scale-selection result: {path}')
    with path.open(newline='', encoding='utf-8') as handle:
        row = next(csv.DictReader(handle))
    return float(row['final_l2']), str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=REPO_ROOT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--problems', nargs='+', choices=tuple(TAGS),
                        default=('burgers', 'helmholtz_4_4'))
    args = parser.parse_args()

    rows, rank_totals = [], {1.0: 0, 5.0: 0}
    for problem in args.problems:
        tags = TAGS[problem]
        values = {scale: read_final_l2(args.root, problem, tag) for scale, tag in tags.items()}
        ordered = sorted(values, key=lambda scale: (values[scale][0], scale))
        ranks = {scale: index + 1 for index, scale in enumerate(ordered)}
        for scale in ranks:
            rank_totals[scale] += ranks[scale]
            rows.append({
                'problem': problem, 'scale': scale, 'final_l2': values[scale][0],
                'rank': ranks[scale], 'summary_path': values[scale][1],
            })
    selected_scale = min(rank_totals, key=lambda scale: (rank_totals[scale], scale))
    payload = {
        'selection_protocol': f'mean_rank_across_{"_and_".join(args.problems)}; ties_select_smaller_scale',
        'metric': 'fixed-final-iteration relative L2 from single held-out selection seed 18018',
        'rank_totals': rank_totals,
        'selected_scale': selected_scale,
        'rows': rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
