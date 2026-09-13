# Experiment utilities

The scripts in this directory follow a verb--object naming convention:

- `run_*`: launch a configured condition or manifest.
- `audit_*`: verify result completeness and protocol consistency.
- `analyze_*`: aggregate metrics and perform statistical analyses.
- `plot_*`: generate manuscript-facing figures from completed results.
- `compute_*` and `aggregate_*`: derive supporting measurements.
- `benchmark_*` and `check_*`: measure or validate the execution environment.

See [`../docs/experiments.md`](../docs/experiments.md) for the mapping from each
study to its configuration, manifest, and analysis scripts.
