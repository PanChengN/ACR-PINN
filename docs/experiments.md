# Experiment map

All commands are run from the repository root. Configuration files describe
training settings, while files ending in `_manifest.json` enumerate the
conditions to execute. Running `experiments/run_manifest.py` without
`--execute` validates a manifest without launching training.

| Study | Configuration and manifest | Analysis utilities |
|---|---|---|
| Seven-benchmark core comparison | `burgers.json`, `core_benchmarks.json`, `helmholtz_4_4.json`, `poisson_5d.json`, `schrodinger.json` and their manifests | `summarize_paired_results.py` |
| LDA architectural controls and gate behavior | `lda_ablation.json`, `lda_ablation_manifest.json` | `analyze_gate_behavior.py` |
| Extended-time and coordinate-encoding comparison | `spatiotemporal.json`, `spatiotemporal_manifest.json` | `analyze_extended_time.py`, `analyze_gate_allocation.py` |
| Task-gradient strategies | `optimizer_comparison.json`, `optimizer_comparison_manifest.json` | `audit_optimizer_results.py`, `analyze_optimizer_comparison.py` |
| Representative PINN baselines | `representative_baselines.json`, `representative_baselines_manifest.json`, `fixed_core_manifest.json` | `audit_baseline_results.py`, `analyze_representative_baselines.py` |
| Fourier-feature scale selection | `fourier_scale_selection.json`, `fourier_scale_selection_manifest.json` | `analyze_fourier_scale.py` |
| Runtime, memory, and equal-time accuracy | `computational_cost.json`, `computational_cost_manifest.json`, `equal_time.json` | `analyze_computational_cost.py`, `analyze_equal_time.py` |
| Sampling sensitivity | `sampling_sensitivity.json`, `sampling_sensitivity_manifest.json` | `analyze_sampling_sensitivity.py` |

The core study uses deterministic, iteration-wise Latin-hypercube resampling.
The representative-baseline study uses fixed collocation points because the RBA
and BRDR methods retain pointwise residual histories. Generated `results_*`
directories are excluded from version control.
