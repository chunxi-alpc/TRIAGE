# M3 Pareto and knee analysis (CPU only)

This independent analysis reads the 70 completed OOF configurations from
[Draft PR #76](https://github.com/chunxi-alpc/TRIAGE/pull/76).
It produces three figures (PNG/PDF/SVG), six key-point comparisons, existing
bootstrap intervals, selection metadata and a Chinese report.

All selection uses the 80-source OOF grid. The old E1 reference and the
20-source dev / 40-source reused holdout results enter only after selection.
Missing evaluations remain empty in CSVs with status `not_evaluated`.
No model packages, training, M3 refits, predictions or new bootstrap draws are used.

## Reproduce on the server

Requirements: the existing `gcx261014` Python environment with matplotlib.
The main numerical analysis uses the Python standard library.

From the repository root:

```bash
source /home/gcx/gcx261014/bin/activate
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export M3_PR76_SOURCE=/home/gcx/aster-m3-dual-objective-20261116/reports/aster_m3_dual_objective_20261116

python -m unittest discover -s experiments/aster_m3_pareto_knee_20261116/tests -v
python -m experiments.aster_m3_pareto_knee_20261116.analyze \
  --source-dir "$M3_PR76_SOURCE" \
  --output-dir reports/aster_m3_pareto_knee_20261116
```

If the PR76 checkout is elsewhere, set `M3_PR76_SOURCE` to its report directory.
To obtain the exact source in a separate checkout:

```bash
git fetch origin refs/pull/76/head
git worktree add --detach ../TRIAGE-pr76-results 2f1849037223fb785056c19dab364043c0d9738c
export M3_PR76_SOURCE=../TRIAGE-pr76-results/reports/aster_m3_dual_objective_20261116
```

The script reads these PR76 files:

- `oof_grid_70.csv`, `pareto_front.csv`, `selection.json`, `grid_bootstrap.csv`.
- `evaluation/{oof_selection,dev,test}/{metrics.csv,bootstrap.csv}`.

It writes only to the new output directory. Source/output directory overlap is rejected.

## Methods and tests

Dominance is exact: another point must have no larger distortion and no smaller
VR, with at least one strict improvement. Identical coordinates remain on the
front. Min-Max bounds come only from the recomputed Pareto front, excluding E1.

A maximizes absolute perpendicular distance to the endpoint chord.
B maximizes normalized VR minus normalized distortion. All real-data interior
points lie above the normalized chord, so A = B / sqrt(2).
Numerical ties (relative tolerance 1e-12, absolute tolerance 1e-14) prefer lower
distortion, then higher VR, then candidate ID. Degenerate fronts have no interior knee.

Tests cover strict/weak dominance, duplicate points, known geometry, scaling,
normalization scope, ties, degenerate fronts, an A/B disagreement example,
input immutability, source protection, the actual 70-point result, missing
evaluations, copying bootstrap intervals and immunity of selection to changed
dev/holdout metrics. Integration tests require `M3_PR76_SOURCE`.

The report distinguishes the eta=1 elbow region from the nearly tied lambda
settings. Existing intervals are marginal per-configuration intervals.
