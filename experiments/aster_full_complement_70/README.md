# Issue #70: full evidence complement

Independent entry point; no changes to the default ASTER method. Uses the archived
Qwen3-8B E1 cache and source folds. All computation runs on the experiment server.

```bash
source /home/gcx/gcx261014/bin/activate
pip install osqp
export CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
python -m unittest experiments.aster_full_complement_70.test_solver
python -m experiments.aster_full_complement_70.run --out reports/aster_full_complement_70
```

`--cache-root` relocates the original report root; cached feature references must
remain available. Outputs are independent; completed model/evaluation stages can
resume from their `done.json` files. Use a fresh output directory for a different
configuration. `REPORT.zh.md` is generated from completed measured results.

The 3-point lambda grid and evidence-first selection rule are fixed in `solver.py`
and `run.py`. Inner M3 fit/development sources are disjoint and do not contain outer
held-out sources. Archived M1/M2 are conditioned on all outer training sources,
as required by the task. All attempted weights and every CF constraint are saved.

The solver uses a full thin QR of weighted AP differences to apply the ridge
inverse, and a signed CF dual QP. No AP singular directions are discarded, and
no feature-sized Hessian is allocated. `model.npz` contains all 6,528 coefficients.
Source bootstrap intervals concatenate the five held-out folds before computing
VR and ER. Archived E1 results are read from saved predictions, never re-scored.
