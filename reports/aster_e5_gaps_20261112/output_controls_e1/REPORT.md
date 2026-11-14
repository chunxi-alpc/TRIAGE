# E1 endpoint output controls

Completed CPU-only fits and cached historical evaluation.

Training: E1 Dataset A, 2540 pairs. Shared endpoints receive summed half-pair weights; weights preserve E1 category/source balance. Native pointwise binary NLL only. No pairwise invariance/evidence loss.

Temperature, endpoint intercept and affine calibration refit the existing output objectives. The independent intercept fitted on 761 original natural nodes is a separate existing result and is not rerun.

The ordinary supervised linear readout uses the frozen model's fixed final attention layer (all 32 heads, 4096 dimensions), not ASTER-selected heads or ASTER subspaces. L2 candidates 0.001/0.01/0.1 are chosen by 199 natural development-node NLL and frozen before test residual loading. This is a native supervised control.

All 40 test sources and 806 natural paths have prior use. Results are historical evaluation, not independent confirmation. NLI4CT and MedPIC are not re-evaluated. Positive temperature preserves decision signs; assess its Brier/log loss.

main_table.csv gives common pair/path/probability metrics. all_source_bootstrap.csv gives 2000 source-cluster bootstrap 95% intervals and paired differences to raw. evaluation/{dev,test}/ contains complete scoring, pair, and traversed-path predictions. Frozen parameters and solver status are preserved.

Protocol difference: the historical ordinary readout was a base-score residual correction on PCA of ASTER-selected heads. The separately named linear control here is a standalone classifier on fixed last-attention-layer outputs (4097 parameters), not an exact refit of that historical feature pipeline. Final residual is absent for new E1 and path inputs; the original fixed last-layer attention outputs are complete.
