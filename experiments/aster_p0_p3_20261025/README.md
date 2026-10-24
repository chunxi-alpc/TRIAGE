# ASTER P0–P3 continuation (2026-10-25)

Base: PR #30, `2a4054015b2fff80242ab3a8c23d1c94a2bc75bd`. This directory preserves all old predictions and registrations. It is independent of the still-running legacy continuation.

## Reproduce on the research server

Activate `/home/gcx/gcx261014/bin/activate` in `/home/gcx/TRIAGE-worktrees/aster-p0-p3-20261025`. Private input: `/home/gcx/aster_e0_e8_20261024/private/cases.jsonl`. Model weights: `/home/gcx/TRIAGE/models/modelscope/{Qwen3-8B,Meta-Llama-3.1-8B-Instruct}`. Only GPUs 0 and 1 are used.

1. Run `python -m unittest discover -s experiments/aster_p0_p3_20261025 -p test_analysis.py`.
2. Run `infer.py --name qwen --phase mismatch` on GPU0 and its `--name llama` counterpart on GPU1. Save logs privately. Each performs 200 real within-tree patient-mismatch cases. Exact-input condition reuse is recorded; blind mismatch uses NC only.
3. Run `analysis.py --freeze` after both mismatch statuses are complete. This performs P0/P1/P2 and creates `results/frozen.json` once. It refuses to replace a freeze. For an unregistered exploratory reanalysis, omit `--freeze`; do not run it after a published freeze because it would replace the selection report timestamps.
4. Run `bash experiments/aster_p0_p3_20261025/launch_test.sh`. It starts fresh test inference, waits without continuous polling, aggregates actual results and pushes this new branch after both workers exit. Failures remain failures; partial row counts are reported. Do not modify frozen analysis/inference code while this is running.
5. `human_audit.py` prepares two private, independent 150-row review forms under `/home/gcx/aster_p0_p3_20261025/human_audit/`. Human reviewers must fill them independently, followed by adjudication. Rerun it to summarize returned reviews; it never fabricates verdicts or overwrites forms.

Scripts run on the server. Public files contain no patient text, rule text or option text. Source files are deployment-bound to the research server paths above; change constants only in a new experiment directory before registration.

## Statistical design and interpretation

P0 uses recorded real development probabilities for post-hoc temperatures .25/.5/1/2/4, uniform option/rule and action-frequency baselines, sum/max/mean mapping, exact condition/action deduplication and one-rule duplication sensitivity. Post-hoc computations are not new GPU measurements. Ties use first option; expected accuracy under uniformly random ties is also reported. Temperatures compare distribution sharpness, not stochastic decoding.

P1 joins records by patient, seed and variant, reports all four correction/harm cells, denominators, confidence, tree and variant strata. Development is 200 cases x 4 variants x 2 models (seed0). Multiple versions of one patient are not independent observations.

P2 uses three-way condition support/contradiction/insufficiency scores, competing-action support and optional blind agreement. This is a model-assessed condition-evidence constraint, not a verified text-extraction method. Candidate selection is leave-one-development-tree-out, with the entire held-out tree excluded across both models. Configuration selection maximizes model/tree macro accuracy, with harm then coverage as tie breakers. The unique final evidence candidate and a simple-threshold comparator are fitted on all development trees before new test inference. Test labels and variant identifiers are absent from the deployed selector API. The four-tree OOF result is exploratory and cannot establish zero population harm.

P3: 1000 independent-tree test cases x 4 core variants x 3 data seeds, plus 48 cases x 2 CS variants x 3 seeds, for each of 2 models. The three seeds vary data construction, not trained model weights or sampling. Deterministic inference is used. Confidence intervals resample whole trees 10,000 times, preserving all patients/versions/seeds; models are reported separately. Report core and CS results separately. The fresh baseline is reused as fallback within a row; this does not count as extra inference. Exact-input condition reuse across variants/seeds is explicit.

Primary comparison: frozen evidence candidate vs fresh no-intervention, with accuracy difference and harm rate; compare against simple threshold and direct condition/old blind mapping. Existing crop/gate comparisons remain development reanalyses. The selected rule is unique even when its development performance does not beat all comparators. No post-test winner switching.

Synthetic unit tests exercise mathematical/control properties only. They are never included in empirical metrics. Workload reports distinguish wall time, peak allocated memory, explicit model-call units and reused inputs; concurrent jobs prevent interpreting allocated GPU hours as exclusive utilization.

## Current evidence

See `results/报告.zh.md`, `legacy_audit.json`, `provenance.json`, `p0_controls.csv`, `p0_mismatch.json`, `p1_cases.jsonl.gz`, `p1_strata.csv`, `p2_cross_validation.json`, and `frozen.json` when present. P3 completion is governed by exact counts in `p3_status.json`, never by merely finding an output file. Human review remains separately pending until genuine reviewer input is received.

FlexAttention has earlier numerical checks, not an efficiency advantage. SEKA/GRPO are not added: no evidence yet justifies their additional inference/training cost for this question. This is a scope decision, not an empirical claim that those methods cannot work.

Public `development_inputs.jsonl.gz` / `test_inputs.jsonl.gz` contain action identities as integers and condition hashes, sufficient to reproduce aggregation and selection without raw text. Run `python experiments/aster_p0_p3_20261025/reanalyze_public.py experiments/aster_p0_p3_20261025/results/development_inputs.jsonl.gz` for an independent stdlib P1 reproduction. `p3_pairwise.json` reports frozen candidate comparisons to no-intervention, simple threshold and direct mapping separately for core and each CS variant.
