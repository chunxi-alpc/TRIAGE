# CARE module 2: archived O2 head-selection reanalysis (2026-10-17)

## Status and scope

**COMPLETED: secondary statistical reanalysis** of the archived Qwen3-4B-Base *fictional-rule* O2 experiment, across all **36 layers x 8 KV heads = 288 layer/head locations**. **NOT RUN: new GPU inference, B1–B4 projected-key intervention, activation patching, clinical generalization**. These are distinct tasks; this README does not claim they were executed.

Input: ../obs2/layer_head_stats.csv (Git blob 5f56624a14420678291bfa009e23f76177098a8d).
Saved machine-readable results: head_selection.json.
Reproducer: ../../scripts/reanalyze_o2_head_selection.py.

## Results from the archived O2 table

| Quantity | Reanalysis |
| --- | ---: |
| Layer × KV-head positions | 288 |
| Applicability **test** AUROC ≥ 0.80 | 252/288 |
| Applicability **test** AUROC ≥ 0.90 | 196/288 |
| Archived test applicability P-value with FDR < 0.05 | 277/288 |
| Mean / median test applicability AUROC | 0.9053 / 0.9410 |
| Pearson r of applicability vs same-side numeric **shift magnitudes** | 0.9295 |
| Heads with positive archived normalized cross-threshold minus same-side statistic | 0/288 |
| Previously showcased head: layer 2, KV 0 | test AUROC 0.7378 |
| Rank of that head by archived training GroupKFold applicability AUROC | 269/288 |
| Highest archived **training-CV** applicability AUROC | layer 20 / KV head 5: CV 0.9952 |
| Test AUROC of the training-CV-selected layer 20 / KV 5 | 1.0000 |
| Mean test AUROC for the top 10 **training-CV-selected** heads | 0.9984 |

Layerwise mean test applicability AUROC: layers 0–8: **0.7663**; 9–17: **0.9224**; 18–26: **0.9853**; 27–35: **0.9473**.

**Important:** This ranking is descriptive. The repository already contained these test results when this analysis was designed; this is not a prospective preregistered confirmation. The CV-based selection itself does not use held-out labels, but the choice to analyse that rule and to emphasize it was made post hoc.

The selected head in the original O2 visualization (2 / 0) was chosen by **development relevance-displacement consistency**, not applicability. A head selected for relevance is not necessarily best for applicability; selecting solely by probe AUC likewise does **not** locate an effective intervention position.

The archived per-unit cross-threshold/same-side diagnostic is non-positive at all 288 locations. This is a specific normalization-dependent observation, not a proof that applicability is absent or that threshold crossing has no causal effect.

The very high decoding scores are compatible with widespread numeric/lexical signals and do not identify an independent applicability mechanism. In particular, the sizeable correlation of ordinary numeric and applicability displacement magnitudes warns against naïve PCA-on-difference interpretation.

## Next experimental question (requires GPU; not executed here)

**Do the representation directions selected for applicability actually improve paired rule decisions on held-out rule families, relative to equally strong relevance-only SEKA, generic PCA, random projectors, and a direct attention-logit-bias control?**

Recommended procedure:

1. Recover archived results/obs2/key_embeddings.npz from the complete existing release/server snapshot; build training-only rank-1/2/4/8/16 projectors for (B1) raw-key PCA, (B2) official SEKA, (B3) paired applicability-displacement PCA and (B4) applicability-contrastive background-whitened space. Report B4 separately until matched-magnitude nonflipping controls have been collected.
2. Hold out entire rule groups. Select heads/ranks/strengths using training/dev only. Include original layer2/KV0, layer20/KV5, random heads and heads in different depth regions. Avoid selecting heads from test performance.
3. Run actual key edits and matched direct-logit-bias controls in the **same model** under fixed input spans and similar disturbance budget. Report **paired accuracy**, margin changes, inapplicable-branch misuse and no-conflict retention.
4. Transfer to clinical rule families using the existing 12-family clinical O1 protocol. Keep the 153 closed-book-disagreement cases as an independent subgroup, not as the sole test population.
5. A representation-probe AUROC of 1.0 is not equivalent to successful steering. Require actual intervention improvements before claiming Module 2 learned an actionable applicability subspace.

## Reproduce the completed reanalysis

From repo root:

    python experiments/care_qwen3_4b_base_20261016/scripts/reanalyze_o2_head_selection.py

This operates only on the committed CSV, needs no GPU and does not modify the archived O2 results. It recreates head_selection.json (adding an SHA256 digest of the local CSV).

## Provenance

- Archived O2 full model execution and statistical pipeline: scripts/run_experiments.py and scripts/analyze_results.py.
- Model: Qwen/Qwen3-4B-Base, 36 layers, 8 KV heads, head_dim=128.
- Original O2 data: 60 fictional rule groups, 36 train / 12 dev / 12 test; not independently verified clinical content.
- Rule family is the sampling unit, not each head or token; these head counts are **not** independent significance tests.
- Previous Qwen3-8B TRIAGE src/module2/localizer.py used 12 query-head candidates on two cached rule contrasts. Its output cannot be directly pooled with this Qwen3-4B experiment.
