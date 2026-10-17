# CARE module 2 follow-up — 2026-10-17

## Status

**Completed:** new paired analysis of archived clinical O1 predictions, with saved metrics, family bootstrap intervals, answer changes and an SVG chart.
**Server access restored on 2026-10-17.** Python compilation, all 8 unit tests and exact Python reproduction of the archived clinical analysis have passed. Both O2 and clinical real-model smokes have passed. Four independent jobs are now running on physical GPUs 0/1/2/3; full B1–B4/random/direct-bias/ablation/patching results remain pending and will be saved by the server launcher.

The initial session's host command interface failed before creating a process. After permissions/environment were restored, SSH connected to gpu-a100-01 and confirmed free 80 GiB A100 cards. Server files are isolated at `/home/gcx/care_module2_followup_20261017`, reusing the original artifact/model root. The server's GitHub SSH credential was unavailable, so source was transferred through the existing authenticated connector and SSH; GitHub updates use the connector.

This work does not repeat the O2 head-selection statistical audit in PR #2.

## New analysis of existing clinical predictions

Input: `results/obs1_clinical/predictions.csv`, Git blob `359cae49be885750baa2f64ac120093fb3b603f3`, at source commit `8c5593e3d9e1a1bc9295bed916cb3e542198c574`.
Only the 1,152 **test** records were used: 384 examples × three methods, 192 complete pairs and 12 clinical rule families. Each family was held out in its outer fold.
All three fold manifests were checked for train/dev/test family separation, test-case membership and agreement between selected and evaluated strengths.

| Method | Case accuracy | Both pair members correct | Change in pair accuracy; 95% CI |
| --- | ---: | ---: | ---: |
| Baseline | 61.72% | 37.50% (72/192) | reference |
| Official SEKA | 61.98% | 38.02% (73/192) | +0.52 pp [−1.04, +2.08] |
| Official PASTA | 61.46% | 38.02% (73/192) | +0.52 pp [−1.04, +2.08] |

Intervals use 10,000 paired clinical-family bootstrap samples, preserving all cases and both methods within a family. They are conditional on the archived fitted outer folds; projection/strength fitting is not repeated in each resample. The analysis was designed after the predictions existed and is exploratory.

For the **25 pairs with both members prior-eligible**, all three methods have **20.0%** both-correct accuracy (5/25); each intervention's paired change is zero. The 153 individually eligible examples are reported separately, without pretending that they form 153 complete pairs.

For the **176 stable closed-book agreement cases**, all three methods have **84.66%** contextual accuracy. This is an operational no-prior-disagreement subgroup, not a claim of clinical safety or of identical case-level behavior.

### Error analysis

SEKA changes **8/384** answers: **4 corrections**, **3 newly wrong answers**, and **1 wrong-to-wrong** change. PASTA changes **5/384** answers: **2 corrections**, **3 newly wrong answers**.
SEKA's wrong-to-wrong glycemia example stops selecting the highlighted inapplicable branch but still misses the gold answer. A lower misuse rate therefore does not automatically imply better decisions.
Both methods hurt `hypoglycemia-hypo_one-06-non`; SEKA corrects the adjacent `hypoglycemia-hypo_one-07-non` case. All changed IDs, before/after answers and family results are saved in the CSVs.

These results provide a concrete baseline for the new subspace experiments. They do not establish that any new B1–B4 method improves decisions.

![Saved archived-data analysis](archived_pair_accuracy.svg)

## Executable follow-up implementation

- `scripts/module2_subspaces.py`: B1 centered raw action-key PCA; B2 pinned official SEKA projectors; B3 uncentered paired-applicability-displacement PCA; B4 regularized generalized signal/background eigenspace; random orthogonal projectors.
- `scripts/launch_module2_multigpu.py`: four independent GPU jobs (O2 plus three clinical folds), server tests/smoke before dispatch, one startup check, logs and merged clinical analysis.
- `scripts/run_module2_followup.py`: fixed-model loading and shard hashes, train/dev-only projector and parameter selection, resumable per-case inference journal, matched direct bias, ablation, opposite-applicability and same-label donor patching.
- `scripts/analyze_module2_followup.py`: complete-run analysis, family bootstrap, pairs, margins, misuse, stable-agreement subgroup and answer transitions; PNG/SVG/PDF from saved outputs.
- `scripts/analyze_module2_archive.py`: standard-library Python reproducer for the new archived clinical analysis, with optional Matplotlib exports.
- `tests/test_module2_followup.py`: disturbance budget, hook cleanup, projected patch, rank limits, background suppression, same-rule magnitude matching, causal mask/GQA support and incomplete pairs. **8 tests passed on the server.**

### B1–B4 definitions and controls

All operators act at QK-normalized **pre-RoPE** action keys, without editing candidate-option tokens. B1 and B3 use span-mean keys, matching the archived O2 representation unit. B3 uses applicable minus inapplicable keys; its main amplification does **not** receive the test applicability label.

B2 calls the unmodified official projection builder at commit `aab9cfed80dc012fb9e83f8bdefb8675c33f40e7`, using its 90%-spectrum positive projector and complement negative projector, combined as `(Ppos + .25 Pneg)/2`. Its rank is whatever the official algorithm returns. It is **not silently truncated to rank 1/4/16**. The controlled comparison explicitly normalizes the key disturbance per example; this is a **calibrated use of the official projection**, not an exact rerun of native uncalibrated SEKA. Archived native SEKA results above remain separate.

B4 background pairs must share the exact training rule, gold label and applicability, with numeric change magnitude within 15% of the applicability-flipping pair. The control IDs and both distances are saved. A configuration without controls is marked blocked rather than replaced by B3. Existing O2 same-side changes are one unit versus a four-unit flip; new training-only same-label values at a four-unit distance are constructed and truly forwarded before B4 fitting. Clinical controls use the existing training-family values; missing coverage remains visible in the audit. This controls one numeric-magnitude confound and is not evidence that every lexical or contextual confound is removed.

Random seeds 42/43/44 are reported separately; no best random seed is selected. Each method is selected on development both-correct pair accuracy, then case accuracy, then smaller disturbance/rank. Strength zero is included, so a method can select no intervention.

Main interventions use unsigned amplification, without applicability/gold labels. All projected edits have the same per-example relative Frobenius key disturbance. Uniform attention-score bias uses the same layer, four GQA query heads, target tokens and causal query positions, with RMS calibrated to the projected edit's actual BF16-induced score differences. It is calibrated using baseline Q/K, without the gold label. The equivalence smoke check compares a **nonuniform QK-difference mask** with a single-layer key edit. It separately reports BF16 output differences, because low-precision kernel/addition rounding prevents an exact BF16 equivalence claim, and tests FP32 algebra with a common math attention backend at a 1e-4 final candidate-probability tolerance. FP32 is a small diagnostic only (approximately 16–20 GiB), with all original buffer precisions restored before the main BF16 experiment. The uniform RMS-matched mask remains the scientific comparator.

For orthogonal B1/B3/B4/random projectors, ablation removes the selected subspace and patching replaces its coordinates using the paired patient's baseline action keys. B2 is a weighted sum of projectors, so its diagnostic modes instead remove the weighted operator component or interpolate the weighted donor difference; they must not be described as exact orthogonal subspace deletion/replacement. Both opposite-applicability and same-label donors are run, with numeric distances saved. These donors and labels are **oracle mechanism diagnostics**, separate from the deployable main decision comparison. Same-label clinical donor distances may be imperfectly matched; stratify by recorded distance before interpreting causal effects.

### Splits and scope

Clinical runs reuse the exact archived three outer-fold family manifests: six training families, two development families and four test families per fold. All training examples of those six families are used for every new projector. This gives more training examples to B2 than the original clinical O1 first-two-pair builder; the new comparison is internally matched and is reported separately from archived O1.

O2 reuses the verified original token-key tensor and official projections for the exact 36 training identifier groups, with 12 dev and 12 test groups. O2 contains one fictional grammar; its held-out groups **do not demonstrate unseen-rule-family generalization**. Clinical family-held-out performance is the relevant primary generalization evaluation.

Model: only `Qwen/Qwen3-4B-Base`, revision `906bfd4b4dc7f14ee4320094d8b41684abff8539`. Every safetensors shard must match the archived official hash; no model substitution or random weights. Clinical rules are source-grounded, patients simulated, and this work makes no clinical efficacy claim.

## Server execution

Before starting, report current GPU availability and the resource estimate to the user. Use four free authorized physical GPUs, preferably 0/1/2/3; substitute 6/7 for occupied cards. Do not overwrite existing experiments.

Estimated full default grid: four fixed layer/KV-head locations (2/0, 20/5, 9/0, 35/0), ranks 1/4/16, three random seeds, four strengths including zero. Up to 76 candidate operators per fold; approximately **65,000 model forwards** across O2 and three clinical folds, depending on empirical rank/control availability. Estimated **10–14 GiB peak per GPU**, recommend **24 GiB per card**, approximately **1–2 hours on four suitable GPUs** (or 2–6 hours serially). The user requested multi-GPU acceleration. O2 and the three clinical outer folds are independent jobs, so each card runs one complete model without cross-GPU model communication. These are planning estimates, not observed use. The archived pilot peak was about 7.63 GiB; full runtime depends on actual hardware and prompt length.

Recommended four-GPU launch from the new branch's experiment directory, after activating the existing environment and preparing the pinned external/SEKA checkout:

```bash
source /home/gcx/gcx261014/bin/activate
nohup env OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false python -u scripts/launch_module2_multigpu.py --output results/module2_followup/parallel > work_module2_launcher.log 2>&1 &
```

The launcher selects the first four free cards from 0/1/2/3/6/7, skipping cards with running compute processes or less than 18,000 MiB free memory. It never terminates other users' jobs. It runs tests, archive reproduction and two real-model smokes first, then launches all four workers concurrently. One check after 30 seconds records startup health; the launcher then waits for process completion without continuing GPU polling. Clinical results are pooled only after all three folds finish, with matching data/model/source fingerprints and no duplicated folds/cases. Outputs remain separate per job.

Equivalent serial commands (also useful for a failed-job retry):

```bash
source /home/gcx/gcx261014/bin/activate
# If external/SEKA is absent in this checkout:
# git clone https://github.com/waylonli/SEKA.git external/SEKA
# git -C external/SEKA checkout aab9cfed80dc012fb9e83f8bdefb8675c33f40e7
# Existing external/SEKA must have the exact pinned commit and no modifications.
python -m pytest -q tests/test_module2_followup.py
python scripts/analyze_module2_archive.py --verify
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_module2_followup.py --dataset o2 --output results/module2_followup/o2_smoke --smoke
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_module2_followup.py --dataset clinical --output results/module2_followup/clinical_smoke --smoke
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_module2_followup.py --dataset o2 --output results/module2_followup/o2
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_module2_followup.py --dataset clinical --output results/module2_followup/clinical
python scripts/analyze_module2_followup.py --run results/module2_followup/o2
python scripts/analyze_module2_followup.py --run results/module2_followup/clinical
```

Default artifact/model root: `/home/gcx/care_20261016/CARE_Codex_Pilot`. A cloned branch may reference that existing root via `--artifacts`; original data/results are read-only. The launcher preflight archive verifier reads the committed clinical predictions in the branch checkout; its optional Matplotlib export can refresh only this follow-up analysis's chart. Install/clone the pinned SEKA under the new branch experiment's `external/SEKA` if needed. Use distinct output folders for smoke and full runs. Journals resume only with an identical data/model/source/config fingerprint. Before O2 cache reuse, 18 examples spanning train/dev/test groups are re-extracted and checked against the archived keys; a mismatch stops rather than mixing representations. Keep detailed logs in the experiment output directory; after stable launch, stop continuous monitoring as requested.

## Current server run

- Root: `/home/gcx/care_module2_followup_20261017`.
- Active outputs: `results/module2_followup/parallel_v3` (earlier failed validation attempts preserved under `parallel` and `parallel_v2`).
- GPU 0: O2; GPU 1/2/3: clinical outer folds 0/1/2.
- Validation: `server_validation.json`. FP32 key-versus-score output errors are 1.13e-6 (O2) and 1.34e-6 (clinical), below 1e-4; peak diagnostic allocation is 15.83 GiB. BF16 differences (0.02335 and 0.00508) remain explicitly reported without an equivalence claim.
- A realized key edit is required in smoke; a favorable downstream decision change is not required. The first clinical diagnostic has a nonzero key change and zero BF16 candidate-probability change, which is retained as a valid null effect.
- O2 token tensors are loaded once, avoiding repeated decompression inside the case/head loop.
- All formal outputs come from BF16 inference after original rotary-buffer dtypes are restored. Main-run memory statistics reset after the FP32 diagnostic.
- Stop continuous monitoring after the four workers are observed to produce development journals; the launcher waits for completion and generates analyses without recurring GPU polling.

## Interpretation limits

1. Python compilation, 8 unit tests, exact archive reproduction and both real-model smokes have passed on the server. Keep the PR a draft while the complete B1–B4 model comparison is running.
2. All model shard and O2 tensor hashes passed. Re-extraction of 18 O2 examples spanning train/dev/test exactly matched the cached keys (maximum absolute error 0).
3. Single-location intervention results cannot be inferred from probe AUC. No new causal improvement is claimed.
4. Frozen clinical test predictions already existed; the next comparisons remain exploratory despite family isolation.
5. Three clinical outer folds share fitted training families; family-bootstrap intervals condition on fitted models and do not include model-selection uncertainty.
6. Archived clinical statistics and their regenerated PNG/SVG/PDF are secondary analyses of real saved predictions. GPU validation is now real; the full B1–B4 decision-effect table and new-intervention figures remain pending.
