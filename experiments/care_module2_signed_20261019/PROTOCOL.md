# CARE module two: signed applicability diagnostics

This experiment was run on 2026-10-19 on gcx@10.29.128.115, using the existing gcx261014 environment. It extends PR #3 without changing its branch or outputs. All patients are simulated; rule text and executable answers are unchanged from the archived clinical-rule experiment. This is an exploratory follow-up designed after the PR #3 results were known.

## Scope and formulas

Fixed model: Qwen/Qwen3-4B-Base, revision 906bfd4b4dc7f14ee4320094d8b41684abff8539, BF16, SDPA, 36 layers / 32 query heads / 8 KV heads / dimension 128. Every weight-shard hash must match the original manifest. No model weight is trained.

Let P=UUᵀ be the original B4 contrastive applicability projector, k a QK-normalized pre-RoPE action key, and s=+1 for gold-applicable cases and −1 otherwise. Compare:

1. Baseline: unchanged model.
2. B4_positive: k'=k+aPk.
3. B4_signed: k'=k+a·sPk.
4. d_app_signed: d_app=mean_train_pairs(mean_action_tokens(k_app)−mean_action_tokens(k_nonapp)); k'=k+a·s·d_app.
5. score_bias_signed: add s·tau directly to the same causal attention-score entries.

The original B4 is a positive semidefinite orthogonal projector, followed by an additive positive coefficient. It is not coordinate clipping or a one-sided scalar projection. Its original online key-edit function has no applicability argument and does not read gold labels. B4's covariance construction discards the global sign of the applicability displacements; changing all training difference signs leaves P unchanged.

Gold applicability is available only as an experimental oracle. It is passed to interventions, never added to prompt text or answer options. Mean directions and projectors use training families only.

## Fixed design and selection

Use the exact 12 clinical families / 384 cases / 192 pairs and three archived family splits. Each outer fold has six training families, two development families and four test families. Pairs and all variants of a family stay together. Each family is tested once.

For each fold, freeze PR #3's B4 development-selected layer, KV head and rank. Refitting from freshly extracted training keys must recover the archived projector within 1e-6. Four methods share that location, the original action span, and its four associated GQA query heads. This is a controlled location comparison, not a claim about globally optimal heads or independently optimized subspaces.

Predeclared target score RMS: tau in {0, .01, .05, .1}. For each input and key intervention, a label-free scalar search calibrates the actual fixed-query QK-score difference after exactly one RoPE and BF16 key/rotation rounding. The support includes all causal query positions for every action token at the selected four query heads, including zero-valued score changes. This is a local score-magnitude match; it does not match downstream changes or the sign distribution of individual scores.

Direct bias is stored in the model's BF16 attention-mask dtype; actual values are 0.010009765625, 0.050048828125 and 0.10009765625. Key calibration retains the closest value found by scalar bisection in the discrete BF16 path. The run's largest relative RMS mismatch is 7.29% at tau=.01; 95th-percentile mismatches are at most 2.31% at tau=.01, 0.55% at tau=.05 and 0.38% at tau=.1. All per-input and aggregate discrepancies are saved. These are approximate, measured matches rather than exact equivalence.

A common tau is selected for all four methods within each fold: maximize mean development paired accuracy across methods, then mean development case accuracy, then prefer smaller tau. The final common strengths are .01, .1, .1 for folds 0/1/2. Selection files are written before test inference. Development-only method-specific selections are secondary. The entire predeclared test-strength grid is reported without choosing a best test value.

Every main forward uses an explicit causal mask, batch 1, no KV cache, identical prompt and controlled single-token A/B/C/D candidate scores. No random weight substitute or generated clinical answer is used.

## Audits and checks

- Recompute all 384 gold labels using the original independent numerical rule oracle; check applicability against the highlighted action label.
- Check every character span and token boundary, unchanged paired action tokens and option maps, and that edits precede the answer options.
- Freshly extract all training keys for the chosen head and compare exactly against PR #3's cache.
- Independently on the fourth GPU, re-extract all 384 cases at all four archived locations and compare every action-key value; baseline candidate probabilities and original B4 candidate probabilities reproduce exactly.
- Before each key edit, assert that the live unedited key equals the captured baseline key. Hooks are scoped, called once, and removed after every forward.
- Verify zero score bias reproduces baseline exactly.
- Numerical CPU tests check post-RoPE score differences against explicit before/after QK products, scalar calibration, orthogonal-projector properties, and the B4 sign ambiguity. These unit-test tensors are not experimental results.

The first startup attempt encountered an attention-mask dtype incompatibility in the direct-bias path and failed. No partial output was promoted to a result. It was corrected before the completed run; failed attempt logs remain under the server work directory. All four completed workers exit with code zero.

## Metrics and uncertainty

Pair accuracy requires both applicable and nonapplicable members to be correct. Also report case accuracy, app/nonapp accuracy, misuse of the highlighted inapplicable action, gold logit margin, gold candidate-probability margin, and every clinical family.

Gold logit margin = gold candidate logit − largest incorrect candidate logit. It is distinct from the probability margin.

The no-conflict set is the original 176 cases where both closedbook formulations agree with gold. No-conflict accuracy uses all 176 cases. Retention is accuracy among its 149 cases that the contextual baseline already answers correctly. Baseline retention is therefore 100% by construction; all denominators are saved. The 153 originally screened conflict-eligible cases and 25 complete eligible pairs are separately reported.

Use 10,000 shared family bootstrap draws, seed 261019, preserving case/method pairs within each sampled family. Save absolute CIs, paired changes against baseline, all ten method-pair comparisons, and family results. Intervals condition on the already fitted outer folds and do not include refitting variability. There are only 12 families. Sparse or zero-width empirical intervals do not establish population equivalence. Secondary comparisons have no simultaneous multiplicity correction.

## Completed computation and stored data

GPU 0/1/2 ran clinical folds; GPU 3 independently validated the old run. Main inference took about 94–96 seconds per fold, including training re-extraction and model provenance checks. Peak main allocated memory was 7.59 GiB or less; peak reserved memory was 7.60 GiB or less. Exact status, package versions, hashes and per-job times are in results.

There are 9,792 main journal records: 576 baseline records and 9,216 method-strength records, including zero-strength baseline reuse. This corresponds to 8,064 main model forwards, including 576 fresh training-key forwards. The independent validation uses 769 additional forwards. These are repeated computations on 384 cases, not thousands of independent clinical cases.

The only GPU startup resource check is followed by a bounded startup check. The launcher then waits for process exits and does not continuously monitor GPU resources.

## Reproduction

Run on the same server:

    ssh gcx@10.29.128.115
    source /home/gcx/gcx261014/bin/activate
    cd /home/gcx/TRIAGE-module2-diagnostic/experiments/care_module2_signed_20261019
    python -m pytest -q test_diagnostic.py
    python analyze.py --run results --output work/replayed_analysis

For a new full run, launch.py expects its results directory to be absent or empty, and requires four free authorized GPUs 0–3. Report the resource plan before restarting, as required by the user's instructions. It uses the existing fixed model and source data at /home/gcx/care_20261016/CARE_Codex_Pilot, plus PR #3's full cache at /home/gcx/care_module2_followup_20261017/results/module2_followup/parallel_v3.

For another checkout, diagnostic.py accepts --artifacts, --archive, --model and --output. Its support.py freezes the required PR #3 training functions and independent oracle; pr3_source_hashes.json records their original complete source hashes. To recover the full PR #3 cache, extract /home/gcx/care_module2_followup_20261017/exports/CARE_module2_full_results_20261017.tar.gz, SHA256 cce01cd680ea046754ca9fea9755d3154ba3c140c58844e04f5a3459de49f544. The large original key tensors are not duplicated in this PR. There is no fallback to a different model or invented cache.

The tracked journals are gzip-compressed after completion. Run materialize_journals.py before analysis or a full-archive replay if uncompressed prediction files are needed. The model and GPU are not required to recompute statistics and figures from saved journals.

PROTOCOL.md documents the implemented completed design. The machine-readable plan with TAUS/methods was saved before dispatch, and each provenance file hashes the exact inference source. Any follow-up protocol changes require a separately labeled run.

