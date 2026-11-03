# Statistical protocol and independent audit plan

Protocol recorded before access to experiment outputs. This is an executable pilot design, not an assertion of completed results.

## Comparison and estimands

A: unmodified model. B: a fixed source-preference direction and coefficient from the paper, to the extent verified. C: the identical direction, layer, and maximum coefficient, with activation governed by CARE module-one applicability. D: original CARE applicability-key subspace only if its supplied specification and available data support faithful implementation. If D is an approximation, label it explicitly and do not call it the original method. B versus C estimates the benefit of the dynamic gate under the same intervention budget; A comparisons estimate total intervention effects. Report factual fidelity to the source separately from predictive performance.

Lock the model checkpoint, tokenizer, precision, prompt format, answer extraction, direction normalization, layer, intervention token positions, and generation settings across comparisons. Record all exceptions. Oracle gold applicability gates may appear only as an explicitly labelled diagnostic ceiling, separate from model-predicted CARE module one. A threshold used by module one must be fitted on train/dev, not test. Record gate raw output, parsed applicability, uncertainty/abstention, latency, and final coefficient.

## Data design and independence

Split semantic rule families before generating patient variants and prompt paraphrases. All members of a minimal flip pair, all paraphrases, and all shared-rule-derived cases remain within one split. Prefer at least 20 independent test families for descriptive cluster intervals; if fewer, still report intervals with the limitation prominently. Never treat two sides of a pair as independent observations.

Primary test strata:

1. A single conflicting new rule is applicable; the old rule is valid or invalid as explicitly recorded.
2. The minimal patient-condition flip makes that new rule inapplicable; the appropriate valid old rule must be used.
3. Multiple new rules apply, requiring jointly correct choices; report exact-set correctness and individual rule-level results.
4. An exception overrides a general rule; include both sides of the exception condition.
5. The old rule remains valid, versus a case in which it is explicitly invalid; correct fallback or abstention must be specified in labels.
6. No conflict: agreement between sources and, separately, irrelevant new-rule context.
7. Cross-task transfer: train the direction only on the source task; freeze layer, coefficient, gate threshold, and prompt-selection decisions before a distinct target-task test.

Separate semantic-transfer and mere prompt-paraphrase tests. Report direction-transfer-only when the gate itself uses target-task labels. If old rules are supplied in the prompt rather than verified parametric beliefs, label the experiment as in-context rule conflict; it does not by itself establish parametric-versus-context knowledge conflict performance. Avoid interpreting fabricated clinical rules as actual clinical recommendations. Label synthetic tasks and provide their source/specification, generation script, and validator.

Run exact-content hashing and canonical semantic-family checks across train/dev/test. Inspect near duplicates, answer-label imbalance, and condition-to-answer shortcuts. Randomize presentation order using a recorded seed, preserving paired identities. Prefer rotating answer codes across independent cases and balancing gold classes. If answer-code randomization occurs, preserve mappings in data and ensure it does not differ between A/B/C/D for the same item.

## Metrics and denominators

- Accuracy: mean(final parsed answer == gold), over all test items and separately within each predeclared stratum. Invalid/unparseable answers count as incorrect; report their rate.
- New-rule misuse rate: count(items selecting a prohibited new-rule action) / count(items where that new rule is inapplicable and has an identifiable prohibited action). Do not infer misuse from every incorrect prediction. Provide explicit forbidden_new_answers or chosen_rule_ids.
- Minimal-pair joint correctness: proportion of complete flip pairs with both answers correct. Also report applicable-side and inapplicable-side accuracy.
- Multi-rule exact accuracy: the complete required action/rule set is correct, with no forbidden additions. Rule-level precision/recall/F1 may supplement it, with empty-set conventions recorded.
- Old-rule fallback accuracy: proportion correctly using the still-valid old rule among cases where no overriding new rule applies and old_valid=true. Also report invalid-old-rule misuse on old_valid=false cases.
- No-conflict accuracy and intervention delta versus A; absence of a statistically significant loss is not evidence of equivalence.
- Gate accuracy, false-apply rate, false-reject rate, and abstention rate, calculated independently of final answer. Gold gate diagnostics are not operational C results.
- Cross-task accuracy and all applicable mechanism metrics, with source/target tasks and all target-data use disclosed.
- Overhead: intervention setup/extraction seconds, generation wall seconds, gate seconds, total end-to-end seconds, input/output tokens, peak allocated and reserved GPU memory, and GPU hardware. Report both totals and per-item median/p95; distinguish synchronization-inclusive measured time from estimates, exclude or explicitly include warm-up and checkpoint loading. C must include gate inference cost. A separate gate model adds its memory and latency.

Each proportion must report its numerator and denominator. Missing methods or incomplete predictions must never silently reduce the comparison cohort. Report complete common-cohort estimates and a transparent inventory of missing rows; failure predictions count as incorrect in an intention-to-evaluate result if a method was actually attempted.

## Inference

The paired cluster is rule_family_id, falling back to pair_id only if families genuinely do not share a rule-generating unit. Bootstrap clusters with replacement, retaining every item and every method within a sampled cluster. Use 10,000 replicates and a fixed bootstrap seed, percentile 95% confidence intervals for each metric and each paired method difference. For simple pair-joint correctness use the same family-level bootstrap. Report number of independent clusters and bootstrap replicates; very few clusters make coverage unreliable.

Primary contrasts: C minus B for accuracy and C minus B for new-rule misuse (a negative misuse difference is favorable). Use paired cluster sign-permutation tests on within-cluster treatment differences; enumerate all sign assignments for up to 20 nonzero clusters, otherwise use 100,000 reproducible Monte Carlo sign permutations with (extreme+1)/(replicates+1). This test relies on exchangeability/symmetry of cluster differences and should be described as exploratory for small synthetic pilots. Exact McNemar tests on item pairs may be shown only as secondary, with the within-family dependence limitation; do not present naive item McNemar as the primary clustered inference.

Apply Holm adjustment across the two primary contrast p-values. All A contrasts, D comparisons, strata, and transfer results are exploratory unless separately predeclared before outcomes are opened. Show unadjusted and adjusted p-values. Do not select best seeds, tasks, layers, or coefficients on the test results.

For multiple genuine independently trained/extracted directions, report per-seed outcomes and seed mean/spread; bootstrap families within seed while retaining seed as an outer resampling level if claiming across-seed uncertainty. Repeated deterministic evaluation of the same checkpoint/direction is not an independent random seed experiment. A one-seed pilot must be identified as such.

## Minimal prediction schema (JSONL)

Required identifiers: run_id, method [A|B|C|D], model_id, model_revision, seed, split, item_id, pair_id, rule_family_id, task, condition.

Ground-truth metadata: applicable, old_valid, conflict, gold_answer, gold_rule_ids, forbidden_new_answers, forbidden_old_answers, source_id, source_version.

Outputs: raw_output, parsed_answer, selected_rule_ids, parse_valid, status, error_summary. Preserve raw decoded output without post-hoc repairs beyond the preregistered parser.

Intervention metadata: direction_id, direction_train_hash, layer, coefficient, gate_raw_output, gate_parsed, gate_confidence, gate_threshold, gate_model_id, gate_is_oracle, subspace_id.

Cost metadata: input_tokens, output_tokens, generation_seconds, gate_seconds, total_seconds, peak_allocated_bytes, peak_reserved_bytes, gpu_device, batch_size, warmup_included.

A canonical immutable dataset file provides shared labels. Prediction rows should carry dataset_sha256 and item_id so that copied labels cannot be altered separately across methods. Save a run manifest with full command, environment versions, hardware, checkpoint identity, all seeds, data hashes, code commit or source hashes, start/end times, and completion state.

## Independent review checklist

Verify source-method fidelity, upload access and faithful D implementation; train/dev/test isolation including templates and family grouping; no oracle leakage into primary C; B/C direction and strength matching; old-rule truth and action mappings; parser symmetry; complete method/item matrix; gate cost accounting; inference clustered at the true shared unit; output statements distinguish actual execution, partial runs, planned work, and unsupported claims. Review code before inspecting method outcomes when practical, and save every material limitation with its effect on interpretation.
