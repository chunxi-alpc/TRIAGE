# 审查标准及实际英文审查指令

以下保留实际使用的英文指令。v1 用于 60 对盲审试点；v2 在任何标签揭示前统一事实计数口径，试点按 v2 复核后纳入最终 600 对。类别定义及“多事实/重复证据/删除不等于否定”规则在主审阶段未再改变。

审查者只看到两侧完整叙述、实际共同候选和文本差异，不看到存储标签、F 标记或历史审核结论。候选名称本来就是实际模型输入的一部分；遮蔽的是其对应哪一侧的真值。每对由一名独立上下文中的 Astra xhigh 审查，不是三名重复评分，也不是临床专家审查。推理强度及模型是请求配置，后端精确版本和 Pro 服务标识未暴露。

# Frozen semantic rubric v1 (pilot)
Each blinded record has a fixed two-candidate diagnosis question and two narratives; no source kind, released labels, case_type, prior review, or parent conclusions are supplied. Candidates are sorted as in the actual prompt, not assigned to either side.
Read both FULL narratives. First write blind decisions and save them; only later obtain released labels for your reviewed records.
Assess SAME target and candidate space; enumerate changed clinical concepts in abstract terms (do not copy patient narratives into outputs). Multiple independent concepts => S3, even if jointly plausible. Multiple textual edits describing the same clinical fact count once. Formatting/paraphrase alone is not a changed clinical fact. Removing a statement is NOT affirmatively negating it.
Blind output per record: review_id, changed_facts (list of abstract clinical concepts), fact_count, change_kind, target_same, candidates_same, unrelated_change, left_choice (exact candidate or uncertain), right_choice, change_sufficient (yes/no/uncertain), provisional_category, reason, confidence (0..1), issues (list), needs_expert.
S1 requires a fixed criterion truth-value task semantically equivalent to MEGA F, single decisive fact, supported flip. A forced-choice differential diagnosis is not S1 merely because encoded as yes/no.
S2 requires exactly one decisive medical fact, fixed candidates/target, medically defensible reversal of relative support, and usable within-pair score direction. This is cross-task auxiliary evidence, not strict compatibility or confirmed score calibration.
S3 requires clear incompatibility (multiple facts; target/candidates change; no real evidence change; unsupported/reversed purported flip; or unresolvable scoring mismatch). Do not classify every diagnostic case S3 merely because the task differs: that is what S2 assesses.
S4 when the medical ranking cannot be determined sufficiently, evidence/label basis missing, or expert adjudication needed. A plausible weaker or stronger suspicion need not support a flip.
Do not equate model-response preservation with clinical correctness. AI review is not clinical-expert review.
After blind records saved, reveal labels, append label_check and final_category. A clear contradiction => S3; unresolved label support => S4; keep blind fields unchanged.
Public outputs contain abstract clinical reasoning only, no full narratives, age/sex/location combinations, or identifying clinical details. Review_id and pair_id are acceptable linkage metadata.

# Freeze after 60 blinded pilot records, v2
Freeze time recorded before label reveal. Apply these clarifications to pilot AND main.
1. Compare net clinical information in the full narratives, not the number of replaced lines. One newly introduced or entirely removed symptom episode with its intrinsic descriptors counts one event. A replacement of one existing attribute (e.g. pain location or cough character) counts one variable. Simultaneous changes in independent attributes or findings count separately. Do not count newly introduced chest pain plus its radiation twice merely because both describe the same new pain episode.
2. Different past diseases or symptoms that can coexist are separate facts: removal of disease/symptom A plus insertion of independent B is two, not a generic 'history changed' variable. Whether statements occupy one line is irrelevant.
3. A removed rash description block is one changed clinical-information group for conservative counting; retained references to rash/pain mean the rash is not established absent. Do not manufacture incompatibility by counting every descriptor in a removed block separately. If fact granularity alone changes eligibility and cannot be resolved, mark S4 and flag count ambiguity.
4. Generic symptoms being specified as an already-described symptom with the SAME provoking/relieving pattern is 0 new facts if reference is clear. If reference is unresolved, count uncertain and S4 unless another clear S3 failure exists.
5. Do not require definitive clinical confirmation or extra labs for every S2. The actual task is relative support between fixed candidates; a clear single-fact ranking reversal can qualify even without certainty of the final diagnosis. A mere change in suspicion without sufficient reversal is S4; a confidently unchanged preference or duplicated/paraphrased evidence is S3.
6. Explicit multi-fact incompatibility suffices for S3 even if disease ranking also remains uncertain; record both issues. S2 means a usable candidate for a separately marked within-task auxiliary constraint, not evidence that identical cross-task score budgets are calibrated.
7. Do not use empirical class rates or desired training size to adjust criteria. Keep the original pilot blind file; write revised pilot blind results to reviewer_N_pilot_blind_results_v2.jsonl. Save BEFORE revealing labels. Unchanged records can be retained unchanged. Record which fact counts or categories changed and why.
