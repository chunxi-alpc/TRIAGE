# V8.1 错误分析

开发集冻结选择：A1。其余数据仅用于评价。

|数据|方案|正确规则|F→T|U 召回|反事实配对|调用|输入/输出 token|
|---|---|---:|---:|---:|---:|---:|---:|
|dev|A1|220/288|24/176|43.8%|12/36|96|213900/15198|
|dev|B1|216/288|22/176|31.2%|10/36|96|227340/15091|
|dev|B2|218/288|24/176|37.5%|12/36|192|461046/30391|
|confirmation|A1|450/576|38/352|26.6%|36/72|192|425322/30506|
|confirmation|B1|449/576|36/352|18.8%|39/72|192|452202/30467|
|confirmation|B2|446/576|34/352|25.0%|36/72|384|917246/61000|
|new72|A1|192/216|3/120|68.8%|20/24|72|154153/12193|
|new72|B1|192/216|3/120|68.8%|19/24|72|164233/12150|
|new72|B2|193/216|3/120|64.6%|22/24|144|334035/24357|
|natural72|A1|189/216|5/120|64.6%|18/24|72|154563/12615|
|natural72|B1|186/216|4/120|56.2%|19/24|72|164643/12685|
|natural72|B2|190/216|4/120|64.6%|19/24|144|335277/25226|

耗时为实际批次 GPU 生成耗时分摊到病例后的总和；包含语义判断、复核和格式重试，不含加载模型。完整逻辑成本包括 B2 复用的 A1 首次判断，本次新增成本单独保存。历史与本次耗时来自不同运行，仅作描述性对照。

## 错误分布

### dev_A1

错误类型：{"T→F": 26, "F→T": 24, "U→F": 9, "U→T": 9}

各病例组错误数：{"mechanism_10_entry_with_two_exclusions": 24, "mechanism_22_five_condition_composition": 21, "mechanism_04_conjunction_nested_disjunction": 13, "mechanism_16_three_alternatives": 10}

- mechanism_04_conjunction_nested_disjunction-p01-canonical-app / R_TARGET：T → F；模型理由：The patient lacks authorization, blocking the age-and-authorization combination.
- mechanism_04_conjunction_nested_disjunction-p01-unit_equivalent-app / R_TARGET：T → F；模型理由：The patient lacks authorization, and the tumor diameter exceeds 2.5 cm.
- mechanism_04_conjunction_nested_disjunction-p01-unit_equivalent-app / R_ALTERNATIVE：F → T；模型理由：The patient fails the age-and-authorization or tumor diameter condition.
- mechanism_04_conjunction_nested_disjunction-p02-canonical-app / R_TARGET：T → F；模型理由：The patient lacks authorization, blocking the age-and-authorization combination.
- mechanism_04_conjunction_nested_disjunction-p02-unit_equivalent-app / R_TARGET：T → F；模型理由：The patient lacks authorization, and the tumor diameter exceeds 2.5 cm.

### dev_B1

错误类型：{"T→F": 28, "F→T": 22, "U→F": 11, "U→T": 11}

各病例组错误数：{"mechanism_22_five_condition_composition": 26, "mechanism_10_entry_with_two_exclusions": 18, "mechanism_04_conjunction_nested_disjunction": 16, "mechanism_16_three_alternatives": 12}

- mechanism_04_conjunction_nested_disjunction-p01-canonical-app / R_TARGET：T → F；模型理由：The patient lacks authorization, blocking the route despite age and tumor size.
- mechanism_04_conjunction_nested_disjunction-p01-unit_equivalent-app / R_TARGET：T → F；模型理由：The patient lacks authorization, and the tumor diameter exceeds 2.5 cm.
- mechanism_04_conjunction_nested_disjunction-p01-unit_equivalent-app / R_ALTERNATIVE：F → T；模型理由：The patient fails the age-and-authorization or tumor diameter condition.
- mechanism_04_conjunction_nested_disjunction-p01-scope_and_order-app / R_TARGET：T → F；模型理由：The patient lacks authorization, blocking the age-and-authorization combination.
- mechanism_04_conjunction_nested_disjunction-p02-canonical-app / R_TARGET：T → F；模型理由：The patient lacks authorization, blocking the route despite age and tumor size.

### dev_B2

错误类型：{"T→F": 26, "F→T": 24, "U→F": 10, "U→T": 10}

各病例组错误数：{"mechanism_10_entry_with_two_exclusions": 24, "mechanism_22_five_condition_composition": 21, "mechanism_04_conjunction_nested_disjunction": 13, "mechanism_16_three_alternatives": 12}

- mechanism_04_conjunction_nested_disjunction-p01-canonical-app / R_TARGET：T → F；模型理由：The patient lacks authorization, blocking the age-and-authorization combination.
- mechanism_04_conjunction_nested_disjunction-p01-unit_equivalent-app / R_TARGET：T → F；模型理由：The patient lacks authorization, and the tumor diameter exceeds 2.5 cm.
- mechanism_04_conjunction_nested_disjunction-p01-unit_equivalent-app / R_ALTERNATIVE：F → T；模型理由：The patient fails the age-and-authorization or tumor diameter condition.
- mechanism_04_conjunction_nested_disjunction-p02-canonical-app / R_TARGET：T → F；模型理由：The patient lacks authorization, blocking the age-and-authorization combination.
- mechanism_04_conjunction_nested_disjunction-p02-unit_equivalent-app / R_TARGET：T → F；模型理由：The patient lacks authorization, and the tumor diameter exceeds 2.5 cm.

### confirmation_A1

错误类型：{"F→T": 38, "T→F": 41, "U→F": 24, "U→T": 23}

各病例组错误数：{"mechanism_14_conditional_requirement": 25, "mechanism_23_exceptional_alternative": 24, "mechanism_17_three_exclusions": 17, "mechanism_20_two_paths_with_exclusions": 14, "mechanism_11_alternative_conjunctions": 13, "mechanism_05_disjunction_nested_conjunction": 12, "mechanism_02_disjunction": 11, "mechanism_08_negated_conjunction": 10}

- mechanism_02_disjunction-p01-unit_equivalent-non / R_TARGET：F → T；模型理由：The patient is 18 years old, meeting the age requirement, and has no suspension flag.
- mechanism_02_disjunction-p01-unit_equivalent-non / R_ALTERNATIVE：T → F；模型理由：The age requirement is met, so the negation of the age-and-authorization combination is false.
- mechanism_02_disjunction-p02-canonical-app / R_TARGET：T → F；模型理由：The age condition fails, but authorization holds, blocked by the unwaived suspension.
- mechanism_02_disjunction-p02-unit_equivalent-non / R_TARGET：F → T；模型理由：The patient is 18 years old, meeting the age requirement, and has no suspension flag.
- mechanism_02_disjunction-p02-unit_equivalent-non / R_ALTERNATIVE：T → F；模型理由：The age requirement is met, so the negation of the age-and-authorization combination is false.

### confirmation_B1

错误类型：{"F→T": 36, "T→F": 37, "U→F": 28, "U→T": 24, "T→U": 1, "F→U": 1}

各病例组错误数：{"mechanism_23_exceptional_alternative": 26, "mechanism_14_conditional_requirement": 21, "mechanism_02_disjunction": 17, "mechanism_17_three_exclusions": 16, "mechanism_20_two_paths_with_exclusions": 16, "mechanism_05_disjunction_nested_conjunction": 12, "mechanism_11_alternative_conjunctions": 11, "mechanism_08_negated_conjunction": 8}

- mechanism_02_disjunction-p01-unit_equivalent-non / R_TARGET：F → T；模型理由：The patient is 18 years old, satisfying the age condition.
- mechanism_02_disjunction-p01-unit_equivalent-non / R_ALTERNATIVE：T → F；模型理由：The age condition is satisfied, so the negation fails.
- mechanism_02_disjunction-p01-scope_and_order-app / R_TARGET：T → F；模型理由：The patient lacks authorization, and the age condition is not met.
- mechanism_02_disjunction-p01-scope_and_order-app / R_ALTERNATIVE：F → T；模型理由：The patient lacks authorization, fulfilling the negation of the age-or-authorization condition.
- mechanism_02_disjunction-p02-canonical-app / R_TARGET：T → F；模型理由：The authorization condition is met, but the unwaived suspension blocks this rule.

### confirmation_B2

错误类型：{"F→T": 34, "T→F": 48, "U→F": 23, "U→T": 25}

各病例组错误数：{"mechanism_14_conditional_requirement": 24, "mechanism_23_exceptional_alternative": 24, "mechanism_17_three_exclusions": 18, "mechanism_08_negated_conjunction": 14, "mechanism_20_two_paths_with_exclusions": 14, "mechanism_11_alternative_conjunctions": 13, "mechanism_05_disjunction_nested_conjunction": 12, "mechanism_02_disjunction": 11}

- mechanism_02_disjunction-p01-unit_equivalent-non / R_TARGET：F → T；模型理由：The patient is 18 years old, meeting the age requirement, and has no suspension flag.
- mechanism_02_disjunction-p01-unit_equivalent-non / R_ALTERNATIVE：T → F；模型理由：The age requirement is met, so the negation of the age-and-authorization combination is false.
- mechanism_02_disjunction-p02-canonical-app / R_TARGET：T → F；模型理由：The age condition fails, but authorization holds, blocked by the unwaived suspension.
- mechanism_02_disjunction-p02-unit_equivalent-non / R_TARGET：F → T；模型理由：The patient is 18 years old, meeting the age requirement, and has no suspension flag.
- mechanism_02_disjunction-p02-unit_equivalent-non / R_ALTERNATIVE：T → F；模型理由：The age requirement is met, so the negation of the age-and-authorization combination is false.

### new72_A1

错误类型：{"U→T": 6, "U→F": 9, "T→F": 5, "F→T": 3, "F→U": 1}

各病例组错误数：{"v8_new_12_diary_review_new_entry": 8, "v8_new_09_checkin_departure_room_override": 4, "v8_new_02_current_callback_permission": 2, "v8_new_04_medication_review_after_discharge": 2, "v8_new_07_interpreter_patient_preference": 2, "v8_new_08_consent_signatory_withdrawal": 2, "v8_new_11_transport_driver_alternative": 2, "v8_new_03_imaging_device_clearance": 1, "v8_new_05_result_contact_channels": 1}

- v8_new_02_current_callback_permission_u02 / R_TARGET：U → T；模型理由：Permission is active and was recorded within the past 30 days, with no suspension requested.
- v8_new_02_current_callback_permission_u02 / R_EXCEPTION：U → F；模型理由：No suspension request has been made by the patient.
- v8_new_03_imaging_device_clearance_p02_T / R_TARGET：T → F；模型理由：The signed waiver negates the specialist review requirement, but the exception rule takes precedence.
- v8_new_04_medication_review_after_discharge_u01 / R_TARGET：U → T；模型理由：The medication list was verified today and the patient brought their own containers, with no discharge since.
- v8_new_04_medication_review_after_discharge_u01 / R_ALTERNATIVE：U → F；模型理由：The medication list was verified today, so the negation of this condition is false.

### new72_B1

错误类型：{"U→T": 6, "U→F": 9, "T→F": 6, "F→T": 3}

各病例组错误数：{"v8_new_12_diary_review_new_entry": 7, "v8_new_09_checkin_departure_room_override": 5, "v8_new_02_current_callback_permission": 2, "v8_new_04_medication_review_after_discharge": 2, "v8_new_07_interpreter_patient_preference": 2, "v8_new_08_consent_signatory_withdrawal": 2, "v8_new_11_transport_driver_alternative": 2, "v8_new_03_imaging_device_clearance": 1, "v8_new_05_result_contact_channels": 1}

- v8_new_02_current_callback_permission_u02 / R_TARGET：U → T；模型理由：Permission is active and was recorded within the past 30 days, with no suspension requested.
- v8_new_02_current_callback_permission_u02 / R_EXCEPTION：U → F；模型理由：No suspension request has been made by the patient.
- v8_new_03_imaging_device_clearance_p02_T / R_TARGET：T → F；模型理由：The signed waiver negates the exception, but the specialist review requirement still blocks this rule.
- v8_new_04_medication_review_after_discharge_u01 / R_TARGET：U → T；模型理由：The medication list was verified today and the patient brought their own containers, with no discharge since.
- v8_new_04_medication_review_after_discharge_u01 / R_ALTERNATIVE：U → F；模型理由：The medication list was verified today, so the negation of this condition is false.

### new72_B2

错误类型：{"U→T": 7, "U→F": 10, "T→F": 3, "F→T": 3}

各病例组错误数：{"v8_new_12_diary_review_new_entry": 8, "v8_new_09_checkin_departure_room_override": 4, "v8_new_02_current_callback_permission": 2, "v8_new_04_medication_review_after_discharge": 2, "v8_new_07_interpreter_patient_preference": 2, "v8_new_08_consent_signatory_withdrawal": 2, "v8_new_11_transport_driver_alternative": 2, "v8_new_05_result_contact_channels": 1}

- v8_new_02_current_callback_permission_u02 / R_TARGET：U → T；模型理由：Permission is active and was recorded within the past 30 days, with no suspension requested.
- v8_new_02_current_callback_permission_u02 / R_EXCEPTION：U → F；模型理由：No suspension request has been made by the patient.
- v8_new_04_medication_review_after_discharge_u01 / R_TARGET：U → T；模型理由：The medication list was verified today and the patient brought their own containers, with no discharge since.
- v8_new_04_medication_review_after_discharge_u01 / R_ALTERNATIVE：U → F；模型理由：The medication list was verified today, so the negation of this condition is false.
- v8_new_05_result_contact_channels_u01 / R_ALTERNATIVE：U → T；模型理由：The patient does not permit either notification method, and there is no active restriction.

### natural72_A1

错误类型：{"F→T": 5, "T→F": 5, "U→T": 9, "U→F": 8}

各病例组错误数：{"v81_natural_02_breathlessness_stairs": 6, "v81_natural_01_headache_daily_activity": 5, "v81_natural_07_rash_new_product": 4, "v81_natural_03_postural_lightheadedness": 2, "v81_natural_05_night_hand_sensation": 2, "v81_natural_06_urinary_discomfort": 2, "v81_natural_08_bowel_pattern_change": 2, "v81_natural_09_ankle_swelling_asymmetry": 2, "v81_natural_11_observed_sleep_pauses": 2}

- v81_natural_01_headache_daily_activity_p01_F / R_TARGET：F → T；模型理由：Headaches do not interrupt activities and occurred at least twice this week.
- v81_natural_01_headache_daily_activity_p01_F / R_ALTERNATIVE：T → F；模型理由：Headaches do not interrupt activities, so the negation of the condition is false.
- v81_natural_01_headache_daily_activity_p02_F / R_TARGET：F → T；模型理由：Headaches began this week, so the standard assessment queue is not applicable.
- v81_natural_01_headache_daily_activity_u02 / R_TARGET：U → T；模型理由：Headaches interrupt activities and occurred at least twice this week.
- v81_natural_01_headache_daily_activity_u02 / R_EXCEPTION：U → F；模型理由：No indication that headaches began for the first time this week.

### natural72_B1

错误类型：{"F→T": 4, "T→F": 5, "U→T": 11, "U→F": 10}

各病例组错误数：{"v81_natural_01_headache_daily_activity": 6, "v81_natural_02_breathlessness_stairs": 6, "v81_natural_09_ankle_swelling_asymmetry": 4, "v81_natural_05_night_hand_sensation": 3, "v81_natural_07_rash_new_product": 3, "v81_natural_03_postural_lightheadedness": 2, "v81_natural_04_solid_food_swallowing": 2, "v81_natural_08_bowel_pattern_change": 2, "v81_natural_11_observed_sleep_pauses": 2}

- v81_natural_01_headache_daily_activity_p01_F / R_TARGET：F → T；模型理由：Headaches do not interrupt activities and occurred at least twice this week.
- v81_natural_01_headache_daily_activity_p01_F / R_ALTERNATIVE：T → F；模型理由：The condition for the alternative queue is not met.
- v81_natural_01_headache_daily_activity_p02_F / R_TARGET：F → T；模型理由：Headaches interrupt activities and occurred at least twice this week.
- v81_natural_01_headache_daily_activity_p02_F / R_EXCEPTION：T → F；模型理由：Headaches did not begin for the first time this week.
- v81_natural_01_headache_daily_activity_u02 / R_TARGET：U → T；模型理由：Headaches interrupt activities and occurred at least twice this week.

### natural72_B2

错误类型：{"F→T": 4, "T→F": 5, "U→T": 9, "U→F": 8}

各病例组错误数：{"v81_natural_02_breathlessness_stairs": 6, "v81_natural_01_headache_daily_activity": 4, "v81_natural_07_rash_new_product": 4, "v81_natural_03_postural_lightheadedness": 2, "v81_natural_05_night_hand_sensation": 2, "v81_natural_06_urinary_discomfort": 2, "v81_natural_08_bowel_pattern_change": 2, "v81_natural_09_ankle_swelling_asymmetry": 2, "v81_natural_11_observed_sleep_pauses": 2}

- v81_natural_01_headache_daily_activity_p01_F / R_TARGET：F → T；模型理由：Headaches do not interrupt activities and occurred at least twice this week.
- v81_natural_01_headache_daily_activity_p01_F / R_ALTERNATIVE：T → F；模型理由：Headaches do not interrupt activities, so the negation of the condition is false.
- v81_natural_01_headache_daily_activity_u02 / R_TARGET：U → T；模型理由：Headaches interrupt activities and occurred at least twice this week.
- v81_natural_01_headache_daily_activity_u02 / R_EXCEPTION：U → F；模型理由：No indication that headaches began for the first time this week.
- v81_natural_02_breathlessness_stairs_p01_F / R_TARGET：F → T；模型理由：Breathlessness occurs when climbing stairs and settles when he stops, with no breathlessness while sitting quietly.

完整病例文本、规则文本、标签、预测和模型理由保存在 errors.jsonl；逐病例结果保存在 case_predictions.jsonl；所有规则结果保存在 rule_predictions.jsonl。
这些病例属于人工编写的开发/验证数据，不能直接估计真实临床部署效果。
