"""Literal JSON examples for fixed-patient counterfactual construction."""
import json

_EXAMPLE = {
    "facts": [
        {"age_years": 17, "eligibility_confirmed": True, "assessment_complete": True},
        {"age_years": 19, "eligibility_confirmed": True, "assessment_complete": True}
    ],
    "predicate": {
        "op": "and",
        "args": [
            {"op": ">=", "field": "age_years", "value": 18},
            {"op": "==", "field": "eligibility_confirmed", "value": True}
        ]
    },
    "ancestor_predicates": [
        {"gid": "G0", "predicate": {"op": "==", "field": "assessment_complete", "value": True}}
    ],
    "decisive_sentences": [
        "The patient is 17 years old. Eligibility has been confirmed.",
        "The patient is 19 years old. Eligibility has been confirmed."
    ],
    "protected_action_ids": [],
    "condition_change": {
        "valid": True,
        "base_state": 0,
        "edits": [{"old": "at least 18 years old", "new": "at least 16 years old"}],
        "predicate": {
            "op": "and",
            "args": [
                {"op": ">=", "field": "age_years", "value": 16},
                {"op": "==", "field": "eligibility_confirmed", "value": True}
            ]
        },
        "ancestor_predicates": [],
        "reason": "The unchanged 17-year-old now meets the changed age threshold; the other required criterion is unchanged."
    },
    "clinical_background": {
        "valid": True,
        "texts": ["There is no history of hypertension.", "There is a history of well-controlled hypertension."],
        "changed_fields": ["hypertension_history"],
        "reason": "In this example hypertension does not enter the target or ancestor condition."
    },
    "treatment_preference": {
        "valid": True,
        "texts": ["The patient prefers conservative symptom-focused care.", "The patient prefers intensive active treatment."],
        "changed_fields": ["treatment_preference"],
        "reason": "In this example preference does not enter the target or ancestor condition."
    },
    "reason": "The two fixed patients differ in age and retain the same ancestor facts."
}

MAIN_INSTRUCTION = """Construct additions to an AUTOMATIC clinical-guideline research dataset.
Return ONE valid JSON object, without Markdown, prose, comments or unquoted placeholders.

The full tree supplies context. The target is ONLY current_condition. Encode its complete logical
condition and the explicitly listed ancestors; do not combine every other tree node into the target.
The supplied patients are FIXED: patients[state=0] has label 0, and patients[state=1] has label 1.
Extract their observations into facts[0] and facts[1] in that same order. Do not reverse states, invent
replacement patients, copy a positive observation into both patients, or replace missing facts with a desired answer.
For the definition subset only, follow the additional definition instruction to construct the specified patients.

Use short ASCII field names for actual observations (age_years, tumor_present, etc.).
Facts are two flat field-to-value dictionaries, NOT expressions or lists of expressions.
Do not use condition_true, answer, the entire question, or the desired branch as a fact field.
Every field used by a target or ancestor expression must exist in BOTH fact dictionaries.
Represent conjunctions, disjunctions, ranges, exclusions and fixed alternatives of the COMPLETE conditions.
Expressions use: and/or with args, not with arg, or leaves with op, field, value.
Allowed leaf operators are <, <=, >, >=, ==, !=, in, not_in. Use actual numeric, Boolean or categorical values.
The target predicate must evaluate false on facts[0] and true on facts[1].
Each ancestor predicate must preserve its supplied label for BOTH patients. An ancestor has its own
condition; never substitute the current predicate for it merely because both involve the same clinical topic.

Keep the schema in the example below, adapting its content to INPUT:
- decisive_sentences are the literal relevant patient facts, separately for state 0 and state 1.
- protected_action_ids retain terminal names/explanations needed to DEFINE the target or an ancestor.
  A treatment mentioned in such a definition is not independent action text.
- condition_change modifies a threshold or a necessary criterion/definition with the patient FIXED.
  Choose base_state 0 OR 1 so the COMPLETE modified predicate actually flips that state's original label.
  Only one state needs to flip. Fixed age/measurement does not prevent changing its decision threshold.
  Categorical criteria may change too. Do not merely negate the whole question.
  edits.old must occur verbatim in supplied conditions or original definitions. Apply changes to all
  equivalent explicit definitions/references so the complete input is consistent. Preserve ancestor truth;
  if an ancestor text changes, include its updated predicate in condition_change.ancestor_predicates.
- clinical_background changes a meaningful clinical detail unrelated to EVERY target/ancestor clause.
- treatment_preference changes preference only when no target/ancestor depends on consent, choice,
  willingness, preference or a treatment plan. Otherwise mark this variant invalid.
- Each valid background/preference has EXACTLY TWO DISTINCT sentences in texts and lists changed_fields.
  They are appended to either fixed patient, so both sentences must be compatible with BOTH patients.
  Changed fields must not be used by the target or ancestors.
- An unavailable variant may have valid:false and its reason. Construct the other available variants.
Do not invent original source quotations, clinical verification, model predictions or reference scores.

The following is a VALID JSON format example for an example age/eligibility condition and its assessment
ancestor. Its facts, node ID, quotations and background are illustrative, not values to copy into this task:
""" + json.dumps(_EXAMPLE, ensure_ascii=True, indent=2)

_CONDITION_EXAMPLE = {
    "condition_change": {
        "valid": True,
        "base_state": 0,
        "edits": [{"old": "at least 18 years old", "new": "at least 16 years old"}],
        "predicate": {
            "op": "and",
            "args": [
                {"op": ">=", "field": "age_years", "value": 16},
                {"op": "==", "field": "eligibility_confirmed", "value": True}
            ]
        },
        "ancestor_predicates": [],
        "reason": "With age fixed at 17 and eligibility confirmed, changing the threshold from 18 to 16 flips the complete condition."
    }
}

CONDITION_INSTRUCTION = """Construct ONLY a condition/definition counterfactual for this automatic research dataset.
Return one valid JSON object with the key condition_change, as in the example below. No Markdown or comments.

INPUT includes already verified facts for state 0 and state 1, the complete current predicate, the original
conditions/definitions/tree, and ancestor predicates/outcomes. Keep those patient facts and their order FIXED.
Choose base_state 0 OR 1. Modify the condition or a necessary definition so its COMPLETE truth flips for
that one fixed patient (0 becomes 1, or 1 becomes 0). The other state's truth need not flip.

A fixed age, laboratory value or category is not a reason to reject this task: change the decision
threshold or required criterion. For example, age 17 fails age >=18 but passes age >=16 while age stays 17.
For categorical conditions, change a clinically interpretable required category or definition, not the
patient. Do not simply prefix NOT to the entire condition, add the desired answer, or turn a population
statistic into an individual diagnostic threshold.

Return a COMPLETE modified predicate over the existing verified observation fields. Retain all other
required conjuncts, disjuncts, ranges and exceptions. Evaluate the full expression, not only the edited clause.
Use the same executable expression format: and/or with args, not with arg; leaves op, field, value.
Allowed leaf operators: <, <=, >, >=, ==, !=, in, not_in.
Use only existing fact fields and do not edit or regenerate either patient's facts.

edits contains exact old/new text replacements. Each old quote must occur literally in the supplied
condition or necessary original definition. Synchronize equivalent explicit expansions/references across
the whole input; do not leave a contradictory old threshold or definition elsewhere. Preserve every
ancestor's outcome for the selected patient. If an edit also changes an ancestor's text or referenced
definition, return its COMPLETE updated expression in ancestor_predicates.

For a definition-subset node, keep the original short target term/question and modify its NECESSARY
ORIGINAL DEFINITION. Update explicit equivalents elsewhere as needed; do not expand the definition into
the short target question. The unchanged patient must flip under the modified definition.
If no coherent change is available, return condition_change with valid:false and a specific reason.
Otherwise include valid, base_state, edits, predicate, ancestor_predicates and reason.

The following is a valid JSON example for fixed age 17 with confirmed eligibility. Adapt all values and
literal quotations to INPUT; do not copy the example when its facts or quotes do not apply:
""" + json.dumps(_CONDITION_EXAMPLE, ensure_ascii=True, indent=2)


_ANCESTOR_TRUTH = """
Ancestor labels are truth values of the ORIGINAL ancestor questions, not indicators that the chosen path is valid.
For an ancestor question "Does the patient have distant metastasis?" with label 0, both patients may have
metastasis_present=false, but its predicate must be {"op":"==","field":"metastasis_present","value":true},
which evaluates FALSE. Do not use value:false just to make the no-branch evaluate true.
If the original question explicitly asks about ABSENCE, preserve that negation; do not mechanically
change every comparison to value:true. Represent each original question faithfully.
"""
MAIN_INSTRUCTION = _ANCESTOR_TRUTH + MAIN_INSTRUCTION
CONDITION_INSTRUCTION = _ANCESTOR_TRUTH + CONDITION_INSTRUCTION

_DEF_EXAMPLE = {
    "definition_valid": True,
    "short_condition": "Does the patient have recurrent BV?",
    "shared_record": "Bacterial vaginosis has been confirmed.",
    "fact_sentences": [
        "There were 2 symptomatic BV episodes in the last 12 months.",
        "There were 4 symptomatic BV episodes in the last 12 months."
    ],
    "facts": [
        {"bv_confirmed": True, "bv_episodes_12mo": 2},
        {"bv_confirmed": True, "bv_episodes_12mo": 4}
    ],
    "predicate": {"op": ">=", "field": "bv_episodes_12mo", "value": 3},
    "ancestor_predicates": [
        {"gid": "G0", "predicate": {"op": "==", "field": "bv_confirmed", "value": True}}
    ],
    "decisive_sentences": [
        "There were 2 symptomatic BV episodes in the last 12 months.",
        "There were 4 symptomatic BV episodes in the last 12 months."
    ],
    "protected_action_ids": [],
    "condition_change": {
        "valid": True,
        "base_state": 0,
        "edits": [
            {"old": "three or more symptomatic episodes in 12 months",
             "new": "two or more symptomatic episodes in 12 months"}
        ],
        "predicate": {"op": ">=", "field": "bv_episodes_12mo", "value": 2},
        "ancestor_predicates": [],
        "reason": "The unchanged patient with two episodes now satisfies the modified definition."
    },
    "clinical_background": {
        "valid": True,
        "texts": ["There is no history of asthma.", "There is a history of well-controlled asthma."],
        "changed_fields": ["asthma_history"],
        "reason": "In this example asthma is absent from the target and ancestors."
    },
    "treatment_preference": {
        "valid": True,
        "texts": ["The patient prefers conservative symptom-focused care.", "The patient prefers active treatment."],
        "changed_fields": ["treatment_preference"],
        "reason": "In this example preference is absent from the target and ancestors."
    },
    "reason": "The two constructed patients differ only in BV recurrence count."
}

DEF_INSTRUCTION = """Construct a DEFINITION counterfactual example for an automatic guideline research pilot.
This is a separate task with the COMPLETE output schema below. Return one compact valid JSON object only.
Always include definition_valid as a JSON boolean. If true, ALL fields in the example are required,
including short_condition, shared_record and fact_sentences. Do not omit these four definition fields.
Use one short sentence for each reason. Do not reproduce the whole input tree in your response.

The supplied full tree, source_rule_evidence, definition_links and original definition quotations are context.
The target is current_condition; use only this target and the listed ancestors for patient construction.
existing_patient is a reference vignette, not two fixed counterfactual patients. Construct TWO explicit
patients: shared_record + fact_sentences[0] has label 0, and shared_record + fact_sentences[1] has label 1.
Keep all shared facts identical and vary one decisive observed field. Preserve the original truth value of
EVERY listed ancestor for BOTH patients. Do not encode unrelated later tree nodes as extra requirements.
""" + _ANCESTOR_TRUTH + """
short_condition must be a natural yes/no question preserving the current logical condition but retaining
the ORIGINAL short term instead of restating its definition. At least one original target_term from
definition_links must occur literally in this short question. Use the source definition to interpret it.
Do not invent an abbreviation, a definition dependency, or an original source quotation.

facts is a list of TWO flat observation dictionaries, in false/true order. Use short ASCII field names and
actual numbers, booleans or categories. Every field referenced by a predicate must exist in BOTH dictionaries.
Do not use the answer, condition_true, severe_condition_met, or the whole question as an observation field.
predicate is the COMPLETE truth condition under the ORIGINAL definition; evaluate it false/true on the two
facts dictionaries. Keep conjunctions, disjunctions, ranges and exceptions. Expressions use and/or with args,
not with arg, and leaves with op, field, value. Allowed leaves: <, <=, >, >=, ==, !=, in, not_in.
ancestor_predicates encodes the original ancestor QUESTIONS, whose outputs must equal their supplied labels.
decisive_sentences contains literal decisive facts from the corresponding constructed patient.

For condition_change keep the short question and patient fixed, change a NECESSARY ORIGINAL DEFINITION,
and flip the complete truth for base_state 0 OR 1. Only that state must flip. Use exact old/new quotations
that occur in the supplied original definition text. Synchronize explicit equivalents elsewhere in the tree
or supplied definitions, preserving ancestor outcomes. Include updated ancestor_predicates when an edit
changes ancestor text/definitions. Changing only the target question is not a definition counterfactual.
A fixed measurement is compatible with a changed definition threshold; evaluate the full modified predicate.
Never merely negate the entire condition or edit a patient fact.

protected_action_ids preserves terminal treatment names/explanations used to define the target or ancestors.
Clinical-background and preference variants are optional: use valid:false with a reason if unavailable.
A valid variant has EXACTLY TWO DISTINCT sentences in texts, compatible with BOTH constructed patients,
and changed_fields unused by every target/ancestor condition. Preference is invalid when consent, willingness,
choice or treatment planning participates in those conditions. Do not fabricate clinical review or model results.

If the source cannot support a coherent definition example, explicitly return
{"definition_valid":false,"reason":"Specific source or ancestor conflict."}
Missing definition_valid is a schema error, not a claim that the task is impossible.
Otherwise return ALL fields in this illustrative valid JSON example, adapting every value and quote to INPUT:
""" + json.dumps(_DEF_EXAMPLE, ensure_ascii=True, indent=2)
