"""Explicit rule-to-source mapping for the fixed A0 clinical development cohort."""
from dataclasses import replace
import json

from aster_seka_adapter import PredictedSpans
from clinical_core import SELECTOR_VERSION, official_prompt, sha, support_spans


def grounded_prompts(template, guideline, patient, rules, tfu):
    by_id = {row['rule_id']: row for row in tfu}
    if len(by_id) != len(tfu) or set(by_id) != {r.rule_id for r in rules}:
        raise ValueError('Each extracted rule needs exactly one saved A0 assessment')
    mapped = []
    for rule in rules:
        if guideline[rule.start:rule.end] != rule.rule_quote:
            raise ValueError('Rule range does not match its original passage')
        quotes = [rule.condition_quote, rule.action_quote, *rule.exception_quotes]
        if any(not quote or quote not in rule.rule_quote for quote in quotes):
            raise ValueError('Condition/action/exception is not grounded in the passage')
        mapped.append({
            'rule_id': rule.rule_id, 'condition_quote': rule.condition_quote,
            'action_quote': rule.action_quote, 'exception_quotes': list(rule.exception_quotes),
            'source_character_range': [rule.start, rule.end],
            'predicted_assessment': by_id[rule.rule_id],
        })
    selected = support_spans(rules, tfu)
    evidence = PredictedSpans(sha(SELECTOR_VERSION), selected)
    base = official_prompt(template, guideline, patient, evidence=evidence)
    if mapped:
        base = replace(base, text=base.text +
            '\nPredicted rule applicability with exact original-source correspondence '
            '(unverified; unknown is not false). Use the complete guideline and patient '
            'history; preserve the required output format. Source ranges are character '
            'offsets in the guideline, not decision-tree node IDs.\n' +
            json.dumps(mapped, ensure_ascii=False, sort_keys=True))
    passages = [guideline[span.start:span.end] for span in selected]
    support = replace(base, text=base.text +
        '\nAttend to these predicted relevant quotations (unverified):\n' +
        '\n\n'.join(passages)) if passages else base
    return base, support, mapped
