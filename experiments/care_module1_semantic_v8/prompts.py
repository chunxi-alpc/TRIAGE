"""Frozen direct semantic applicability prompts; no patient fact parser or retrieval."""
import json

SYSTEM = """Determine whether each complete natural-language guideline rule applies to the described patient, using the question, patient narrative, and all neighboring rules. The rules describe an invented routing simulation, not medical care.
For each rule, use T when the narrative establishes that the whole rule applies; F when it establishes that the whole rule does not apply; U when relevant information is missing or unresolved. Missing information is not false. Interpret negation, exceptions, current versus historical observations, and patient versus relatives directly from their meaning. One irrelevant unknown need not prevent a conclusion supported by the other information.
Return one JSON object with a rules array and exactly one entry per supplied rule. Each entry has rule_id, state (T/F/U), evidence_quote (one short exact contiguous quote from the patient narrative, or an empty string if none), and reason_short (one brief sentence explaining the applicability decision). Quotes are evidence excerpts, not a replacement for assessing the whole rule. Keep each quote under 35 words and each reason under 25 words. Do not produce an action, confidence score, preliminary reasoning, or additional prose."""


def render_clause(tree, atoms):
    if isinstance(tree, str):
        text = atoms[tree]['text'].strip().rstrip('.')
        return text[0].lower() + text[1:]
    op = tree['op'].upper(); args = tree.get('args', [])
    if not args:
        return 'the rule has no additional prerequisite' if op == 'AND' else 'no exception applies'
    rendered = [render_clause(x, atoms) for x in args]
    if op == 'NOT':
        return 'it is not the case that (' + rendered[0] + ')'
    return '(' + (' and ' if op == 'AND' else ' or ').join(rendered) + ')'


def natural_rule(rule):
    # Legacy data contain only pre-parsed guideline semantics. This is an offline
    # lossless verbalization of G, never a parser of X or reference labels.
    if rule.get('text'):
        return rule['text']
    if rule.get('natural_rule_text'):
        return rule['natural_rule_text']
    atoms = {a['condition_id']: a for a in rule.get('conditions', []) + rule.get('exceptions', [])}
    text = 'When ' + render_clause(rule['logic'], atoms)
    exception = rule.get('exception_logic', {'op':'OR', 'args':[]})
    if isinstance(exception, str) or exception.get('args'):
        text += ', unless ' + render_clause(exception, atoms)
    return text + ', assign ' + rule['action'] + '.'


def prediction_input(case):
    # A positive allowlist prevents any patient structured facts, reference state,
    # reference evidence, target action, or gold answer from entering the model.
    return {'question': case.get('question', case.get('query', 'Which rule applies?')),
            'patient': case['patient'],
            'rules': [{'rule_id': r['rule_id'], 'text': natural_rule(r), 'action': r['action']} for r in case['rules']]}


def user_content(inputs):
    return ('Question: ' + inputs['question'] + '\nPatient: ' + inputs['patient'] +
            '\nGuideline rules and neighboring candidate branches:\n' +
            '\n'.join(r['rule_id'] + ': ' + r['text'] for r in inputs['rules']) +
            '\nAssess every listed rule. Return {"rules":[{"rule_id":"...","state":"T|F|U","evidence_quote":"...","reason_short":"..."}]}.')


def messages_for(inputs, arm, examples):
    messages = [{'role':'system', 'content':SYSTEM}]
    if arm == 'A1':
        # Three demonstrations: one minimal T/F pair (two narratives), one
        # exception/negation narrative, and one missing-information narrative.
        for demonstration in examples['demonstrations']:
            for example in demonstration['cases']:
                messages += [{'role':'user', 'content':user_content(example['input'])},
                             {'role':'assistant', 'content':json.dumps({'rules':example['output']}, ensure_ascii=False)}]
    messages.append({'role':'user','content':user_content(inputs)})
    return messages


def parse_output(text, rule_ids):
    """One JSON schema check, no evidence gates or semantic state rewriting."""
    decoder = json.JSONDecoder(); candidate = None
    clean = text.split('</think>')[-1]
    for offset, ch in enumerate(clean):
        if ch != '{':
            continue
        try:
            obj, _ = decoder.raw_decode(clean, offset)
        except ValueError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get('rules'), list):
            candidate = obj['rules']; break
    if candidate is None:
        return None, 'missing_rules_json'
    if len(candidate) != len(rule_ids):
        return None, 'wrong_rule_count'
    found = {}
    for row in candidate:
        if not isinstance(row, dict) or row.get('rule_id') not in rule_ids or row['rule_id'] in found:
            return None, 'invalid_or_duplicate_rule_id'
        if row.get('state') not in ('T','F','U'):
            return None, 'invalid_state'
        if not isinstance(row.get('evidence_quote'), str) or not isinstance(row.get('reason_short'), str):
            return None, 'missing_text_fields'
        found[row['rule_id']] = {k:row[k] for k in ('rule_id','state','evidence_quote','reason_short')}
    return [found[rid] for rid in rule_ids], None
