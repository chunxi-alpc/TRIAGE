#!/usr/bin/env python3
"""Read-only audit of saved data/predictions; does not run a model."""
import collections
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read(name):
    p = ROOT / name
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []

def sha(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()

def check(condition, description):
    if not condition:
        errors.append(description)

errors = []
cases = read('data/cases.jsonl')
index = {r['item_id']: r for r in cases}
check(len(index) == len(cases), 'Duplicate item IDs in cases')
family_splits = collections.defaultdict(set)
prompt_splits = collections.defaultdict(set)
family_cases = collections.defaultdict(dict)
for r in cases:
    family_splits[r['rule_family_id']].add(r['split'])
    prompt_splits[r['prompt']].add(r['split'])
    family_cases[r['rule_family_id']][r['condition']] = r
    # Independently derive truth from visible observations and provided rule text.
    nr = next(x for x in r['structured_rules'] if x['role'] == 'new')
    old = next(x for x in r['structured_rules'] if x['role'] == 'old')
    extra = next(x for x in r['structured_rules'] if x['role'] == 'independent')
    threshold = int(re.search(r'age is at least (\d+) years', nr['text']).group(1))
    age_match = re.search(r'age is (\d+) years', r['patient'])
    authorized = 'has routing authorization.' in r['patient']
    suspended = 'has a suspension flag.' in r['patient']
    joint = 'has an additional recording requirement.' in r['patient']
    state = 'U' if age_match is None else 'T' if int(age_match.group(1)) >= threshold and authorized and not suspended else 'F'
    old_valid = 'is VALID.' in old['text']
    action = nr['action'] if state == 'T' else old['action'] if state == 'F' and old_valid else 'Request review'
    if joint and action != 'Request review':
        action += ' + ' + extra['action']
    check(state == r['applicability_state'], f"Applicability mismatch {r['item_id']}")
    check(action == r['options'][r['gold']], f"Gold action mismatch {r['item_id']}")
    check(old_valid == r['old_valid'], f"Old validity mismatch {r['item_id']}")
check(all(len(v) == 1 for v in family_splits.values()), 'Family leakage across splits')
check(all(len(v) == 1 for v in prompt_splits.values()), 'Exact prompt leakage across splits')
for family, cc in family_cases.items():
    for one, two in [('min_apply','min_not_apply'), ('multi_apply','multi_not_apply'), ('old_invalid_applicable','old_invalid_inapplicable')]:
        a, b = cc[one], cc[two]
        check(a['options'] == b['options'] and a['structured_rules'] == b['structured_rules'], f'Pair nonpatient mismatch {family}/{one}')
        pa, pb = re.sub(r'age is \d+ years', 'age is AGE years', a['patient']), re.sub(r'age is \d+ years', 'age is AGE years', b['patient'])
        check(pa == pb, f'Nonminimal numeric pair {family}/{one}')
    a, b = cc['exception'], cc['exception_clear']
    check(a['patient'].replace('has a suspension flag.', 'does not have a suspension flag.') == b['patient'], f'Nonminimal exception pair {family}')
    check(a['options'] == b['options'] and a['structured_rules'] == b['structured_rules'], f'Exception pair context mismatch {family}')
facts = read('data/facts.jsonl')
qsplit = collections.defaultdict(set)
for r in facts:
    q = re.sub(r'[^a-z0-9]+', ' ', r['question'].lower()).strip()
    qsplit[q].add(r['split'])
check(all(len(v) == 1 for v in qsplit.values()), 'Normalized factual question leakage')
predictions = read('results/predictions.jsonl')
seen = set()
coverage = collections.Counter()
raw_top_is_choice = collections.Counter()
nonzero_trace_count = collections.Counter()
alpha_sets = collections.defaultdict(set)
for r in predictions:
    key = r['method'], r['item_id']
    check(key not in seen, f'Duplicate prediction {key}')
    seen.add(key)
    truth = index[r['item_id']]
    for field in ['split','task','condition','rule_family_id','pair_id','applicable','old_valid','conflict','gold']:
        check(r[field] == truth[field], f'Prediction metadata mismatch {key}/{field}')
    check(r['correct'] == (r['prediction'] == truth['gold']), f'Incorrect correctness label {key}')
    choice = truth['options'][r['prediction']]
    check(r['new_action_selected'] == (truth['new_action'] in choice), f'Incorrect new misuse marker {key}')
    coverage[(r['method'], r['split'])] += 1
    raw_top_is_choice[r['method']] += r.get('raw_output','').strip() in truth['options']
    alpha = r.get('alpha',0)
    alpha_sets[r['method']].add(alpha)
    if alpha:
        trace = r['steering']
        check(trace['calls'] == 1 and trace['edited_positions'] == 1 and trace['realized_delta_norm'] > 0, f'Steering trace inconsistent {key}')
        nonzero_trace_count[r['method']] += 1
    else:
        check(r['steering']['calls'] == 0, f'Unexpected residual edit {key}')
    if r['method'].startswith('C') and not r['conflict']:
        check(alpha == 0, f'No-conflict C edit {key}')
    if r['method'].startswith('C') and (r['applicable'] is None or r['old_valid'] is False and r['applicable'] is False):
        check(alpha == 0, f'Unknown or invalid fallback C edit {key}')
paired_predictions = {(r['method'],r['item_id']):r for r in predictions}
for r in predictions:
    reference_method = 'A' if r['method']=='D_adapt' and r.get('epsilon',0)==0 else ('A' if r['method']=='C_reduced' and r.get('alpha',0)==0 else 'B' if r['method']=='C_reduced' and r.get('signed_gate')==1 else None)
    if reference_method is not None and (reference_method,r['item_id']) in paired_predictions:
        reference=paired_predictions[(reference_method,r['item_id'])]
        check(r['scores']==reference['scores'],f"Equal interventions differ for {r['method']}/{r['item_id']} versus {reference_method}")
filter_rows = read('results/fact_filter.jsonl')
findex = {r['item_id']: r for r in facts}
ambiguous = []
for r in filter_rows:
    if r['stage'] != 'conflict':
        continue
    factual = findex[r['item_id']]
    output = re.sub(r'[^a-z0-9]+',' ',r['raw_output'].lower()).strip()
    old = re.sub(r'[^a-z0-9]+',' ',factual['answer_original'].lower()).strip()
    new = re.sub(r'[^a-z0-9]+',' ',factual['answer_mod'].lower()).strip()
    def present(answer):
        return bool(answer) and (answer == output or len(answer)>=4 and re.search(r'(?:^| )'+re.escape(answer)+r'(?: |$)',output) is not None)
    if present(old) and present(new):
        ambiguous.append({'item_id':r['item_id'],'raw_output':r['raw_output'],'old':factual['answer_original'],'new':factual['answer_mod']})
report = {'passed': not errors, 'errors': errors, 'case_count': len(cases), 'family_count': len(family_splits),
          'case_sha256':sha('data/cases.jsonl'), 'fact_count':len(facts), 'normalized_question_split_leakage':any(len(v)>1 for v in qsplit.values()),
          'prediction_rows':len(predictions), 'coverage':{f'{k[0]}/{k[1]}':v for k,v in coverage.items()},
          'raw_top_token_is_choice_count':dict(raw_top_is_choice), 'nonzero_residual_edits':dict(nonzero_trace_count),
          'alpha_values':{k:sorted(v) for k,v in alpha_sets.items()}, 'ambiguous_fact_training_conflict_matches':ambiguous,
          'checks':['independent visible-observation label derivation','exact family split isolation','exact prompt split isolation','minimal numeric/exception pair checks',
                    'normalized source question isolation','prediction labels and misuse markers','nonzero/zero residual hook traces','C unknown/invalid/no-conflict gate','identical interventions have identical full option score dictionaries'],
          'limitations':['Shared logical templates across splits; this is not unseen logical structure transfer.',
                         'The target task supplies both old and new rules in context, not parametric-versus-context knowledge.',
                         'Structured rule predicates are authored input; CARE free-text parsing and learned classifiers are not tested.',
                         'Constrained single-token option scoring is not free-generation instruction-following accuracy.',
                         'D_adapt is an independently specified key projector, not verified original CARE.',
                         'Current peak memory measurements are process high-water marks, not reset per item.',
                         'train/dev/test splitting does not protect against public-source model pretraining contamination.']}
(ROOT/'audit/independent_review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'passed':report['passed'],'errors':errors,'cases':len(cases),'predictions':len(predictions),'ambiguous_training_conflicts':len(ambiguous)}))
