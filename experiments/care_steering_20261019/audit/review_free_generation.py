#!/usr/bin/env python3
"""Audit completed exploratory free-generation outputs without invoking a model."""
import ast
import collections
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read(name):
    return [json.loads(s) for s in (ROOT / name).read_text().splitlines() if s.strip()]

if not (ROOT / 'results/free_complete.json').exists():
    raise RuntimeError('Free generation has not completed; no partial audit is substituted')
cases = {r['item_id']:r for r in read('data/cases.jsonl') if r['split'] in ('test','transfer')}
completions = read('results/free_completions.jsonl')
predictions = read('results/free_predictions.jsonl')
selected = json.loads((ROOT/'results/selected.json').read_text())['alpha']
source_tree = ast.parse((ROOT/'src/run_free.py').read_text())
parser_function = next(node for node in source_tree.body if isinstance(node,ast.FunctionDef) and node.name=='parse_answer')
namespace = {'re':re}
exec(compile(ast.Module(body=[parser_function],type_ignores=[]),'<saved_parser>','exec'),namespace)
parse = namespace['parse_answer']
opts = {'A':'Route Alder','B':'Route Birch + Record Cedar','C':'Request review','D':'Route Dogwood'}
checks = [('A','A'),('Final answer: **B. Route Birch + Record Cedar**','B'),('The answer is C.','C'),
          ('Route Dogwood.','D'),('Reasoning mentions A and B\nNo clear conclusion','INVALID'),
          ('B. Route Alder','INVALID'),('Route Birch','INVALID'),('Either A or B','INVALID')]
for raw, expected in checks:
    assert parse(raw,opts)[0]==expected,(raw,parse(raw,opts),expected)
assert not any(isinstance(node,ast.Name) and 'gold' in node.id.lower() for node in ast.walk(parser_function))
cache = {}
errors = []
for row in completions:
    key=(row['item_id'],row['alpha'])
    if key in cache:errors.append(f'Duplicate completion {key}')
    cache[key]=row
    trace=row['steering']
    if row['alpha']:
        if not (trace['calls']==1 and trace['edited_positions']==row['batch_size'] and trace['realized_delta_norm']>0):
            errors.append(f'Invalid batched prefill edit trace {key}')
    elif trace['calls']!=0:
        errors.append(f'Unexpected zero-alpha edit {key}')
    prediction,reason=parse(row['raw_output'],cases[row['item_id']]['options'])
    if prediction!=row['prediction'] or reason!=row['parse_reason']:
        errors.append(f'Parser mismatch {key}')
seen=set();reuse=[];counts=collections.defaultdict(collections.Counter)
for row in predictions:
    key=(row['method'],row['item_id'])
    if key in seen:errors.append(f'Duplicate prediction {key}')
    seen.add(key)
    original=cache.get((row['item_id'],row['alpha']))
    if original is None or original['raw_output']!=row['raw_output'] or original['prediction']!=row['prediction']:
        errors.append(f'Prediction not backed by actual completion {key}')
    truth=cases[row['item_id']]
    if row['correct'] != (row['prediction']==truth['gold']):errors.append(f'Incorrect score {key}')
    valid=row['prediction'] in truth['options']
    if valid!=row['valid_answer']:errors.append(f'Invalid parse validity {key}')
    if not valid and (row['correct'] or row['new_action_selected'] is not None):errors.append(f'Invalid answer removed from failure/misuse accounting {key}')
    expected_alpha=0.0 if row['method'] in ('A','D_adapt') else selected if row['method']=='B' else selected*row['signed_gate']
    if row['alpha']!=expected_alpha:errors.append(f'Method coefficient mismatch {key}')
    owner='A' if row['alpha']==0 else 'B' if row['alpha']==selected else 'C_reduced'
    completion_id=hashlib.sha256(f"{row['item_id']}|{row['alpha']:.17g}".encode()).hexdigest()[:24]
    reuse.append({'item_id':row['item_id'],'method':row['method'],'alpha':row['alpha'],'completion_id':completion_id,
                  'executed':row['method']==owner,'reused_from':None if row['method']==owner else owner})
    counts[row['method']]['items']+=1
    counts[row['method']]['correct']+=row['correct']
    counts[row['method']]['invalid']+=not valid
    counts[row['method']]['truncated']+=row['truncated']
    counts[row['method']]['owns_unique_completion']+=row['method']==owner
for method in ['A','B','C_reduced','D_adapt']:
    actual={iid for m,iid in seen if m==method}
    if actual!=set(cases):errors.append(f'Method coverage mismatch {method}')
if sum(r['executed'] for r in reuse)!=len(completions):errors.append('Unique-completion ownership count mismatch')
(ROOT/'audit/free_reuse_manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in reuse))
report={'passed':not errors,'errors':errors,'parser_tests_passed':len(checks),'unique_completions':len(completions),'prediction_rows':len(predictions),
        'method_summary':{m:dict(c) for m,c in counts.items()},
        'limitations':['Parser is deliberately strict, case-sensitive for option actions, and uses only the final nonempty line; semantically equivalent prose may be marked invalid.',
                       'Invalid outputs remain incorrect in accuracy. New-rule misuse is only known for parsed outputs; report coverage and full-denominator lower/upper bounds.',
                       'The raw reused_completion field labels all C rows as reused. The derived manifest correctly marks positive-alpha C completions as separately executed.',
                       'Batch wall time allocated across items is not isolated latency. True total compute is reported by free_complete.json, not the sum of replicated method-row times.',
                       'Free-generation gate_seconds is an unmeasured per-item placeholder; total run time includes gate computation.']}
(ROOT/'audit/free_generation_review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'passed':not errors,'errors':errors,'unique_completions':len(completions),'method_summary':report['method_summary']}))
