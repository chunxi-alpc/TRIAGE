"""CPU-only reparse and deterministic cached-evidence replay; no model or gold calls."""
from __future__ import annotations
import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.module1.care_v5 import eval_logic, tri_and, tri_not
from src.module1.trialmatch_parser import PARSER_VERSION, parse_trialmatch_response
EXP = ROOT / 'experiments/care_module1_v5'


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def applicability(rule, states):
    base = eval_logic(rule['logic'], states)
    return tri_and([base, tri_not(eval_logic(rule['exception_logic'], states))]) if rule.get('exceptions') else base


def evidence_status(row):
    relations = {item.get('relation') for item in row.get('nli_candidate_scores', [])}
    conflict = 'entailment' in relations and 'contradiction' in relations
    state = 'U' if conflict or not relations else 'T' if 'entailment' in relations else 'F' if 'contradiction' in relations else 'U'
    return state, conflict


def combine_condition(tm_state, numeric_status, has_numeric, evidence_state, evidence_conflict):
    if has_numeric and numeric_status in ('T', 'F'):
        final, resolution = numeric_status, 'deterministic_numeric'
        conflict = (tm_state not in ('U', numeric_status) or evidence_conflict or
                    evidence_state in ('T', 'F') and evidence_state != numeric_status)
    elif has_numeric and numeric_status == 'U':
        final, resolution = 'U', 'unresolved_numeric_observation'
        conflict = tm_state in ('T', 'F') or evidence_state in ('T', 'F')
    elif evidence_conflict or (tm_state in ('T', 'F') and evidence_state in ('T', 'F') and tm_state != evidence_state):
        final, resolution, conflict = 'U', 'unresolved_model_evidence_conflict', True
    elif tm_state in ('T', 'F'):
        final, resolution, conflict = tm_state, 'trialmatch_retained', False
    elif evidence_state in ('T', 'F'):
        final, resolution, conflict = evidence_state, 'mednli_resolved_unknown', False
    else:
        final, resolution, conflict = 'U', 'unknown', False
    return final, resolution, conflict


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw', type=Path, default=EXP/'runs/v5_20261018/private/trialmatch_precomputed.jsonl')
    p.add_argument('--rules', type=Path, default=EXP/'data/rules_v5.jsonl')
    p.add_argument('--baseline', type=Path, default=EXP/'runs/v5_20261018/predictions_public.jsonl')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    private = a.output/'private'; private.mkdir()
    started = time.perf_counter()
    originals = {str(path): sha(path) for path in (a.raw, a.rules, a.baseline)}
    freeze = {'created_utc': datetime.now(timezone.utc).isoformat(), 'parser_version': PARSER_VERSION,
        'source_sha256': originals, 'code_sha256': {str(path.relative_to(ROOT)): sha(path) for path in
        (ROOT/'src/module1/trialmatch_parser.py', ROOT/'src/module1/care_v5.py', Path(__file__))},
        'gold_labels_read': False, 'model_calls': 0, 'gpu_used': False,
        'policy': 'Unique verbatim target and exact official label; presentation-only extraction; no JSON syntax repair; '
                  'ambiguous target and genuine generation truncation remain U; trial-wide final decision is not a condition verdict.'}
    write_json(a.output/'REPARSE_FREEZE.json', freeze)
    raw = read(a.raw); rules = {row['rule_id']: row for row in read(a.rules)}; baseline = read(a.baseline)
    expected, old_by_sample = {}, {}
    for sample in baseline:
        sid, rid = sample['sample_id'], sample['rule_id']
        if sid in old_by_sample:
            raise ValueError('Duplicate baseline sample')
        old_by_sample[sid] = sample
        old_conditions = {c['condition_id']: c for c in sample['conditions']}
        rule = rules[rid]
        for kind, group in (('inclusion', rule['conditions']), ('exclusion', rule.get('exceptions', []))):
            for atom in group:
                key = sid, rid, atom['condition_id'], kind
                expected[key] = atom, old_conditions[atom['condition_id']]
    keys = [tuple(row[k] for k in ('sample_id','rule_id','condition_id','criterion_kind')) for row in raw]
    if len(raw) != 492 or len(baseline) != 192 or len(set(keys)) != len(keys) or set(keys) != set(expected):
        raise ValueError('Expected the exact saved 492 conditions for 192 samples')
    public, reparsed, by_sample, result_by_key = [], [], collections.defaultdict(list), {}
    for row, key in zip(raw, keys):
        atom, previous = expected[key]
        old = row['trialmatch']; raw_hash = hashlib.sha256(old['raw_output'].encode()).hexdigest()
        if previous['trialmatch_output_sha256'] != raw_hash:
            raise ValueError('Raw completion differs from published original digest')
        parsed = parse_trialmatch_response(old['raw_output'], atom['text'], key[3], truncated=old['truncated'])
        repaired = {k: v for k, v in old.items() if k not in ('parsed_response','parse_error_detail','justification')}
        repaired.update(parsed)
        reparsed.append({**row, 'trialmatch': repaired})
        item = {k: row[k] for k in ('sample_id','rule_id','condition_id','criterion_kind')}
        item.update(criterion_text=atom['text'], raw_output_sha256=raw_hash,
            old_state=old['state'], old_raw_label=old.get('raw_label'), old_parse_error=old.get('parse_error'),
            state=parsed['state'], raw_label=parsed['raw_label'], parse_status=parsed['parse_status'],
            parse_error=parsed['parse_error'], unknown_reason=parsed['unknown_reason'], truncated=old['truncated'],
            output_tokens=old['output_tokens'], parse_warnings=parsed['parse_warnings'],
            ignored_non_target_placeholder_count=len(parsed.get('ignored_non_target_placeholders', [])),
            ignored_target_entry_count=parsed.get('ignored_target_entry_count', 0),
            exact_target_count=parsed.get('exact_target_count'), target_match_kind=parsed.get('target_match_kind'),
            non_target_entry_count=parsed.get('non_target_entry_count',0),
            matched_criterion_index=parsed.get('matched_criterion_index'),
            numeric_status=previous.get('numeric_status','U'))
        if old.get('parse_error') is None and (old['state'], old['raw_label']) != (parsed['state'], parsed['raw_label']):
            raise ValueError('Previously accepted target label changed')
        public.append(item); by_sample[key[0]].append(item); result_by_key[key] = item
    predictions = []
    for old_sample in baseline:
        sid, rid = old_sample['sample_id'], old_sample['rule_id']; rule = rules[rid]
        rows = by_sample[sid]
        specs = {atom['condition_id']: atom for atom in rule['conditions']+rule.get('exceptions', [])}
        updated_sample = copy.deepcopy(old_sample)
        new_conditions = []
        tm_states, numeric_states, final_states = {}, {}, {}
        for condition in updated_sample['conditions']:
            item = result_by_key[(sid, rid, condition['condition_id'], condition.get('criterion_kind','inclusion'))]
            atom = specs[condition['condition_id']]
            tm_state = item['state']; numeric_status = condition.get('numeric_status','U')
            has_numeric = isinstance(atom.get('numeric'), dict)
            numeric_state = numeric_status if has_numeric else tm_state
            nli_state, nli_conflict = evidence_status(condition)
            final, resolution, conflict = combine_condition(tm_state, numeric_state, has_numeric, nli_state, nli_conflict)
            condition.update(trialmatch_state=tm_state, trialmatch_raw_label=item['raw_label'],
                trialmatch_parse_status=item['parse_status'], trialmatch_parse_error=item['parse_error'],
                trialmatch_unknown_reason=item['unknown_reason'], numeric_status=numeric_status,
                nli_evidence_state=nli_state, state=final, resolution=resolution, conflict=conflict)
            tm_states[condition['condition_id']] = tm_state
            numeric_states[condition['condition_id']] = numeric_state
            final_states[condition['condition_id']] = final
            new_conditions.append(condition)
        model_state = applicability(rule, tm_states); numeric_state = applicability(rule, numeric_states)
        combined_state = applicability(rule, final_states)
        updated_sample.update(trialmatch_model_only_applicability=model_state,
            trialmatch_applicability=numeric_state, applicability=combined_state,
            uncertain=combined_state == 'U' or any(c['conflict'] for c in new_conditions if c['state'] in ('T','F')) or
                any(c['nli_evidence_state'] == 'U' and not (
                    isinstance(specs[c['condition_id']].get('numeric'), dict) and c['numeric_status'] in ('T','F'))
                    for c in new_conditions))
        predictions.append(updated_sample)
    write_rows(private/'trialmatch_reparsed.jsonl', reparsed)
    write_rows(a.output/'conditions_public.jsonl', public)
    write_rows(a.output/'predictions_public.jsonl', predictions)
    if any(sha(path) != fingerprint for path, fingerprint in originals.items()):
        raise ValueError('An original source file changed during reparsing')
    status = {'status':'completed', 'parser_version':PARSER_VERSION, 'conditions':len(raw), 'samples':len(baseline),
        'model_calls':0,'gpu_used':False,'gold_labels_read':False,'source_files_unchanged':True,
        'previous_parse_accepted':sum(row['old_parse_error'] is None for row in public),
        'parse_status_counts':dict(collections.Counter(row['parse_status'] for row in public)),
        'parse_error_counts':dict(collections.Counter(row['parse_error'] for row in public if row['parse_error'])),
        'unknown_reason_counts':dict(collections.Counter(row['unknown_reason'] for row in public if row['unknown_reason'])),
        'accepted_label_counts':dict(collections.Counter(row['raw_label'] for row in public if row['parse_status']=='accepted')),
        'raw_hashes_verified':len(public),'previous_accepted_labels_unchanged':True,
        'cached_nli_outputs_reused':True,'inference_repeated':False,
        'reparse_seconds':time.perf_counter()-started,'finished_utc':datetime.now(timezone.utc).isoformat()}
    write_json(a.output/'run_status.json', status)
    print(json.dumps(status, ensure_ascii=False))

if __name__ == '__main__':
    main()
