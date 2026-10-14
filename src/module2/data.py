"""Small automatic rule contrasts using official non-test MEGA-CDP trees.

These are synthetic calibration probes, not new manual clinical annotations.
Original source notes are retained for audit but not mixed into contradictory probes.
"""
from __future__ import annotations
import json
from src.data_loader import load_dataset


def _paths(node, conditions=()):
    if node.get('leaf') and isinstance(node.get('conclusion'), str):
        yield list(conditions), node['conclusion'], node.get('index')
        return
    condition = node.get('description')
    if not isinstance(condition, str):
        return
    for branch in ('yes', 'no'):
        if isinstance(node.get(branch), dict):
            yield from _paths(node[branch], (*conditions, {'text': condition,
                'answer': branch == 'yes', 'node_index': node.get('index')}))


def build_calibration_pairs(limit=2):
    """Return one automatically contrasted rule from each of <=limit train_val trees.

    Applicability is exact conjunction of official branch predicates. One finding
    is flipped for the inapplicable probe; no model-derived or human labels enter.
    MedPIC-Bench is deliberately excluded because its local split is test only.
    """
    if limit < 1:
        raise ValueError('limit must be positive')
    pairs, seen = [], set()
    for record in load_dataset('MEGA-CDP', split='train_val'):
        source = record['raw']['decision_tree']
        filename = source['filename']
        if filename in seen or source.get('language') != 'en':
            continue
        seen.add(filename)
        tree = json.loads(source['text']) if isinstance(source['text'], str) else source['text']
        paths = [(p, a, leaf) for p, a, leaf in _paths(tree['decision_tree']) if 1 <= len(p) <= 4]
        if not paths:
            continue
        # Prefer a restrictive clinical rule rather than an outside-scope fallback.
        paths.sort(key=lambda x: (-sum(c['answer'] for c in x[0]), len(x[0]), x[2]))
        conditions, action, leaf = paths[0]
        rule_conditions = '\n'.join(f"{i+1}. {c['text']} Required finding: {'YES' if c['answer'] else 'NO'}."
                                       for i, c in enumerate(conditions))
        rule = f'Apply this recommendation only if ALL required findings hold:\n{rule_conditions}\nRecommendation: {action}'
        def patient(flip=False):
            return 'Patient assessment (explicit findings):\n' + '\n'.join(
                f"{i+1}. {c['text']} Finding: {'YES' if c['answer'] != (flip and i == len(conditions)-1) else 'NO'}."
                for i, c in enumerate(conditions))
        pair_id = f'{filename}:leaf{leaf}'
        pairs.append({'pair_id': pair_id, 'rule': rule, 'action': action,
            'conditions': conditions, 'applicable_patient': patient(),
            'inapplicable_patient': patient(True), 'flipped_condition_index': len(conditions)-1,
            'provenance': {'dataset': 'MEGA-CDP', 'split': 'train_val',
                'filename': filename, 'guideline_id': source.get('guideline_id'),
                'leaf_index': leaf, 'source_record_id': record['id'],
                'original_patient': record['question'],
                'original_rule_text': action,
                'construction': 'official conjunction path; flip exactly one required branch predicate',
                'labels': 'automatic rule-logic applicability only; not clinical outcome annotation',
                'reference_answers_used': False, 'synthetic_patient': True}})
        if len(pairs) >= limit:
            break
    if not pairs:
        raise ValueError('No usable English non-test tree paths found')
    return pairs


def _logic_value(logic, statuses):
    if isinstance(logic, str):
        return statuses[logic]
    values = [_logic_value(arg, statuses) for arg in logic.get('args', [])]
    op = logic.get('op', 'AND')
    if op == 'AND': return all(values)
    if op == 'OR': return any(values)
    if op == 'NOT' and len(values) == 1: return not values[0]
    raise ValueError('Unsupported cached module-one logic')


def _probe_context(output, rule_id):
    """Cached relations and gates select causal hook targets; labels never enter model prompts."""
    return {'schema_version': output['schema_version'], 'question': output['question'],
        'relations': [{k: row[k] for k in ('condition_id', 'condition', 'status', 'evidence',
                       'weight', 'signed_weight', 'binding_gate', 'action_gate', 'prior_competition_weight') if k in row}
                      for row in output['relation_weights'] if row['rule_id'] == rule_id],
        'binding_gate': output['binding_gate'].get(rule_id, {}),
        'action_gate': output['action_gate'].get(rule_id, 0.0)}


def build_cached_calibration_pairs(cache_path=None, smoke_path=None, limit=2):
    """Use saved module-one gid3 results; never rerun module one inside search.

    Original labels/relations/gates come from the actual cached output. The single
    evidence sentence counterfactual flips a shared base predicate, then reuses
    module-one logical structure and fixed relevance/conflict weights. Derived
    gates are explicitly counterfactual approximations, not new NLI/model outputs.
    """
    import copy
    import hashlib
    from pathlib import Path
    from src.data_loader import ROOT
    cache_path = Path(cache_path or ROOT/'results/module1_output.json')
    smoke_path = Path(smoke_path or ROOT/'results/module1_smoke.json')
    cached = json.loads(cache_path.read_text())
    if cached.get('schema_version') != 'module1.v1':
        raise ValueError('Expected actual module1.v1 cached output')
    smoke = json.loads(smoke_path.read_text())
    candidates = [s for s in smoke['samples'] if s['input_source'].get('guideline_id') == 3
        and s.get('question') == cached['question']]
    if len(candidates) != 1:
        raise ValueError('Need exactly one gid3 smoke patient matching module-one cache')
    sample = candidates[0]
    if sample['input_source'].get('split') not in {'train', 'train_val', 'validation', 'calibration'}:
        raise ValueError('Cached module-one source cannot be a test split')
    patient = sample['patient']
    relations = cached['condition_evidence_relations']
    apps = {r['rule_id']: r for r in cached['applicability']}
    # Use an actual applicable rule and the nearest inapplicable alternative.
    rules = sorted(cached['candidate_rules'], key=lambda r: (
        apps[r['rule_id']]['status'] != 'applicable',
        len(apps[r['rule_id']].get('blockers', [])), -len(r.get('conditions', []))))
    pairs = []
    for rule in rules:
        rid = rule['rule_id']
        if apps[rid]['status'] not in {'applicable', 'not_applicable'}:
            continue
        is_app = apps[rid]['status'] == 'applicable'
        rule_relations = [r for r in relations if r['rule_id']==rid]
        if not is_app:
            rule_relations = [r for r in rule_relations if r['condition_id'] in apps[rid].get('blockers', [])]
        # Last decisive condition produces a minimal matched leaf5/leaf6 contrast.
        rule_relations = [r for r in rule_relations if r.get('evidence') and
                          r['status'] in {'satisfied', 'not_satisfied'}]
        if not rule_relations:
            continue
        target = rule_relations[-1]
        evidence = target['evidence'][0]
        if patient[evidence['start']:evidence['end']] != evidence['text']:
            raise ValueError('Cached evidence offsets do not match the source patient')
        underlying = target.get('underlying_condition', target['condition'])
        old_base = target.get('base_condition_status', 'satisfied' if
            (target['status']=='satisfied') != bool(target.get('negated')) else 'not_satisfied')
        new_base = old_base != 'satisfied'
        replacement = f"The explicit patient finding for '{underlying}' is {'YES' if new_base else 'NO'}."
        cf_patient = patient[:evidence['start']] + replacement + patient[evidence['end']:]
        derived = copy.deepcopy(cached)
        altered_conditions = {}
        delta = len(replacement) - (evidence['end']-evidence['start'])
        for row in derived['condition_evidence_relations']:
            for span in row.get('evidence', []):
                if span['start'] >= evidence['end']:
                    span['start'] += delta
                    span['end'] += delta
            if row.get('underlying_condition', row['condition']) != underlying:
                continue
            satisfied = new_base != bool(row.get('negated', False))
            row['status'] = 'satisfied' if satisfied else 'not_satisfied'
            row['base_condition_status'] = 'satisfied' if new_base else 'not_satisfied'
            row['evidence'] = [{'text': replacement, 'start': evidence['start'],
                'end': evidence['start']+len(replacement), 'sentence_id': evidence.get('sentence_id')}]
            row['method'] = 'automatic_counterfactual_from_cached_base_predicate'
            altered_conditions[(row['rule_id'], row['condition_id'])] = row
        for app in derived['applicability']:
            item = next(r for r in derived['candidate_rules'] if r['rule_id']==app['rule_id'])
            rows = [r for r in derived['condition_evidence_relations'] if r['rule_id']==app['rule_id']]
            statuses = {r['condition_id']: r['status']=='satisfied' for r in rows}
            applies = _logic_value(item['logic'], statuses) and not _logic_value(item.get('exception_logic', {'op':'OR','args':[]}), statuses)
            app['status'] = 'applicable' if applies else 'not_applicable'
            relevance = next(x['score'] for x in derived['relevance_scores'] if x['rule_id']==app['rule_id'])
            conflict = next(x['score'] for x in derived['conflict_scores'] if x['rule_id']==app['rule_id'])
            derived['action_gate'][app['rule_id']] = round(float(relevance)*(0.5+0.5*float(conflict)) if applies else 0.,6)
        derived_relations = {(r['rule_id'], r['condition_id']): r for r in derived['condition_evidence_relations']}
        for row in derived['relation_weights']:
            updated = derived_relations.get((row['rule_id'], row['condition_id']))
            if updated:
                row['status'], row['evidence'] = updated['status'], updated['evidence']
                row['signed_weight'] = row['weight'] if row['status']=='satisfied' else -row['weight']
            row['action_gate'] = derived['action_gate'][row['rule_id']]
            conflict = next(x['score'] for x in derived['conflict_scores'] if x['rule_id']==row['rule_id'])
            row['prior_competition_weight'] = round(row['action_gate']*float(conflict),6)
        cf_is_app = next(a['status']=='applicable' for a in derived['applicability'] if a['rule_id']==rid)
        if cf_is_app == is_app:
            continue
        original_context, cf_context = _probe_context(cached,rid), _probe_context(derived,rid)
        pairs.append({'pair_id': f"cached_gid3:{rid}", 'rule_id': rid, 'rule': rule['text'],
            'action': rule['action'], 'conditions': rule['conditions'],
            'applicable_patient': patient if is_app else cf_patient,
            'inapplicable_patient': cf_patient if is_app else patient,
            'applicable_module1_context': original_context if is_app else cf_context,
            'inapplicable_module1_context': cf_context if is_app else original_context,
            'provenance': {'dataset':'MEGA-CDP', 'split':sample['input_source']['split'],
                'guideline_id':3, 'filename':sample['input_source']['filename'],
                'cache_path':str(cache_path), 'cache_sha256':hashlib.sha256(cache_path.read_bytes()).hexdigest(),
                'source_patient_cache_path':str(smoke_path), 'source_patient_cache_sha256':hashlib.sha256(smoke_path.read_bytes()).hexdigest(),
                'original_patient':patient, 'source_input':sample['input_source'],
                'original_rule_id':rid, 'original_applicability':apps[rid]['status'],
                'original_action_gate':cached['action_gate'][rid],
                'synthetic_patient':False, 'counterfactual_side':'inapplicable' if is_app else 'applicable',
                'flipped_condition_id':target['condition_id'], 'flipped_underlying_condition':underlying,
                'replaced_evidence':evidence, 'replacement_evidence':replacement,
                'construction':'exact cached evidence replacement plus automatic cached-logic recomputation',
                'counterfactual_gate_caveat':'fixed original relevance/conflict and NLI confidence; not rerun module one',
                'reference_answers_used':False, 'module1_calls_during_localization':0}})
        if len(pairs)>=limit:
            break
    if len(pairs)<limit:
        raise ValueError(f'Only {len(pairs)} usable cached contrasts, requested {limit}')
    return pairs
