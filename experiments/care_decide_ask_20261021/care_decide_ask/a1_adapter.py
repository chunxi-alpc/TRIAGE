"""Frozen A1 interoperability. Hard predictions do NOT imply calibrated scores."""
from __future__ import annotations


def frozen_a1_hard_sets(prediction,expected_ids):
    """Strictly non-calibrated hard-label policy. No probability or risk claim."""
    if not isinstance(prediction,dict) or prediction.get('arm') not in ('A1',None):
        raise ValueError('expected frozen A1 prediction')
    if prediction.get('arm') is None and prediction.get('signal_source') != 'care.module1.semantic_a1.frozen':
        raise ValueError('frozen A1 provenance required')
    # Frozen exporter treats a failed whole-case parse as unavailable.
    # Do not mine apparent sub-rules from a failed parsed case.
    rules=prediction.get('rules',[]) if prediction.get('status') in ('ok',None) else []
    if not isinstance(rules,list):rules=[]
    if len(set(expected_ids))!=len(expected_ids):raise ValueError('duplicate expected rule IDs')
    states={};counts={}
    for item in rules:
        if not isinstance(item,dict):continue
        rid=item.get('rule_id')
        if rid not in expected_ids:raise ValueError('unexpected rule ID in A1 prediction')
        counts[rid]=counts.get(rid,0)+1
        states[rid]=item.get('state') if counts[rid]==1 else None
    out={}
    for rid in expected_ids:
        v=states.get(rid)
        out[rid]=['T'] if v=='T' else ['F'] if v=='F' else ['T','F','U']
    return out


def require_a1_probabilities(data,expected_ids):
    """Never manufacture class probabilities from A1's JSON hard states."""
    if not isinstance(data,dict) or 'rule_probs' not in data:
        raise ValueError('FROZEN_A1_NO_PROBABILITIES: new separately validated scoring is required')
    meta=data.get('score_provenance',{})
    if not isinstance(meta,dict) or meta.get('frozen_before_calibration') is not True or not meta.get('scoring_model_id'):
        raise ValueError('independent validated scoring provenance required')
    p=data['rule_probs']
    if set(p)!=set(expected_ids):raise ValueError('scores must cover every original rule')
    from .calibration import checked_probs
    for item in p.values():checked_probs(item)
    return p
