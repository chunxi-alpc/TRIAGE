"""Case-joint split conformal over all rule labels (T/F/U).

Guarantee under exchangeability of cases with fixed score model: marginal
P(all truth labels included in prediction sets) >= 1-alpha. This is NOT a
per-decision conditional error guarantee and does not compensate for a wrong
policy AST. Frozen A1 supplies hard labels ONLY; new score extraction is needed.
"""
from __future__ import annotations
from math import ceil,inf,isfinite
from collections import defaultdict

LABELS=('T','F','U')


def checked_probs(scores):
    if scores is None:return None
    if not isinstance(scores,dict) or set(scores)!=set(LABELS):
        raise ValueError('complete T/F/U distribution required for each scored rule')
    if any(not isinstance(scores[v],(int,float)) or not isfinite(scores[v]) or
           scores[v]<0 or scores[v]>1 for v in LABELS):
        raise ValueError('invalid state probabilities')
    if abs(sum(scores.values())-1)>1e-6:raise ValueError('state probabilities must sum to one')
    return scores


def case_nonconformity(row):
    truth=row.get('truth');probs=row.get('probs')
    if not isinstance(truth,dict) or not isinstance(probs,dict) or set(truth)!=set(probs) or not truth:
        raise ValueError('truth/probs need identical nonempty rule IDs')
    scores=[]
    for rule_id,label in truth.items():
        if label not in LABELS:raise ValueError('unknown truth state')
        model=checked_probs(probs[rule_id])
        # Missing score => deliberately full prediction set for this rule;
        # its nonconformity is zero and it cannot cause a false exclusion.
        scores.append(0. if model is None else 1.-model[label])
    return max(scores)


def calibrate_joint(rows,alpha,unit='case'):
    """Simultaneous rule coverage per case or per independent *family*.

    For unit='family', first aggregate maximum nonconformity over all cases
    in a family. Assumes exchangeability of full families, not merely of rows.
    """
    if not 0<alpha<1:raise ValueError('alpha strictly between zero and one')
    if unit not in ('case','family'):raise ValueError('unit must be case or family')
    rows=list(rows)
    if not rows:raise ValueError('empty calibration data')
    if unit=='family':
        grouped=defaultdict(list)
        for r in rows:
            fid=r.get('family_id')
            if not isinstance(fid,str) or not fid:raise ValueError('family_id required')
            grouped[fid].append(case_nonconformity(r))
        scores=sorted(max(group) for group in grouped.values())
    else:
        scores=sorted(case_nonconformity(r) for r in rows)
    n=len(scores);k=ceil((n+1)*(1-alpha))
    q=scores[k-1] if k<=n else inf
    return {'schema':'care.m2.calibration.v1','alpha':alpha,'n_units':n,
            'n_cases':len(rows),'unit':unit,'quantile_index_one_based':k,
            'threshold':q if isfinite(q) else None,
            'infinite_threshold':not isfinite(q),
            'guarantee':('marginal_all_rules_per_case' if unit=='case'
                         else 'marginal_all_rules_in_new_exchangeable_family'),
            'conditional_decision_error_bound':False,
            'assumes_exchangeable_units':True,
            'independence_unit':unit}

def make_prediction_sets(distributions,calibration):
    if calibration.get('schema')!='care.m2.calibration.v1':
        raise ValueError('unrecognized calibration')
    q=inf if calibration.get('infinite_threshold') else calibration['threshold']
    if not isinstance(q,(int,float)) or q<0:raise ValueError('invalid q')
    if not isinstance(distributions,dict) or not distributions:
        raise ValueError('rule distribution dictionary required')
    out={}
    for rid,p in distributions.items():
        p=checked_probs(p)
        out[rid]=list(LABELS if p is None else [label for label in LABELS if 1-p[label]<=q+1e-12])
    return out
