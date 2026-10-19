"""Report all assigned cases, including failed and no-op model runs."""
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path


def read_jsonl(path):
    path = Path(path)
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def avg(values):
    values = [x for x in values if x is not None]
    return sum(values) / len(values) if values else None


def candidate_margin(row, gold):
    if row is None:
        return None
    if isinstance(row.get('margin'), (float, int)):
        return row['margin']
    logits = row.get('candidate_logits')
    if isinstance(logits, dict):
        logits = [logits.get(letter) for letter in 'ABCD']
    if not logits or len(logits) != 4 or any(x is None for x in logits):
        return None
    i = 'ABCD'.index(gold)
    return logits[i] - max(x for j, x in enumerate(logits) if j != i)


def score_cases(cases, records, baseline=()):
    records = list(records)
    index = {r['case_id']: r for r in records}
    if len(index) != len(records):
        raise ValueError('Duplicate case outputs in one arm')
    base = {r['case_id']: r for r in baseline}
    correct, bcorrect, errors, margins, mdeltas = {}, {}, [], [], []
    for case in cases:
        cid = case['case_id']
        row = index.get(cid, {})
        prediction = row.get('prediction')
        correct[cid] = prediction == case['gold_answer']
        if cid in base:
            bcorrect[cid] = base[cid].get('prediction') == case['gold_answer']
        margin = candidate_margin(row, case['gold_answer'])
        bmargin = candidate_margin(base.get(cid), case['gold_answer'])
        margins.append(margin)
        mdeltas.append(None if margin is None or bmargin is None else margin - bmargin)
        if not correct[cid] or (cid in bcorrect and correct[cid] != bcorrect[cid]):
            errors.append({'case_id':cid,'family_id':case['family_id'],'variant':case['variant'],
                           'reference_state':case['reference_state'],'gold_answer':case['gold_answer'],
                           'prediction':prediction,'baseline_prediction':base.get(cid,{}).get('prediction'),
                           'technical_failure_or_missing':prediction not in 'ABCD' if isinstance(prediction,str) else True,
                           'corrected':correct[cid] and not bcorrect.get(cid,True),
                           'damaged':not correct[cid] and bcorrect.get(cid,False),
                           'margin':margin,'margin_delta':mdeltas[-1]})
    def subgroup(selected):
        pairs = defaultdict(list)
        for c in selected:
            if c.get('pair_eligible'):
                pairs[c['pair_id']].append(c)
        return {'cases':len(selected),'correct':sum(correct[c['case_id']] for c in selected),
                'accuracy':avg([correct[c['case_id']] for c in selected]),
                'pairs':len(pairs),
                'pair_correct':sum(len(pair)==2 and all(correct[c['case_id']] for c in pair) for pair in pairs.values())}
    result = subgroup(cases)
    result.update({'actual_outputs':sum(index.get(c['case_id'],{}).get('prediction') in ('A','B','C','D') for c in cases),
                   'missing_or_failed':sum(index.get(c['case_id'],{}).get('prediction') not in ('A','B','C','D') for c in cases),
                   'mean_gold_margin':avg(margins),'mean_margin_delta_vs_baseline':avg(mdeltas),
                   'corrected':sum(correct[cid] and not old for cid,old in bcorrect.items()),
                   'damaged':sum(not correct[cid] and old for cid,old in bcorrect.items()),
                   'prediction_changed':sum(index.get(cid,{}).get('prediction')!=r.get('prediction') for cid,r in base.items() if cid in correct),
                   'primary_canonical':subgroup([c for c in cases if c['variant']=='canonical' and c.get('pair_eligible')]),
                   'by_variant':{v:subgroup([c for c in cases if c['variant']==v]) for v in sorted({c['variant'] for c in cases})},
                   'by_state':{s:subgroup([c for c in cases if c['reference_state']==s]) for s in ('T','F','U')},
                   'by_family':{f:subgroup([c for c in cases if c['family_id']==f]) for f in sorted({c['family_id'] for c in cases})}})
    result['net_corrected'] = result['corrected'] - result['damaged']
    return result, errors
