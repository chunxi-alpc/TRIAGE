"""Training-only N5 contrasts from frozen predicted states and original pair IDs."""
from collections import defaultdict, Counter

FIT_CONTRACT = 'n5_same_pair_rule_predicted_TF_v3'

def matched_pairs(rows):
    if not rows or any(r['task'] != 'N5' or r['split'] != 'train' for r in rows):
        raise ValueError('N5 fit requires nonempty train-only inputs')
    groups = defaultdict(list)
    observations = []
    seen = set()
    for r in sorted(rows, key=lambda x: x['case_id']):
        for s in sorted(r['signals'], key=lambda x: x['rule_id']):
            oid = (r['case_id'], s['rule_id'])
            if oid in seen:
                raise ValueError('duplicate case/rule in fit')
            seen.add(oid)
            if s['signal'] != {'T': 1, 'F': -1, 'U': 0, None: None}[s['state']]:
                raise ValueError('inconsistent frozen predicted state/signal')
            o = {'case_id': r['case_id'], 'family_id': r['family_id'], 'pair_id': r['pair_id'],
                 **{k: s[k] for k in ('rule_id', 'rule_sha256', 'state', 'signal', 'action', 'span')},
                 'source_commit': r['source_commit'], 'a1_method': r['a1_method']}
            observations.append(o)
            if s['signal'] not in (1, -1):
                o['fit_status'] = 'no_TF_signal'
            elif not s['action'] or not s['span']:
                o['fit_status'] = 'no_action_span'
            elif not r['pair_id'] or r['pair_id'] == r['family_id']:
                o['fit_status'] = 'no_original_pair'
            else:
                key = (r['family_id'], r['pair_id'], s['rule_id'], s['rule_sha256'],
                       s['action'], r['source_commit'], r['a1_method'])
                groups[key].append(o)
    pairs = []
    for key, group in sorted(groups.items()):
        p = [o for o in group if o['signal'] == 1]
        n = [o for o in group if o['signal'] == -1]
        if len(group) == 2 and len(p) == len(n) == 1 and p[0]['case_id'] != n[0]['case_id']:
            for o in group:
                o['fit_status'] = 'paired'
            pairs.append({'family_id': key[0], 'pair_id': key[1], 'rule_id': key[2],
                          'positive': p[0], 'negative': n[0]})
        else:
            for o in group:
                o['fit_status'] = 'no_unique_TF_counterpart'
    audit = {'fit_contract': FIT_CONTRACT, 'uses_reference_labels': False,
             'mother_cases': len(rows), 'mother_signals': len(observations),
             'pairs': pairs, 'observations': observations,
             'status_counts': dict(Counter(o['fit_status'] for o in observations)),
             'pairs_per_family': dict(Counter(p['family_id'] for p in pairs)),
             'weighting': 'equal matched pairs; both centroids and covariance use exactly the same observations',
             'evaluation_population_changed': False}
    return pairs, audit
