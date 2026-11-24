"""Evaluate the V3 empty-event E1 protection on the archived 80 -> 20 split."""
from collections import defaultdict
import csv
import math
import time

import numpy as np

from experiments.aster_cr_retrain_paired_v2_20261111 import evaluation as ev
from .s2_events import E1, TRAINING, OUTPUT, rows, read, write_json, write_csv

OUT = OUTPUT.parent/'s2_development'
STRUCTURE = TRAINING.parents[2]/'aster-table2-20261123/reports/aster_table2_20261123/dev_pair_metrics.csv'


def budget_scale(maximum, delta):
    return min(1., delta / maximum) if maximum > 0 else 1.


def protected_score(s, u, q):
    if q is None:  # Empty training events. V3 specifies an inactive operator.
        return u
    if s == 0:
        return min(0., u)
    if abs(s) < q:
        return u
    return math.copysign(max(math.copysign(1., s)*u, 0.1*abs(s)), s)


def table(path):
    with path.open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def count_transitions(records):
    b = sum(not a and v for a, v in records)
    h = sum(a and not v for a, v in records)
    k = sum(a and v for a, v in records)
    w = len(records)-b-h-k
    return dict(B=b,H=h,kept_correct=k,kept_wrong=w,
                B_over_wrong=b/(b+w) if b+w else None,
                H_over_correct=h/(h+k) if h+k else None,
                net=(b-h)/len(records) if records else None)


def main():
    started, cpu = time.perf_counter(), time.process_time()
    OUT.mkdir(exist_ok=True, parents=True)
    config = read(OUTPUT/'e1_train80_parameters.json')
    if config['changed'] or config['q'] is not None or config['gate_eligible']:
        raise ValueError('This evaluation reuses paths only for the archived empty-event stop.')
    dev = E1/'evaluation/dev'
    training = read(TRAINING/'models/E1/full/m3.json')
    delta, maximum = training['delta'], training['max_CF_correction']
    a = budget_scale(maximum, delta)
    eval_rows = list(rows(dev/'predictions.jsonl.gz'))
    dev_sources = sorted({r['source_id'] for r in eval_rows})
    fit_sources = read(TRAINING/'models/E1/full/model.json')['train_sources']
    assert len(dev_sources)==20 and len(fit_sources)==80 and set(dev_sources).isdisjoint(fit_sources)
    score, units, changes = {}, {}, []
    grouped = defaultdict(list)
    subset = read(E1/'evaluation/dev/completion.json')
    assert subset['complete']
    # Archived #85 structure labels apply to both A/P endpoints (same predicate).
    types = {r['pair_id']:r['node_type'] for r in table(STRUCTURE) if r['kind'] in ('A','P')}
    node_types = defaultdict(set)
    pairs = [p for p in rows(dev/'pair_predictions.jsonl.gz') if p['method']=='raw']
    for p in pairs:
        if p['pair_id'] in types:
            for side in ('left','right'):
                node_types[p[side]].add(types[p['pair_id']])
    max_score_gap = 0.
    for r in eval_rows:
        uid,s,u = r['view_id'], r['scores']['raw'], r['scores']['E1_pair_only']
        value = s + a*(protected_score(s, u, config['q'])-s)
        if (value > 0) != (u > 0):
            raise ValueError(f'Archived path linkage invalid at {uid}')
        max_score_gap = max(max_score_gap,abs(value-u))
        score[uid] = value
        units[uid] = s
        label = r['label']
        if label is None:
            continue
        rc, cc = (s>0)==label, (value>0)==label
        if (s>0)!=(value>0):
            changes.append(dict(uid=uid,source=r['source_id'],s=s,u=u,protected=value,label=label))
        levels = ['all_labeled']
        if r['in_evaluation_views']:
            labels = node_types[uid]
            structure = next(iter(labels)) if len(labels)==1 else '未标注'
            levels += ['evaluation_nodes','structure_'+structure]
        if r['natural']:
            levels.append('natural_nodes')
        for level in levels:
            grouped[level,r['source_id']].append((rc,cc))
    # Every dynamic node decision is identical to E1. Reuse both original dynamic
    # traces without consulting reference branches to choose candidate predictions.
    paths = defaultdict(dict)
    for r in rows(dev/'path_predictions.jsonl.gz'):
        if r['method'] in ('raw','E1_pair_only'):
            key = r['source_id'],r['case_id'],r['variant'],r['interference']
            paths[key][r['method']] = r['correct']
    for (source,_,_,interf), methods in paths.items():
        grouped['interference_path' if interf else 'natural_path',source].append(
            (methods['raw'], methods['E1_pair_only']))
    pair_rows = ev.observations([], pairs, {'raw': units,'protected': score},
                                {name:{k:int(v>0) for k,v in x.items()}
                                 for name,x in [('raw',units),('protected',score)]})
    novel = [r for r in pair_rows if r['method']=='protected']
    # Existing bootstrap intervals can be reused for numerically equivalent metrics.
    # Compute exact post-scaling point estimates from the adjusted pair scores.
    estimates = []
    for group in ('AP','CF','F0'):
        rr=[p for p in novel if p['group']==group]
        d=np.array([p['difference'] for p in rr])
        raw=np.array([p['raw_difference'] for p in rr])
        corr=d-raw
        direction=[int((p['right_label']-p['left_label'])*p['difference']>0) for p in rr if p['right_label']!=p['left_label']]
        estimates.append(dict(group=group,n=len(rr),sources=len({p['source_id'] for p in rr}),
                              raw_variance=float(np.var(raw)),adjusted_variance=float(np.var(d)),
                              variance_reduction=float(1-np.var(d)/np.var(raw)),correction_rms=float(np.sqrt(np.mean(corr*corr))),
                              both_correct=sum(p['paired_correct'] for p in rr),direction_correct=sum(direction) if direction else None,
                              direction_total=len(direction),
                              over_budget=sum(abs(x)>delta+1e-9 for x in corr) if group=="CF" else None,
                              max_abs_correction=float(max(abs(corr)))))
    write_csv(OUT/'pair_metrics.csv', estimates)
    # Historical CIs already cover both reference arms. New endpoints get common
    # source draws only for conditional repair/harm risks and signed direction.
    boot = ev.SourceBootstrap(dev_sources)
    sources, pooled, intervals = [], [], []
    for level in sorted({k[0] for k in grouped}):
        available={source:grouped[level,source] for source in dev_sources if (level,source) in grouped}
        total=[p for pairs_in_source in available.values() for p in pairs_in_source]
        pooled.append(dict(level=level, sources=len(available),n=len(total),**count_transitions(total)))
        aa, cc, ids = [], [], []
        for source,records in available.items():
            sources.append(dict(level=level,source=source,n=len(records),**count_transitions(records)))
            aa.extend(r[0] for r in records)
            cc.extend(r[1] for r in records)
            ids.extend([source]*len(records))
        aa = np.array(aa)
        cc = np.array(cc)
        for name,num,den in [('B_over_wrong',~aa & cc,~aa),('H_over_correct',aa & ~cc,aa)]:
            intervals.append(dict(level=level, metric=name, **boot.summary(ev.ratio(boot.sums(num,ids),boot.sums(den,ids)))))
    cf=[p for p in novel if p['group']=='CF']
    for name,key in [('direction_accuracy',lambda p:p['difference']),('raw_direction_accuracy',lambda p:p['raw_difference'])]:
        vv=[(p['right_label']-p['left_label'])*key(p)>0 for p in cf]
        intervals.append(dict(level='CF',metric=name,**boot.summary(boot.mean(vv,[p['source_id'] for p in cf]))))
    write_csv(OUT/'pooled_transitions.csv', pooled)
    write_csv(OUT/'source_transitions.csv', sources)
    write_csv(OUT/'new_intervals.csv', intervals)
    if changes:
        write_csv(OUT/'changed_labeled_nodes.csv', changes)
    archive_ci=[r for r in table(dev/'source_bootstrap.csv') if r['comparison'] in
                ('raw','E1_pair_only','E1_pair_only minus raw')]
    write_csv(OUT/'reused_intervals.csv',archive_ci)
    write_json(OUT/'status.json',dict(status='complete',candidate='E1_pair_only',q='+infinity',
               training_cf_pairs=training['CF_pairs'],delta=delta,training_M=maximum,
               M_source='models/E1/full/m3.json:max_CF_correction; protected operator is identity',
               a=a,training_M_after=a*maximum,protected_vs_same_scale_max_abs=0.,
               protected_vs_archived_max_abs=max_score_gap,development_inputs=len(score),
               protected_nodes=0,blocked_benefit=0,blocked_harm=0,changed_vs_unprotected=0,
               reused_paths_per_method=len(paths),adjusted_branch_disagreements=0,
               supervised_gate='skipped',joint='skipped',same_scale_control='identical to protected',
               new_path_replays=0,backbone_forwards=0,gpu_hours=0,bootstrap_draws=2000,
               seed=20261111,wall_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu))


if __name__=='__main__':
    main()
