"""Join archived out-of-fold pair predictions and fill only missing J_zero F0 scalars."""
import csv
import gzip
import time
from pathlib import Path

import numpy as np

from .s2_events import E1, JOINT, ROOT, TRAINING, ZERO, read, rows, write_csv, write_json


OUT = ROOT/'reports/aster_v3_selective_20261124/s2_pairs'
METHODS = ('R0','E1_pair_only','J_zero')


def pairs():
    data = {}
    for n in range(1,6):
        fold = f'fold{n}'
        for pool in ('common','historical'):
            file = E1/'evaluation'/f'cv_{fold}_{pool}'/'pair_predictions.jsonl.gz'
            for r in rows(file):
                if r['method'] not in ('raw','E1_pair_only') or (pool == 'historical' and r['kind'] != 'F0'):
                    continue
                key = fold + '::' + r['pair_id']
                group = r['kind'] if r['kind'] == 'F0' else r['group']
                item = data.setdefault(key,dict(group=group,source=r['source_id'],kind=r['kind'],
                    label=(r['left_label'],r['right_label']), uid=(r['left'],r['right']),fold=fold,delta=r['evidence_budget']))
                method = 'R0' if r['method'] == 'raw' else r['method']
                item[method] = (r['left_score'],r['right_score'])
    return data


def join_j(data):
    needed = {(r['fold'],u) for r in data.values() for u in r['uid']}
    scores = {}
    for r in rows(ZERO/'A_scores.jsonl.gz'):
        fold, uid = r['view_id'].split('::')
        if (fold,uid) in needed:
            scores[fold,uid] = (r['scores']['R0'],r['scores']['J_zero'])
    prior = len(scores)
    extra = OUT/'f0_endpoint_scores.csv'
    reused = {}
    if extra.exists():
        with extra.open() as f:
            reused = {(r['fold'],r['uid']):r for r in csv.DictReader(f)}
    runtime = TRAINING/'runtime'
    index = {u:i for i,u in enumerate(read(runtime/'fit_uids.json'))}
    heads = np.load(runtime/'fit_heads.npy',mmap_mode='r')
    present = np.load(runtime/'fit_present.npy',mmap_mode='r')
    models = {}
    def frozen_readout(fold, uid, raw):
        if fold not in models:
            with np.load(TRAINING/'models/E1'/fold/'m2.npz') as m:
                m2 = {k:m[k] for k in ['mu','scale','N','heads']}
            models[fold] = m2, read(JOINT/'models'/fold/'J.json')
        m2, j = models[fold]
        index_row = index[uid]
        if not int(present[index_row]) & 1:
            raise ValueError(f'Missing archived head activation for {uid}')
        h = heads[index_row]
        flat = h.reshape(-1,h.shape[-1])[m2['heads']].astype(np.float32).reshape(-1)
        trace = ((flat.astype(float)-m2['mu'])/m2['scale']) @ m2['N']
        return raw + float((trace - j['mu']) @ j['w'])

    checks = {}
    for r in data.values():
        fold = r['fold']
        if r['group'] == 'F0' and fold not in checks:
            left = r['uid'][0]
            raw, expected = scores[fold,left]
            gap = abs(frozen_readout(fold,left,raw)-expected)
            if gap > 1e-8:
                raise ValueError(f'Frozen projection score differs from archive in {fold}')
            checks[fold] = dict(fold=fold,uid=left,absolute_difference=gap)
    write_csv(OUT/'f0_projection_checks.csv',list(checks.values()))
    computed = []
    for r in data.values():
        result = []
        for i,uid in enumerate(r['uid']):
            key = (r['fold'],uid)
            raw = r['R0'][i]
            if key not in scores:
                if r['group'] != 'F0':
                    raise ValueError(f'Missing common pair endpoint {key}')
                if key in reused:
                    entry = reused[key]
                    if entry['source'] != r['source'] or float(entry['raw']) != raw:
                        raise ValueError('Cached F0 input differs from current pair')
                    value = float(entry['J_zero'])
                else:
                    fold = r['fold']
                    value = frozen_readout(fold,uid,raw)
                    entry = dict(fold=fold,source=r['source'],uid=uid,raw=raw,J_zero=value)
                    computed.append(entry)
                scores[key] = raw,value
            archived_raw,value = scores[key]
            if abs(archived_raw - raw) > 1e-8:
                raise ValueError(f'Original score conflict at {key}')
            result.append(value)
        r['J_zero'] = result
    if computed:
        write_csv(extra, list(reused.values()) + computed)
    return prior, len(computed), len(reused)


def csv_rows(path):
    with path.open(encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def calculate(data):
    out, rowwise, sources = [],[],[]
    intervals = {}
    for directory, names in [(E1/'evaluation/cv_pooled_common',{'raw':'R0','E1_pair_only':'E1_pair_only'}),
                             (JOINT/'evaluation',{'J_shared':'J_zero'})]:
        for r in csv_rows(directory/'source_bootstrap.csv'):
            method = names.get(r['comparison'])
            if method:
                intervals[method,r['metric']] = r
    for r in csv_rows(E1/'evaluation/cv_pooled_historical/source_bootstrap.csv'):
        method = {'raw':'R0','E1_pair_only':'E1_pair_only'}.get(r['comparison'])
        if method and r['metric'].startswith('f0_'):
            intervals[method,r['metric']] = r
    all_sources = sorted({r['source'] for r in data.values()})
    idx = {s:i for i,s in enumerate(all_sources)}
    n = len(idx)
    weights = np.vstack([np.ones(n), np.random.default_rng(20261111).multinomial(n,np.full(n,1/n),size=2000)])
    for method in METHODS:
        for group in ('AP','CF','F0'):
            selected = [(k,r) for k,r in data.items() if r['group']==group]
            a = np.zeros((n,11))
            for k,r in selected:
                raw_delta = r['R0'][1]-r['R0'][0]
                left,right = r[method]
                delta = right-left
                adjust = delta-raw_delta
                yl,yr = r['label']
                both = int((left>0)==yl and (right>0)==yr)
                relation = int(((left>0)==(right>0)) == (yl==yr))
                direction = int((yr-yl)*delta>0)
                over = int(abs(adjust)>r['delta']+1e-9)
                a[idx[r['source']]] += [1,both,relation,direction,over,raw_delta,delta,raw_delta**2,delta**2,adjust**2,int(yl!=yr)]
                if method=='J_zero':
                    rowwise.append(dict(fold=r['fold'],source_id=r['source'],pair_id=k.split('::',1)[1],group=group,
                                        left_uid=r['uid'][0],right_uid=r['uid'][1],both_correct=both,
                                        relation_correct=relation,direction_correct=direction if yl!=yr else '',over_budget=over))
            for source, values in zip(all_sources,a):
                if values[0]:
                    sources.append(dict(source_id=source,method=method,group=group,n=int(values[0]),both=int(values[1]),
                        relation=int(values[2]),direction=int(values[3]) if group=='CF' else '',above_delta=int(values[4])))
            total = weights @ a
            cnt,both,relation,direct,over,raw_sum,new_sum,raw_sq,new_sq,dc_sq,direct_n = total.T
            with np.errstate(divide='ignore',invalid='ignore'):
                metrics = {'paired_accuracy':both/cnt,'relation_accuracy':relation/cnt,'correction_rms':np.sqrt(dc_sq/cnt)}
                if group=='AP':
                    metrics['vr']=1-(new_sq/cnt-(new_sum/cnt)**2)/(raw_sq/cnt-(raw_sum/cnt)**2)
                if group=='CF':
                    metrics.update(direction_accuracy=direct/direct_n, over_budget_rate=over/cnt)
            for metric,values in metrics.items():
                archived_metric = group.lower()+'_'+metric
                ref = intervals.get((method,archived_metric))
                if method=='J_zero' and metric not in {'vr','correction_rms','over_budget_rate','direction_accuracy'}:
                    ref=None
                if ref:
                    if abs(float(ref['estimate'])-values[0])>1e-8:
                        raise ValueError(f'Pooled archived {archived_metric} differs for {method}')
                    lo,hi = float(ref['low']),float(ref['high'])
                    origin = 'SJ J_shared; verified intercept-invariant' if method=='J_zero' else 'E1 pooled archive'
                else:
                    lo,hi = np.nanquantile(values[1:],[.025,.975])
                    origin = 'paired 2000 source draws'
                numerator = {'paired_accuracy':both[0],'relation_accuracy':relation[0],'direction_accuracy':direct[0],'over_budget_rate':over[0]}.get(metric)
                out.append(dict(method=method,group=group,metric=metric,numerator='' if numerator is None else int(numerator),
                            denominator=int(direct_n[0] if metric=='direction_accuracy' else cnt[0]),sources=int(np.sum(a[:,0]>0)),
                            estimate=values[0],low=lo,high=hi,interval_source=origin))
    with gzip.open(OUT/'j_zero_pair_results.csv.gz','wt',newline='') as stream:
        w=csv.DictWriter(stream, lineterminator="\n", fieldnames=rowwise[0].keys());w.writeheader();w.writerows(rowwise)
    write_csv(OUT/'metrics.csv',out)
    write_csv(OUT/'by_source.csv',sources)


def main():
    start=time.process_time()
    OUT.mkdir(parents=True,exist_ok=True)
    data=pairs()
    prior,computed,reused=join_j(data)
    calculate(data)
    status_file = OUT/'status.json'
    previous = read(status_file) if status_file.exists() else {}
    current_cpu = time.process_time()-start
    cumulative_cpu = previous.get('ledger_cpu',previous.get('cpu_seconds',0)) + current_cpu
    new_v3_scores = computed + reused
    write_json(status_file,dict(status='complete',common_pair_ids=sum(r['group']!='F0' for r in data.values()),
               legacy_fit_f0_pair_ids=sum(r['group']=='F0' for r in data.values()), sources=len({r['source'] for r in data.values()}),
               reused_R0_E1_method_pair_records=2*len(data),reused_J_zero_endpoint_scores=prior,
               newly_computed_J_zero_scalar_scores=new_v3_scores,current_run_scalar_readouts=computed,current_run_reused_V3_scores=reused,
               new_evaluated_J_zero_pairs=len(data), new_backbone_forwards=0,new_fits=0,new_path_replays=0,
               gpu_hours=0,cpu_seconds=current_cpu, bootstrap_draws=2000,seed=20261111,
               ledger_cpu=cumulative_cpu, ledger_reused=f"R0/E1配对输出 {2*len(data):,} 条、J_zero 折外端点分数 {prior:,} 条",
               ledger_new=f"J_zero 新增 F0 标量读出 {new_v3_scores} 个、配对核验 {len(data):,} 对",
               output_endpoints="f0_endpoint_scores.csv", output_by_source="by_source.csv"))


if __name__=='__main__':
    main()
