"""Issue #70: one full-complement variant, existing inputs and frozen folds only."""
import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from experiments.aster_cr_retrain_paired_v2_20261111 import evaluation as old
from experiments.aster_cr_retrain_paired_v2_20261111.common import load_records, RuntimeFeatures
from experiments.aster_cr_retrain_paired_v2_20261111.training import pair_weights, WEIGHT_TOTAL
from .solver import DELTA, GRID, orthogonal_signal, residual, calibrate, solve, disjoint

DEFAULT = '/home/gcx/aster-cr-retrain-paired-v2-20261111/reports/aster_cr_retrain_paired_v2_20261111'


def read(p):
    return json.loads(Path(p).read_text())


def write(p, v):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def rows(p):
    p = Path(p)
    with (gzip.open(p, 'rt') if p.suffix == '.gz' else p.open()) as f:
        return [json.loads(s) for s in f if s.strip()]


def lines(p, rr):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, 'wt', encoding='utf-8') as f:
        for r in rr: f.write(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n')


def csvwrite(p, rr):
    rr = list(rr)
    if not rr: return
    with Path(p).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rr for k in r)), lineterminator='\n')
        w.writeheader(); w.writerows(rr)


class Store:
    def __init__(self, root):
        self.records = load_records(root)
        self.runtime = RuntimeFeatures(root)

    def ell(self, key): return float(self.records[key]['ell'])

    def heads(self, key, selected):
        record = self.records[key]
        a = self.runtime.selected(key, record, selected)
        if a is not None: return a
        with np.load(record['feature'], allow_pickle=False) as f:
            h = f['heads']; h = h.reshape(-1, h.shape[-1])
            if 'head_ids' in f and len(h) != 1152:
                lookup = {int(k):i for i,k in enumerate(f['head_ids'])}
                h = h[[lookup[int(k)] for k in selected]]
            else: h = h[selected]
            result = h.astype(np.float64).reshape(-1)
        if not np.isfinite(result).all(): raise ValueError('Missing selected heads: '+key)
        return result


def split_inner(sources, seed):
    ordered = np.array(sorted(sources))
    np.random.default_rng(seed).shuffle(ordered)
    ndev = 16
    return sorted(ordered[ndev:].tolist()), sorted(ordered[:ndev].tolist())


def selection_key(r):
    # Evidence protection first: budget exceedance, direction, distance from ER=1,
    # CF correction RMS; then AP RMS, then the larger lambda for exact ties.
    return (r['over_budget_rate'], r['direction_degradation_rate'], abs(r['ER']-1),
            r['CF_correction_RMS'], r['AP_RMS'], -r['lambda'])


def development_metrics(pp, diff, ell, w, lam):
    ap = np.array([p['kind'] in ('A','P') for p in pp])
    cf = ~ap
    y = np.array([ell[p['right']]-ell[p['left']] for p in pp])
    s = np.array([p['right_label']-p['left_label'] for p in pp])
    correction = diff@w; corrected = y+correction
    denominator = np.mean(s[cf]*y[cf])
    if abs(denominator)<1e-12: raise ValueError('Undefined development ER')
    return dict(lambda_value=lam, **{'lambda':lam},
                over_budget_rate=float(np.mean(np.abs(correction[cf])>DELTA+1e-9)),
                direction_degradation_rate=float(np.mean((s[cf]*y[cf]>0)&(s[cf]*corrected[cf]<=0))),
                ER=float(np.mean(s[cf]*corrected[cf])/denominator),
                CF_correction_RMS=float(np.sqrt(np.mean(correction[cf]**2))),
                AP_RMS=float(np.sqrt(np.mean(corrected[ap]**2))))


def fit_model(name, sources, heldout, pairs, natural, root, out, store):
    dest = out/'models'/name; dest.mkdir(parents=True, exist_ok=True)
    if (dest/'done.json').exists():
        return dict(np.load(dest/'model.npz')), read(dest/'done.json')
    with np.load(root/'models/E1'/name/'m2.npz') as f:
        support = {k:f[k] for k in ('heads','mu','scale','S')}
    support['S'] = orthogonal_signal(support['S'])
    allowed = set(sources)
    ps = [p for p in pairs if p['source_id'] in allowed]
    nv = [v for v in natural if v['source_id'] in allowed and v.get('variant','full')=='full']
    ids = sorted({p[k] for p in ps for k in ('left','right')} | {v['view_id'] for v in nv})
    index = {k:i for i,k in enumerate(ids)}
    raw = np.stack([store.heads(k,support['heads']) for k in ids])
    r = residual((raw-support['mu'])/support['scale'],support['S']); del raw
    ell = {k:store.ell(k) for k in ids}
    diff = r[[index[p['right']] for p in ps]]-r[[index[p['left']] for p in ps]]
    inner_fit, inner_dev = split_inner(sources,20261116+int(name[-1]) if name!='full' else 20261116)
    disjoint(inner_fit, inner_dev, heldout)
    write(dest/'sources.json',dict(fit=sources,inner_fit=inner_fit,inner_dev=inner_dev,heldout=heldout,
          representation_sources=sources,selection_scope='M3 conditional on archived outer-fit M1/M2'))
    def fit(which, lam, tag):
        indices = [i for i,p in enumerate(ps) if p['source_id'] in set(which)]
        ap = [i for i in indices if ps[i]['kind'] in ('A','P')]
        cf = [i for i in indices if ps[i]['kind'] in ('C','F')]
        y = np.array([ell[ps[i]['right']]-ell[ps[i]['left']] for i in ap])
        w, diag, corr, nu = solve(diff[ap], y, pair_weights([ps[i] for i in ap],'E1',WEIGHT_TOTAL),diff[cf],lam)
        write(dest/(tag+'_solver.json'),diag)
        csvwrite(dest/(tag+'_constraints.csv'),[dict(pair_id=ps[i]['pair_id'],source_id=ps[i]['source_id'],
                 correction=float(c),abs_excess=max(0.,abs(float(c))-DELTA),dual_multiplier=float(n))
                 for i,c,n in zip(cf,corr,nu)])
        np.savez_compressed(dest/(tag+'_weights.npz'),w=w)
        return w,diag
    attempted=[]
    dev_i=[i for i,p in enumerate(ps) if p['source_id'] in set(inner_dev)]
    for lam in GRID:
        w,diag=fit(inner_fit,lam,'lambda_'+str(lam))
        attempted.append(development_metrics([ps[i] for i in dev_i],diff[dev_i],ell,w,lam))
        write(dest/'selection.json',dict(grid=GRID,trials=attempted,rule='lexicographic: CF budget exceedance, direction degradation, abs(ER-1), CF correction RMS, AP RMS, -lambda'))
    chosen=min(attempted,key=selection_key)['lambda']
    w,diag=fit(sources,chosen,'final')
    nc=np.array([r[index[v['view_id']]]@w for v in nv])
    center,b=calibrate(np.array([ell[v['view_id']] for v in nv]),nc,np.array([v['label'] for v in nv]))
    # Numerically ensure the saved linear score acts only in S complement.
    w=residual(w,support['S'])
    model=dict(**support,w=w,center=np.array(center),b=np.array(b),regularization=np.array(chosen))
    np.savez_compressed(dest/'model.npz',**model)
    done=dict(selected_lambda=chosen,solver=diag,calibration_nodes=len(nv),center=center,b=b,
              S_orthogonality_inf=float(np.max(np.abs(support['S'].T@w))),
              selected_reason='minimum predeclared evidence-first lexicographic development key')
    write(dest/'done.json',done)
    return model,done


def archive(root, scope, filename):
    p=root/'evaluation'/scope/(filename+'.jsonl')
    if not p.exists(): p=root/'predictions'/scope/(filename+'.jsonl.gz')
    return rows(p)


def score(model, ids, store):
    # The orthogonal projection is self-adjoint; r(x)'w=z(x)'w for w in S-perp.
    alpha=model['w']/model['scale']
    offset=float(model['b']-model['center']-model['mu']@alpha)
    def one(key): return key,store.ell(key)+float(store.heads(key,model['heads'])@alpha)+offset
    with ThreadPoolExecutor(max_workers=8) as pool:
        return dict(pool.map(one,sorted(set(ids)),chunksize=32))


def evaluate(name, sources, model, pp, nv, pv, cases, root, out, store, archive_scope):
    dest=out/'evaluation'/name; dest.mkdir(parents=True,exist_ok=True)
    if (dest/'done.json').exists(): return
    allow=set(sources)
    pp=[p for p in pp if p['source_id'] in allow]
    nv=[v for v in nv if v['source_id'] in allow]
    pv=[v for v in pv if v['source_id'] in allow]
    cases=[c for c in cases if c['source_id'] in allow]
    views={v['view_id']:v for v in nv}
    for p in pp:
        for side in ('left','right'):
            views.setdefault(p[side],dict(view_id=p[side],source_id=p['source_id'],label=p[side+'_label'],original=False))
    ids=set(views)|{v['view_id'] for v in pv}
    scores=score(model,ids,store)
    raw={key:store.ell(key) for key in ids}
    all_scores={'raw':raw,'full_complement':scores}
    decisions,_=old.decisions_for(list(views.values()),all_scores)
    new_pairs=[r for r in old.observations(list(views.values()),pp,all_scores,decisions) if r['method']=='full_complement']
    # Archived paired predictions are subset by IDs; never rerun the old scorer.
    pair_ids={p['pair_id'] for p in pp}
    saved_pairs=[r for r in archive(root,archive_scope,'pair_predictions') if r['method'] in ('E1','raw') and r['pair_id'] in pair_ids]
    if {r['pair_id'] for r in saved_pairs if r['method']=='E1'} != pair_ids:
        raise ValueError('Archived E1 pair predictions do not cover requested Dataset A subset')
    saved_paths=[r for r in archive(root,archive_scope,'path_predictions') if r['method'] in ('E1','raw')]
    new_paths=old.path_predictions(cases,pv,{'full_complement':scores})
    if {(r['case_id'],r['variant']) for r in saved_paths if r['method']=='E1'} != {(r['case_id'],r['variant']) for r in new_paths}:
        raise ValueError('Archived and new path denominators differ')
    saved_nodes={r['view_id']:r for r in archive(root,archive_scope,'predictions')}
    nodes=[]
    for v in nv:
        if v.get('label') is None: continue
        k=v['view_id']
        for method,value in [('full_complement',scores[k]),('raw',saved_nodes[k]['scores']['raw']),('E1',saved_nodes[k]['scores']['E1'])]:
            nodes.append(dict(view_id=k,source_id=v['source_id'],method=method,label=v['label'],score=float(value),correct=int(value>0)==v['label']))
    for rr in (new_pairs,saved_pairs,new_paths,saved_paths,nodes):
        for r in rr: r['fold']=name
    lines(dest/'pairs.jsonl.gz',new_pairs+saved_pairs)
    lines(dest/'paths.jsonl.gz',new_paths+saved_paths)
    lines(dest/'nodes.jsonl.gz',nodes)
    lines(dest/'scores.jsonl.gz',[dict(view_id=k,score=scores[k],raw=raw[k]) for k in sorted(ids)])
    write(dest/'done.json',dict(inputs=len(ids),pairs=len(pp),sources=len(allow),new_paths=len(new_paths),
                              archive_scope=archive_scope,old_method='read archived predictions; no old scoring'))


def summarize(out, names, label):
    pp=[r for name in names for r in rows(out/'evaluation'/name/'pairs.jsonl.gz')]
    paths=[r for name in names for r in rows(out/'evaluation'/name/'paths.jsonl.gz')]
    nodes=[r for name in names for r in rows(out/'evaluation'/name/'nodes.jsonl.gz')]
    boot=old.SourceBootstrap([r['source_id'] for r in pp])
    summary={}; arrays={}; source_rows=[]
    for method in ('raw','E1','full_complement'):
        mm,cc,ss=old.pair_statistics([r for r in pp if r['method']==method],boot)
        source_rows.extend(ss)
        for target,group in [('natural_node',nodes),('natural_path',[r for r in paths if not r['interference']]),('interference_path',[r for r in paths if r['interference']])]:
            rr=[r for r in group if r['method']==method]
            mm[target+'_accuracy']=boot.mean([r['correct'] for r in rr],[r['source_id'] for r in rr])
            cc[target+'_count']=len(rr);cc[target+'_correct']=sum(r['correct'] for r in rr)
        summary[method]=dict(counts=cc,metrics={k:boot.summary(v) for k,v in mm.items()})
        arrays[method]=mm
    contrasts={k:boot.summary(v-arrays['E1'][k]) for k,v in arrays['full_complement'].items()}
    csvwrite(out/(label+'_source_metrics.csv'),source_rows)
    write(out/(label+'_summary.json'),dict(models=summary,full_minus_archived_E1=contrasts,bootstrap_draws=2000,
           bootstrap_unit='source; paired draws; concatenate folds before computing ratios'))
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cache-root',default=DEFAULT)
    parser.add_argument('--out',default='reports/aster_full_complement_70')
    args=parser.parse_args();root=Path(args.cache_root);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();folds=read(root/'folds.json');config=read(root/'configuration.json')
    pairs=old.data_rows(root,'E1_pairs.jsonl');natural=old.data_rows(root,'natural_views.jsonl')
    if len(pairs)!=2540 or config['delta']!=DELTA or len(folds['all_sources'])!=80:
        raise ValueError('Protocol conflict: Dataset A, budget or source count')
    for f in folds['folds']:
        disjoint(f['train_sources'],f['validation_sources'])
        meta=read(root/'models/E1'/('fold'+str(f['fold']))/'model.json')
        if set(meta['train_sources'])!=set(f['train_sources']) or len(f['train_sources'])!=64 or len(f['validation_sources'])!=16:
            raise ValueError('Protocol conflict: archived fold models')
    write(out/'configuration.json',dict(task=70,variant='full evidence orthogonal complement, linear constrained M3',
          cache_root=str(root),base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          delta=DELTA,lambda_grid=GRID,folds=folds,internal_selection='48/16 for CV, 64/16 for full model; conditional on reused M1/M2',
          selection_rule='CF over-budget rate, CF direction degradation, abs(ER-1), CF correction RMS, AP RMS, -lambda',
          GPUs=[],expected_GPU_GiB=0,new_model_forwards=0,expected_minutes=[20,40],
          deduplication='main experiments/reports and server M3 joint study checked; no same full-complement hard-budget run found',
          dependencies=dict(numpy=np.__version__,osqp=__import__('osqp').__version__,scipy=__import__('scipy').__version__)))
    store=Store(root)
    path_views=[r for n in ('path_features.jsonl','path_interference_views.jsonl') for r in old.data_rows(root,n)]
    cases=rows(Path(config['historical_root'])/'data/path_cases.jsonl')
    for f in folds['folds']:
        name='fold'+str(f['fold'])
        write(out/'status.json',dict(state='running',stage='fit',fold=name))
        model,done=fit_model(name,f['train_sources'],f['validation_sources'],pairs,natural,root,out,store)
        write(out/'status.json',dict(state='running',stage='score',fold=name))
        evaluate(name,f['validation_sources'],model,pairs,natural,[v for v in path_views if v['split']=='fit'],
                 [c for c in cases if c['split']=='fit'],root,out,store,'cv_E1_'+name+'_common_frozen_pool')
    summarize(out,['fold'+str(k) for k in range(1,6)],'cv')
    test_pairs=[r for r in old.data_rows(root,'historical_pairs.jsonl') if r['split']=='test']
    test_views=[r for r in old.data_rows(root,'historical_views.jsonl') if r['split']=='test' and r.get('original')]
    test_sources=sorted({r['source_id'] for r in test_pairs})
    disjoint(folds['all_sources'],test_sources)
    if len(test_sources)!=40: raise ValueError('Protocol conflict: controlled test source count')
    write(out/'status.json',dict(state='running',stage='full_fit'))
    model,done=fit_model('full',folds['all_sources'],test_sources,pairs,natural,root,out,store)
    write(out/'status.json',dict(state='running',stage='test_score'))
    evaluate('test',test_sources,model,test_pairs,test_views,[v for v in path_views if v['split']=='test'],
             [c for c in cases if c['split']=='test'],root,out,store,'test')
    summarize(out,['test'],'test')
    import resource
    write(out/'status.json',dict(state='complete',real_experiment=True,new_model_forwards=0,GPU_GiB=0,
                               seconds=time.monotonic()-start,peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2))
    from .report import report
    report(out)


if __name__=='__main__':
    try: main()
    except Exception as e:
        import sys,traceback
        p=Path(sys.argv[sys.argv.index('--out')+1]) if '--out' in sys.argv else Path('reports/aster_full_complement_70')
        write(p/'failure.json',dict(error_type=type(e).__name__,message=str(e)))
        raise
