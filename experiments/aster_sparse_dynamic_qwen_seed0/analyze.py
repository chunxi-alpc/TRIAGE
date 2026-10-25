"""All candidates retained; leave-one-development-tree-out parameter selection."""
import argparse,csv,hashlib,itertools,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'aster_dynamic_fusion_seed0_20261025'))
from balanced_analysis import paired as tree_paired
from sparse_engine import configurations

GRID=[dict(beta=b,t0=t,ts=s) for b,t,s in itertools.product((.5,.75,1.),(1.,2.,4.),(.25,.5,1.))]
VARIANTS=('NC','AC','IC','SW')


def paired(rows,a,b):
    result=tree_paired(rows,a,b)
    a=np.asarray(a);b=np.asarray(b);y=np.array([r['gold_option'] for r in rows])
    result['by_variant']={}
    for v in VARIANTS:
        ix=np.array([r['variant']==v for r in rows])
        fix=int(((a==y)&(b!=y)&ix).sum());harm=int(((a!=y)&(b==y)&ix).sum())
        result['by_variant'][v]=dict(n=int(ix.sum()),corrected=fix,regressed=harm,accuracy_delta=(fix-harm)/int(ix.sum()))
    return result

def softmax(x):
    x=np.asarray(x,dtype=float);e=np.exp(x-x.max());return e/e.sum()
def support_prob(s,t,base):
    s=np.asarray(s,dtype=float)
    if s.sum()<=0:return base.copy()
    s=s/s.sum();return softmax(np.log(np.maximum(s,1e-30))/t)
def predict(row,config,kind):
    base=softmax(np.asarray(row['scores']['original'])/config['t0'])
    support=support_prob(row['option_S'],config['ts'],base)
    beta=config['beta']
    if kind=='dynamic_fusion':
        s=np.asarray(row['option_S'],dtype=float);s=s/s.sum() if s.sum()>0 else base
        ranked=np.sort(s);margin=ranked[-1]-ranked[-2]
        beta=float(np.clip(beta+.25*(2*margin-1),0,1))
    mixed=(1-beta)*base+beta*support
    delta=np.zeros(len(base))
    if kind in ('static_fusion','dynamic_fusion','no_residual','zero_strength'):pass
    elif kind=='legacy_attention_residual':
        delta=np.array(row['scores']['action_bias_1.0_all'])-row['scores']['zero_all']
    elif kind=='numeric_drift':
        delta=np.array(row['scores']['zero_midlate4'])-row['scores']['original']
    else:
        cfg=configurations()[kind];zero='zero_'+cfg['geometry']
        delta=np.array(row['scores'][kind])-row['scores'][zero]
    shift=.25*np.tanh(delta/2.)
    return int(np.argmax(np.log(np.maximum(mixed,1e-30))+shift))

def metrics(rows,preds):
    p=np.asarray(preds);y=np.array([r['gold_option'] for r in rows]);o=np.array([r['answers']['original'] for r in rows])
    def subset(ix):
        ix=np.asarray(ix,dtype=bool);n=int(ix.sum())
        return dict(n=n,correct=int(((p==y)&ix).sum()),accuracy=float((p[ix]==y[ix]).mean()),
            corrected=int(((p==y)&(o!=y)&ix).sum()),regressed=int(((p!=y)&(o==y)&ix).sum()),
            changed=int(((p!=o)&ix).sum()),covered=int(sum(r['covered'] for i,r in enumerate(rows) if ix[i])))
    result=subset(np.ones(len(rows),bool));result['by_variant']={v:subset([r['variant']==v for r in rows]) for v in VARIANTS}
    result['by_tree']={t:subset([r['tree']==t for r in rows]) for t in sorted({r['tree'] for r in rows})}
    return result

def fit(rows,kind):
    matrix=np.array([[predict(r,c,kind) for r in rows] for c in GRID]);y=np.array([r['gold_option'] for r in rows])
    trees=np.array([r['tree'] for r in rows]);groups=sorted(set(trees))
    scores=np.array([[(p[trees==t]==y[trees==t]).mean() for t in groups] for p in matrix])
    oof=np.zeros(len(rows),int);matched=np.zeros(len(rows),int);folds={}
    for j,t in enumerate(groups):
        train=np.delete(scores,j,axis=1).mean(axis=1);best=int(np.argmax(train));ix=trees==t
        oof[ix]=matrix[best,ix]
        matched[ix]=[predict(r,GRID[best],'no_residual') for r in rows if r['tree']==t]
        folds[t]=dict(config_index=best,config=GRID[best],training_macro_accuracy=float(train[best]))
    return dict(metrics=metrics(rows,oof),predictions=oof.tolist(),matched_no_residual=matched.tolist(),
        matched_pair=paired(rows,oof,matched),folds=folds,
        all_candidates=[dict(index=i,config=c,metrics=metrics(rows,matrix[i]),predictions=matrix[i].tolist()) for i,c in enumerate(GRID)])

def load(old,out):
    cached=[json.loads(x) for x in (old/'qwen/predictions.jsonl').read_text().splitlines()]
    new=[];resources={}
    for gpu in (0,1,6):
        folder=out/f'shard{gpu}'
        done=json.loads((folder/'DONE.json').read_text());assert done['complete']
        status=json.loads((folder/'status.json').read_text());assert status['complete']
        chunk=[json.loads(x) for x in (folder/'predictions.jsonl').read_text().splitlines()]
        assert len(chunk)==status['total'];new.extend(chunk)
        resources[str(gpu)]=dict(status=status,manifest=json.loads((folder/'manifest.json').read_text()),gpu_tests=json.loads((folder/'gpu_tests.json').read_text()),sha256=hashlib.sha256((folder/'predictions.jsonl').read_bytes()).hexdigest())
    by={(r['id'],r['variant']):r for r in new};assert len(new)==len(by)==len(cached)==800
    rows=[]
    for oldrow in cached:
        r=by[oldrow['id'],oldrow['variant']]
        for key in ('tree','seed','split','gold_option','p','S','option_S','option_to_action','rule_to_action'):
            assert r[key]==oldrow[key],key
        assert set(r['scores'])==set(configurations())
        assert all(np.isfinite(v).all() for v in r['scores'].values())
        assert all(r['answers'][k]==int(np.argmax(v)) for k,v in r['scores'].items())
        rows.append({**r,'scores':{**oldrow['scores'],**r['scores']},'answers':{**oldrow['answers'],**r['answers']},'cached_seconds':oldrow['seconds']})
    assert len({r['id'] for r in rows})==200
    for pid in {r['id'] for r in rows}:assert {r['variant'] for r in rows if r['id']==pid}==set(VARIANTS)
    return rows,resources

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--old',type=Path,default=Path('/home/gcx/aster_dynamic_review_20261025'))
    ap.add_argument('--out',type=Path,default=Path('/home/gcx/aster_sparse_dynamic_qwen_seed0_20261025'));ap.add_argument('--numeric',type=Path);ap.add_argument('--metadata',type=Path);ap.add_argument('--destination',type=Path,default=HERE);a=ap.parse_args()
    if a.numeric:
        rows=[json.loads(x) for x in a.numeric.read_text().splitlines()]
        resources=json.loads(a.metadata.read_text())['resources']
        assert len(rows)==800 and len({(r['id'],r['variant']) for r in rows})==800
    else:
        rows,resources=load(a.old,a.out)
    # Publish pseudonyms only; no patient text, rules, conditions, or original patient identifiers.
    for r in rows:
        if not a.numeric:r['id']='case_'+hashlib.sha256(('aster-sparse-seed0:'+r['id']).encode()).hexdigest()[:16]
    destination=a.destination;destination.mkdir(parents=True,exist_ok=True)
    published=destination/'numeric';published.mkdir(exist_ok=True)
    (published/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    kinds=['static_fusion','dynamic_fusion','legacy_attention_residual','dynamic_midlate4','dynamic_half','dynamic_late4','dynamic_midlate2','static_sparse','misaligned','reverse','sign_flip','zero_strength','numeric_drift']
    fitted={k:fit(rows,k) for k in kinds}
    primary=fitted['dynamic_midlate4'];matched={}
    for k in ['no_residual','zero_strength','misaligned','reverse','sign_flip','numeric_drift','static_sparse','dynamic_half','dynamic_late4','dynamic_midlate2']:
        preds=[predict(r,primary['folds'][r['tree']]['config'],k) for r in rows]
        matched[k]=dict(metrics=metrics(rows,preds),predictions=preds,paired_primary_vs_control=paired(rows,primary['predictions'],preds))
    raw_keys=['original','action_bias_1.0_all','mass_0.5_all']+list(configurations())
    raw={k:metrics(rows,[r['answers'][k] for r in rows]) for k in raw_keys}
    drift={k:dict(changed=sum(r['answers'][k]!=r['answers']['original'] for r in rows),max_abs=max(float(np.max(np.abs(np.array(r['scores'][k])-r['scores']['original']))) for r in rows),mean_abs=float(np.mean([np.mean(np.abs(np.array(r['scores'][k])-r['scores']['original'])) for r in rows]))) for k in configurations() if k.startswith('zero_')}
    timing={k:dict(total_seconds=sum(r['seconds'][k] for r in rows),mean_seconds=float(np.mean([r['seconds'][k] for r in rows])),p95_seconds=float(np.quantile([r['seconds'][k] for r in rows],.95))) for k in configurations()}
    for v in timing.values():v['relative_to_new_zero']=v['mean_seconds']/timing['zero_midlate4']['mean_seconds']
    result=dict(scope='Exploratory: all 800 records, 200 patients, 4 known development trees; no independent P3 confirmation',
        fusion_grid=GRID,residual=dict(gamma=.25,scale=2.,formula='gamma*tanh((intervened_logits-matched_zero_logits)/scale)'),
        primary='dynamic_midlate4',methods=fitted,matched_primary_controls=matched,raw=raw,numerical_drift=drift,
        performance=timing,resources=resources,raw_paired_vs_matched_zero={k:paired(rows,[r['answers'][k] for r in rows],[r['answers']['zero_'+v['geometry']] for r in rows]) for k,v in configurations().items()},paired_vs_static={k:paired(rows,v['predictions'],fitted['static_fusion']['predictions']) for k,v in fitted.items()},
        coverage=dict(rows=len(rows),patients=200,covered=sum(r['covered'] for r in rows),rule_action_many_to_one=sum(len(r['p'])>len(r['S']) for r in rows),option_action_many_to_one=sum(len([x for x in r['option_to_action'] if x>=0])>len(set(x for x in r['option_to_action'] if x>=0)) for r in rows),orphan_option_rows=sum(-1 in r['option_to_action'] for r in rows)))
    (destination/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    with (destination/'predictions.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator="\n");w.writerow(['case','tree','variant','gold','original']+kinds)
        for i,r in enumerate(rows):w.writerow([r['id'],r['tree'],r['variant'],r['gold_option'],r['answers']['original']]+[fitted[k]['predictions'][i] for k in kinds])
    print(json.dumps({k:v['metrics']['correct'] for k,v in fitted.items()}))
if __name__=='__main__':main()
