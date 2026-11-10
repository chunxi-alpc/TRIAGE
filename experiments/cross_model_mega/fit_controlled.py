"""Experiment B: reuse each model's independent MEGA features; fit one random control."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse,csv,gzip,json,shutil,subprocess,time
from pathlib import Path
import numpy as np
from . import cached_fit_helpers as H

REPO=Path(__file__).resolve().parents[2]
OLD=Path('/home/gcx/aster_five_models_20261109')
DATA=OLD/'data/mega'
OUT=REPO/'reports/aster_cancel_story_20261109/meeting/cross_model_mega'
NAMES=['raw','bias_only','aster_cr','random_trace']
BOOT=2000

def dump(p,x):H.dump(p,x)
def csvwrite(p,rr):
    if not rr:return
    with p.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rr for k in r)));w.writeheader();w.writerows(rr)
def finite(x):return None if not np.isfinite(x) else float(x)
def bootstrap_pairs(raw,corrected,sign,sources,seed=20261111):
    ids,inv=np.unique(np.asarray(sources),return_inverse=True);n=len(ids)
    a=np.column_stack([np.bincount(inv),np.bincount(inv,weights=raw),np.bincount(inv,weights=raw**2),
       np.bincount(inv,weights=corrected),np.bincount(inv,weights=corrected**2),
       np.bincount(inv,weights=sign*raw),np.bincount(inv,weights=sign*corrected),
       np.bincount(inv,weights=(corrected-raw)**2)])
    rng=np.random.default_rng(seed);sample=rng.integers(n,size=(BOOT,n));v=a[sample].sum(1)
    varraw=v[:,2]/v[:,0]-(v[:,1]/v[:,0])**2
    varnew=v[:,4]/v[:,0]-(v[:,3]/v[:,0])**2
    metrics=dict(variance_reduction=1-varnew/varraw,
        prediction_r2=1-v[:,4]/(v[:,2]-v[:,1]**2/v[:,0]),
        difference_rms=np.sqrt(v[:,4]/v[:,0]),correction_rms=np.sqrt(v[:,7]/v[:,0]),
        evidence_response_retention=v[:,6]/v[:,5])
    return {k:[finite(y) for y in np.nanpercentile(x,[2.5,97.5])] for k,x in metrics.items()},ids,a

def fit_one(model):
    started=time.time();old=OLD/'results'/model/'mega';out=OUT/model;out.mkdir(parents=True,exist_ok=True)
    selection=json.loads((old/'selection.json').read_text());heads=selection.get('heads',selection.get('H'))
    data=H.load_available(old,DATA,heads)
    with np.load(old/'fitted.npz',allow_pickle=False) as f:params={k:f[k] for k in f.files}
    frozen=json.loads((old/'selection-freeze.json').read_text())
    # All values below use this model's fit records and this model's existing normalization.
    raw,ell=data['raw'],data['ell'];z=(raw-params['mu'])/params['scale']
    ap,al,ar=H.pair_indices(data,'fit',('A','P'));cf,cl,cr=H.pair_indices(data,'fit',('C','F'))
    rng=np.random.default_rng(20261111);rand=np.linalg.qr(rng.normal(size=(raw.shape[1],16)),mode='reduced')[0]
    dx=z[ar]-z[al];ex=z[cr]-z[cl];y=ell[ar]-ell[al]
    w,solver=H.solve(dx@rand,y,ex@rand,float(params['delta']),True)
    coef=rand@w;fi=H.natural_indices(data,'fit')
    bias=H.calibrate(ell[fi]+z[fi]@coef,data['labels'][fi])
    params.update(random_trace_basis=rand,random_trace_w=w,random_trace_coef=coef,random_trace_b=bias)
    keep=['heads','mu','scale','S','U','N','alpha','tau','delta','random_trace_basis','random_trace_w']
    keep += [m+s for m in NAMES for s in ('_coef','_b')]
    np.savez_compressed(out/'fitted.npz',**{k:params[k] for k in keep})
    for name in ('selection.json','selection-freeze.json','model_metadata.json'):
        if (old/name).exists():shutil.copy2(old/name,out/('reused_'+name))
    spec=json.loads((Path(__file__).parent/'models.json').read_text())[model]
    config=dict(model=model,model_name=spec['name'],model_weight_path=spec['path'],weight_revision='unrecorded',
      weight_revision_note='Local checkpoint metadata does not record an immutable upstream revision; no revision invented.',
      code_base_git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
      new_code_directory='experiments/cross_model_mega',old_cache_root=str(old),input_manifest=str(DATA),
      data_protocol='Same pre-existing fit80/dev20/test40 MEGA A/P and C/F source split; official 7458 test paths.',
      methods=NAMES,random_seed=20261111,random_rank=16,alpha=float(params['alpha']),tau=float(params['tau']),delta=float(params['delta']),
      head_source='This model independently selected heads on fit only; reused completed selection and S/N/normalization/linear calibration.',
      selected_heads=heads,random_solver=solver,random_b=float(bias),random_fit_ap=len(ap),random_fit_cf=len(cf),
      calibration='Natural fit labels for intercept only, same as reused ASTER-CR.',
      selection_uses_test=False,official_test_used_for_fit=False,created_utc=time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime()))
    dump(out/'config.json',config)
    ss={m:ell+z@params[m+'_coef']+float(params[m+'_b']) for m in NAMES}
    summary=[];pair_rows=[];source_rows=[]
    labels=data['labels'];views=data['views'];natural_sources={}
    for split in ('fit','dev','test'):
        for m in NAMES:
            for family,kinds in [('AP',('A','P')),('CF',('C','F'))]:
                pp,li,ri=H.pair_indices(data,split,kinds)
                gm=H.group_metrics(data,ss[m],split,kinds,float(params['delta']))
                rawdy=ell[ri]-ell[li];dy=ss[m][ri]-ss[m][li]
                sign=np.array([p['right_label']-p['left_label'] for p in pp])
                ci,ids,a=bootstrap_pairs(rawdy,dy,sign,[p['source_id'] for p in pp])
                gm.update(model=model,method=m,split=split,group=family,prediction_rmse=H.rms(dy) if family=='AP' else None,
                   prediction_r2=1-float(dy@dy)/float(np.sum((rawdy-rawdy.mean())**2)) if family=='AP' else None)
                for metric,interval in ci.items():
                    if family=='AP' and metric=='evidence_response_retention':continue
                    if family=='CF' and metric in ('variance_reduction','prediction_r2'):continue
                    gm[metric+'_ci_low'],gm[metric+'_ci_high']=interval
                gm.pop('by_kind',None);summary.append(gm)
                for sid,vals in zip(ids,a):
                    source_rows.append(dict(model=model,method=m,split=split,group=family,source_id=sid,
                      n=int(vals[0]),raw_sum=vals[1],raw_squared_sum=vals[2],sum=vals[3],squared_sum=vals[4],
                      raw_signed_sum=vals[5],signed_sum=vals[6],correction_squared_sum=vals[7]))
                for j,p in enumerate(pp):
                    pair_rows.append(dict(model=model,method=m,split=split,pair_id=p['pair_id'],source_id=p['source_id'],
                      kind=p['kind'],left=p['left'],right=p['right'],left_label=p['left_label'],right_label=p['right_label'],
                      raw_left=ell[li[j]],raw_right=ell[ri[j]],corrected_left=ss[m][li[j]],corrected_right=ss[m][ri[j]],
                      raw_difference=rawdy[j],corrected_difference=dy[j],correction=dy[j]-rawdy[j],
                      exceeds_fit_budget=bool(abs(dy[j]-rawdy[j])>float(params['delta'])+1e-8)))
            ni=H.natural_indices(data,split);cor=(ss[m][ni]>0)==labels[ni];base=(ell[ni]>0)==labels[ni]
            sid=np.array([views[i]['source_id'] for i in ni]);ids=np.unique(sid)
            totals=np.array([[(sid==s).sum(),cor[sid==s].sum(),base[sid==s].sum()] for s in ids])
            rng=np.random.default_rng(20261111);t=totals[rng.integers(len(ids),size=(BOOT,len(ids)))].sum(1)
            interval=np.percentile((t[:,1]-t[:,2])/t[:,0],[2.5,97.5])
            summary.append(dict(model=model,method=m,split=split,group='natural',n=len(ni),correct=int(cor.sum()),
                accuracy=float(cor.mean()),delta_correct=int(cor.sum()-base.sum()),accuracy_delta_ci_low=interval[0],accuracy_delta_ci_high=interval[1]))
            for source,vals in zip(ids,totals):
                source_rows.append(dict(model=model,method=m,split=split,group='natural',source_id=source,n=vals[0],correct=vals[1],raw_correct=vals[2]))
    csvwrite(out/'controlled_summary.csv',summary);csvwrite(out/'controlled_source_metrics.csv',source_rows)
    with gzip.open(out/'controlled_pairs.jsonl.gz','wt',encoding='utf8') as f:
        for r in pair_rows:f.write(json.dumps(r,ensure_ascii=False,default=lambda x:x.item() if hasattr(x,'item') else x)+'\n')
    with gzip.open(out/'controlled_predictions.jsonl.gz','wt',encoding='utf8') as f:
        for i,v in enumerate(views):
            r={k:v.get(k) for k in ('view_id','source_id','group_id','split','kind','variant','original','label')}
            r['scores']={m:float(ss[m][i]) for m in NAMES};f.write(json.dumps(r,ensure_ascii=False)+'\n')
    dump(out/'controlled_complete.json',dict(state='complete',seconds=time.time()-started,views=len(views),pairs=len(data['pairs']),random_solver=solver))
    print(json.dumps(dict(model=model,state='controlled_complete',seconds=time.time()-started)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);a=p.parse_args();fit_one(a.model)
