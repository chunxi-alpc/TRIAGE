"""E1 preparation/refitting entry points for audited representation adaptations.
Only --mode extract-truth loads Qwen. Historical reports are read-only.
No ASTER loss or ASTER fitting; LEACE-pair uses explicitly shared frozen E1 heads.
"""
import argparse,ast,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
HIST=Path('/home/gcx/aster_cr_repair_v2_run_20261108')
E1=Path('/home/gcx/aster-cr-retrain-paired-v2-20261111/reports/aster_cr_retrain_paired_v2_20261111')
OUT=ROOT/'reports/aster_e5_gaps_20261112/representation_e1_plan'
os.environ.setdefault('ASTER_CR_ROOT',str(HIST))
sys.path.insert(0,str(ROOT))
from experiments.aster_cr_retrain_paired_v2_20261111.common import Store,rows

def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
def prepared():
    pairs=rows(E1/'data/E1_pairs.jsonl')
    views=rows(E1/'data/E1_views.jsonl')
    unique={}
    for v in views:unique.setdefault(v['uid'],v)
    return pairs,unique
def plan():
    pairs,unique=prepared()
    m1=read(E1/'models/E1/full/m1.json')
    data=dict(dataset='E1 / Dataset A',pairs=len(pairs),unique_endpoints=len(unique),sources=len({p['source_id'] for p in pairs}),
        audit='docs/ASTER/baseline-audit-20261112.zh.md',no_aster_training=True,
        training_input=str(E1/'data/E1_pairs.jsonl'),feature_store=str(E1),raw_endpoint_features='Reuse exact existing E1 cache; do not extract again.',
        methods={
          'ITI-style task adaptation':dict(action='refit',objective='Correct/incorrect answer completion activation probes and mean direction; no ASTER pair loss',
             endpoints='All unique endpoints of the frozen 2540 pairs',native_head_selection='Logistic probes C=1,max_iter=100; fit-source held-out sources[::5]; top51',
             strength=15.,requires_gpu='Only missing correct/incorrect answer completion activations',randomness='Deterministic lbfgs probe; random_state=11 retained for reproducibility, no fake seed variance.'),
          'LEACE-pair adaptation':dict(action='refit',objective='Existing pair-side signed A/P kind+subtype concept covariance erasure',
             fit_function='experiments.aster_closed_form.operators.fit_operators variant=leace',
             shared_supervision='Frozen E1 M1 51-head positions; explicit ASTER A/P,C/F localization supervision. This is a task-adapted control, not standard LEACE.',
             heads=m1['heads'],eta=.95,kmax=4,theta=45,effect_filter='Existing |delta logit| >=0.1 for AP only, retained unchanged; 2540 input pairs passed to old entry point.',
             standard_LEACE='Not implemented here: requires stable instance concept labels; delegated to separate new-erasure task.'),
          'SEA task adaptation':dict(action='hold_until_fallback_diagnostic_resolved',historical_results='Reuse complete historical scores and paths; do not repeat them.',
             objective='Existing correct/incorrect completion spectral projection; no ASTER loss',ratio=.998,
             intervention='Official vendor AdapterLayer, final residual, l2_norm; no algorithm rewrite',
             caveat='Singleton-token per-coordinate normalization can produce 0/0. Existing fallback must be described; GPU numeric parity unresolved.')},
        hyperparameters='Single historical configuration fixed before E1 test; no historical test tuning.',
        evaluation='Development first, freeze settings; historical MEGA40/806 evaluation only if authorized. Do not repeat NLI500 or MedPIC188.')
    write(OUT/'plan.json',data);print(json.dumps({k:data[k] for k in ['pairs','unique_endpoints','sources']}))
def fit_leace():
    if (OUT/'leace_pair/fit.json').exists():return
    from experiments.aster_closed_form.operators import fit_operators
    pairs,unique=prepared();store=Store(E1);heads=read(E1/'models/E1/full/m1.json')['heads']
    features={}
    for uid,v in unique.items():
        features[uid]=dict(heads=store.heads(uid,heads).reshape(len(heads),128),ell=store.ell(uid),split='fit',original=v.get('original',False))
    local=list(range(len(heads)))
    started=time.time()
    ops,meta=fit_operators(features,pairs,local,{h:['A','P'] for h in local},eta=.95,kmax=4,theta=45,variant='leace')
    ops={int(heads[h]):op for h,op in ops.items()}
    dest=OUT/'leace_pair';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'operator.pt').exists():raise FileExistsError(dest/'operator.pt')
    torch.save(ops,dest/'operator.pt')
    meta.update(method='LEACE-pair adaptation',shared_supervision='Frozen E1 M1 head positions; explicit task adaptation',head_mapping=heads,seconds=time.time()-started,input_pair_count=len(pairs))
    write(dest/'fit.json',meta)
    print(json.dumps(dict(method='LEACE-pair adaptation',heads=len(heads),output=str(dest))))
def extract_truth(device):
    if (OUT/'truth_extraction.json').exists():return
    from experiments.aster_cr.engine import Engine
    started=time.time()
    pairs,unique=prepared();store=Store(E1)
    dest=OUT/'truth_features';dest.mkdir(parents=True,exist_ok=True)
    oldbase=[v for v in rows(HIST/'data/views.jsonl') if v['split']=='fit' and v['variant']=='full']
    legacy_path=HIST/'qwen/comparators/truth_features.pt'
    legacy=torch.load(legacy_path,map_location='cpu',weights_only=False) if legacy_path.exists() else None
    oldindex={v['prompt']:i for i,v in enumerate(oldbase)} if legacy is not None else {}
    eng=None;reused=fresh=raw_residual_forwards=0
    for idx,(uid,v) in enumerate(unique.items(),1):
        path=dest/(uid+'.npz')
        if path.exists():continue
        prompt=store.inputs[uid]['prompt'];label=int(v['label'])
        if prompt in oldindex and len(oldbase)==len(legacy['pos']):
            i=oldindex[prompt]
            if int(oldbase[i]['label'])!=label:raise ValueError('Cached completion label mismatch')
            values=dict(pos=legacy['pos'][i],neg=legacy['neg'][i],xp=legacy['xp'][i],xn=legacy['xn'][i],xb=legacy['xb'][i])
            reused+=1
        else:
            if eng is None:eng=Engine('qwen',device=device)
            original_render=eng.render;rr={}
            try:
                for answer in [0,1]:
                    token=chr(26159) if answer else chr(21542)
                    eng.render=lambda text,task='binary',t=token:original_render(text,task)+t
                    rr[answer]=eng.score(prompt,capture=True)
            finally:eng.render=original_render
            p,n=rr[label],rr[1-label]
            values=dict(pos=p['heads'].flatten(0,1).numpy(),neg=n['heads'].flatten(0,1).numpy(),xp=p['residual'].numpy(),xn=n['residual'].numpy())
            with np.load(store.record(uid)['feature']) as f:
                values['xb']=f['residual'].astype(np.float32) if 'residual' in f.files else None
            if values['xb'] is None:
                captured={}
                def residual_hook(module,args,out):
                    val=out[0] if isinstance(out,tuple) else out
                    captured['xb']=val[0,-1].detach().float().cpu().numpy()
                context=eng.prepare(prompt)
                handle=eng.layers[-1].register_forward_hook(residual_hook)
                try:
                    with torch.no_grad():eng.run(context,capture=False)
                finally:handle.remove()
                values['xb']=captured['xb'];raw_residual_forwards+=1
            fresh+=1
        np.savez(path,**values)
        if idx%100==0:update_status(dict(state='extracting_completion_activations',done=idx,total=len(unique),new_completion_endpoints=fresh,new_raw_residual_forwards=raw_residual_forwards,seconds=time.time()-started))
    write(OUT/'truth_extraction.json',dict(endpoints=len(unique),new_completed=fresh,reused_this_run=reused,new_raw_residual_forwards=raw_residual_forwards,
        new_forward_calls=2*fresh+raw_residual_forwards,seconds=time.time()-started,peak_gib=torch.cuda.max_memory_allocated(eng.device)/2**30 if eng else 0,
        note='Existing base features reused; only missing baseline residuals extracted along with missing correct/incorrect completion activations.'))

def truth_arrays():
    _,unique=prepared();uids=list(unique);dest=OUT/'truth_features'
    missing=[u for u in uids if not (dest/(u+'.npz')).exists()]
    if missing:raise RuntimeError(f'{len(missing)} missing completion activations; run extract-truth first')
    return unique,uids,dest
def fit_iti():
    if (OUT/'iti_style/fit.json').exists():return
    started=time.time()
    from sklearn.linear_model import LogisticRegression
    unique,uids,dest=truth_arrays()
    sources=sorted({v['source_id'] for v in unique.values()});held=set(sources[::5])
    mask=np.array([unique[u]['source_id'] not in held for u in uids])
    # Disk-backed arrays keep all 1152 heads without repeated NPZ loads.
    cache=OUT/'iti_cache';cache.mkdir(parents=True,exist_ok=True)
    shape=(len(uids),1152,128)
    pos=np.lib.format.open_memmap(cache/'positive.npy',mode='w+',dtype='float32',shape=shape)
    neg=np.lib.format.open_memmap(cache/'negative.npy',mode='w+',dtype='float32',shape=shape)
    for i,u in enumerate(uids):
        with np.load(dest/(u+'.npz')) as f:pos[i]=f['pos'];neg[i]=f['neg']
    labels=np.r_[np.ones(mask.sum()),np.zeros(mask.sum())];eval_labels=np.r_[np.ones((~mask).sum()),np.zeros((~mask).sum())]
    acc=[]
    for h in range(1152):
        probe=LogisticRegression(C=1.,max_iter=100,random_state=11).fit(np.r_[pos[mask,h],neg[mask,h]],labels)
        acc.append(float(probe.score(np.r_[pos[~mask,h],neg[~mask,h]],eval_labels)))
    chosen=np.argsort(-np.array(acc))[:51];directions={}
    for h in chosen:
        d=pos[:,h].mean(0)-neg[:,h].mean(0);d/=max(np.linalg.norm(d),1e-8)
        sigma=np.std(np.r_[pos[:,h],neg[:,h]]@d)
        directions[int(h)]=torch.tensor(15*sigma*d)
    dest=OUT/'iti_style';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'operator.pt').exists():raise FileExistsError(dest/'operator.pt')
    torch.save(directions,dest/'operator.pt')
    write(dest/'fit.json',dict(method='ITI-style task adaptation',alpha=15.,heads=chosen.tolist(),probe_scores=acc,unique_endpoints=len(uids),held_fit_sources=sorted(held),objective='Original local correct/incorrect activation probe and mean direction; no ASTER loss',seconds=time.time()-started))
def fit_sea():
    if (OUT/'sea_adaptation/fit.json').exists():return
    started=time.time()
    from experiments.aster_cr.comparators import sea_basis
    unique,uids,dest=truth_arrays();store=Store(E1);xp=[];xn=[];xb=[]
    for u in uids:
        with np.load(dest/(u+'.npz')) as f:xp.append(f['xp']);xn.append(f['xn']);xb.append(f['xb'].astype(np.float32))
    sp=sea_basis(np.stack(xb),np.stack(xp));sn=sea_basis(np.stack(xb),np.stack(xn));d=len(xb[0])
    dest=OUT/'sea_adaptation';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'operator.pt').exists():raise FileExistsError(dest/'operator.pt')
    torch.save(dict(positive=torch.tensor(sp@sp.T),negative=torch.tensor(np.eye(d,dtype=np.float32)-sn@sn.T)),dest/'operator.pt')
    write(dest/'fit.json',dict(method='SEA task adaptation',endpoints=len(uids),positive_rank=sp.shape[1],negative_rank=sn.shape[1],ratio=.998,
        objective='Existing sea_basis, unchanged; original nonfinite coordinate fallback retained.',seconds=time.time()-started))
def diagnose_sea(limit):
    from experiments.aster_cr.comparators import sea_class
    state=torch.load(HIST/'qwen/comparators/sea.pt',map_location='cpu',weights_only=False)
    archive={p.name:read(p).get('SEA_zero_norm_fallback_positions') for p in sorted((HIST/'qwen/comparators').glob('*_metrics.json'))}
    records=[r for p in sorted((HIST/'qwen/scores').glob('dev_*.jsonl')) for r in rows(p) if r.get('status')=='ok'][:limit]
    module=sea_class()(state['positive'].clone(),state['negative'].clone(),'l2_norm',None)
    results=[]
    for r in records:
        with np.load(r['feature']) as f:h=torch.from_numpy(f['residual'].astype(np.float32)).reshape(1,1,-1).to(torch.bfloat16)
        with torch.no_grad():
            y=module(h);bad=~torch.isfinite(y)
            x=h.reshape(-1,1)
            combined=state['positive'].to(torch.bfloat16)@x+state['negative'].to(torch.bfloat16)@x
            zero=combined.float().norm(dim=-1)==0
            restored=torch.where(bad,h,y)
        results.append(dict(view_id=r['view_id'],bad_elements=int(bad.sum()),zero_combined_coordinates=int(zero.sum()),
            total_coordinates=h.numel(),changed_coordinates=int((restored!=h).sum()),delta_norm=float((restored.float()-h.float()).norm())))
    write(OUT/'sea_fallback_diagnostic.json',dict(scope='CPU replay of official vendor AdapterLayer using existing development residuals; no model inference.',
        sample_size=len(records),samples=results,total_bad_elements=sum(x['bad_elements'] for x in results),archive_fallback_counts=archive,
        meaning='Counter sums nonfinite scalar coordinates, not examples or token positions.',
        mechanism='Official class transposes singleton-token input to [4096,1] then normalizes each coordinate across the length/batch dimension. BF16 cancellation to zero gives 0/0. The existing wrapper restores only those original coordinates.',
        remaining='CPU and GPU BF16 matmul can differ; archived exact fallback locations were not saved, so this diagnostic cannot attribute every historical count. No algorithm or existing result modified.',
        evaluation_status='Historical SEA score/path outputs exist; baseline must be named SEA task adaptation with official singleton-token coordinate normalization and per-coordinate nonfinite fallback.'))
    print(json.dumps(dict(samples=len(results),bad_elements=sum(x['bad_elements'] for x in results),changed_samples=sum(x['changed_coordinates']>0 for x in results))))

METHODS={'leace_pair':('LEACE-pair adaptation','leace'),'iti_style':('ITI-style task adaptation','iti'),'sea_adaptation':('SEA task adaptation','sea')}
def update_status(value):
    p=OUT/'pipeline_status.json';p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));temp.replace(p)
def append(p,value):
    with Path(p).open('a',encoding='utf-8') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n')
def inspect_truth():
    _,unique=prepared();store=Store(E1)
    oldbase=[v for v in rows(HIST/'data/views.jsonl') if v['split']=='fit' and v['variant']=='full']
    prompts={v['prompt'] for v in oldbase}
    existing=[u for u in unique if (OUT/'truth_features'/(u+'.npz')).exists()]
    reusable=[u for u in unique if u not in existing and store.inputs[u]['prompt'] in prompts]
    missing=[u for u in unique if u not in existing and u not in reusable]
    info=dict(endpoints=len(unique),existing_completion_files=len(existing),legacy_reusable_endpoints=len(reusable),
        missing_completion_endpoints=len(missing),additional_forward_calls=2*len(missing),
        estimate='GPU extraction time: missing forwards * 0.3-1.0 seconds, plus model load/I/O; actual time saved after execution.',
        extraction_hours=[2*len(missing)*.3/3600,2*len(missing)/3600])
    path=OUT/'truth_inventory.json'
    if not path.exists():write(path,info)
    print(json.dumps(info))
    return info
def evaluate(method,split,device):
    from experiments.aster_cr.comparators import Comparators,sea_class
    from experiments.aster_cr.engine import Engine
    from experiments.aster_cr.common import TASK_KINDS
    from experiments.aster_cr.paths import execute
    from experiments.aster_closed_form.data import canonical_guide,prompt
    from experiments.aster_e5_gaps_20261112.loreft_native import bootstrap_summary
    dest=OUT/method
    if (dest/(split+'_metrics.json')).exists():return
    training=read(dest/'fit.json')
    if split=='test' and not read(dest/'selection.json')['frozen']:raise RuntimeError('Test needs frozen development record')
    started=time.time();eng=Engine('qwen',device=device);eng.model.eval()
    comp=Comparators.__new__(Comparators);comp.eng=eng;comp.ops={};comp.sea_zero_norm_fallbacks=0
    if method=='sea_adaptation':
        state=torch.load(dest/'operator.pt',map_location=eng.device,weights_only=False)
        comp.sea=sea_class()(state['positive'],state['negative'],'l2_norm',None)
    else:comp.ops[METHODS[method][1]]=torch.load(dest/'operator.pt',map_location=eng.device,weights_only=False)
    torch.cuda.reset_peak_memory_stats(eng.device)
    views=[v for v in rows(HIST/'data/views.jsonl') if v['split']==split and v['kind'] in TASK_KINDS]
    pairs=[p for p in rows(HIST/'data/pairs.jsonl') if p['split']==split]
    cases=[c for c in rows(HIST/'data/path_cases.jsonl') if c['split']==split]
    raw={r['view_id']:r['ell'] for p in (HIST/'qwen/scores').glob(split+'_*.jsonl') for r in rows(p)}
    vf=dest/(split+'_input_predictions.jsonl');pf=dest/(split+'_path_predictions.jsonl');nf=dest/(split+'_path_node_scores.jsonl')
    saved={r['view_id']:r['score'] for r in rows(vf)}
    cache={v['prompt']:saved[v['view_id']] for v in views if v['view_id'] in saved}
    cache.update({r['prompt']:r['score'] for r in rows(nf)})
    def score(text):
        if text not in cache:
            context=eng.prepare(text)
            with torch.no_grad():cache[text]=float(Comparators.edited(comp,context,METHODS[method][1]))
        return cache[text]
    for i,v in enumerate(views,1):
        if v['view_id'] not in saved:
            s=score(v['prompt']);saved[v['view_id']]=s
            append(vf,dict(view_id=v['view_id'],source_id=v['source_id'],label=v['label'],score=s,prediction=int(s>0),raw_score=raw[v['view_id']]))
        if i%100==0:update_status(dict(state='evaluating_inputs',method=method,split=split,done=i,total=len(views)))
    done={r['case_id'] for r in rows(pf)}
    for i,c in enumerate(cases,1):
        if c['case_id'] in done:continue
        tree=canonical_guide(c['guideline']);nodes={n['id']:n for n in tree['nodes']};gid=tree['nodes'][0]['id'];scores={}
        while gid in nodes and gid not in scores:
            text=prompt(c['guideline'],c['definitions'],c['patient'],gid);s=score(text);scores[gid]=s
            append(nf,dict(case_id=c['case_id'],source_id=c['source_id'],gid=gid,prompt=text,score=s))
            gid=nodes[gid]['yes' if s>0 else 'no']
        append(pf,dict(case_id=c['case_id'],source_id=c['source_id'],**execute(c,scores)))
        if i%50==0:update_status(dict(state='evaluating_paths',method=method,split=split,done=i,total=len(cases)))
    metrics,per_pair=bootstrap_summary(views,pairs,saved,raw,rows(pf),rows(HIST/'qwen/path_results'/('full_'+split+'.jsonl')))
    with (dest/(split+'_pair_predictions.jsonl')).open('x') as f:
        for row in per_pair:f.write(json.dumps(row)+'\n')
    metrics.update(method=METHODS[method][0],split=split,training=training,seconds=time.time()-started,
        peak_gib=torch.cuda.max_memory_allocated(eng.device)/2**30,engine_counts=eng.counts,
        SEA_nonfinite_coordinate_fallbacks=comp.sea_zero_norm_fallbacks,historical_evaluation=True,new_independent_test=False,
        extra_ASTER_loss=False,config_candidates=1)
    write(dest/(split+'_metrics.json'),metrics)
    print(json.dumps(dict(method=method,split=split,path=metrics['path'])),flush=True)
def freeze(method):
    dest=OUT/method
    if (dest/'selection.json').exists():return
    metrics=read(dest/'dev_metrics.json')
    write(dest/'selection.json',dict(frozen=True,method=METHODS[method][0],candidates=1,
        selection_rule='The sole historical hyperparameter configuration was fixed before E1 evaluation. No test-driven tuning.',
        dev_path=metrics['path'],dev_natural=metrics['groups']['natural'],test_used_for_selection=False,
        frozen_at=time.strftime('%Y-%m-%d %H:%M:%S')))
def pipeline(device):
    import subprocess,csv
    started=time.time()
    steps=[('fit-leace',None,None),('extract-truth',None,None),('fit-iti',None,None),('fit-sea',None,None)]
    steps += [('evaluate',m,'dev') for m in METHODS]
    steps += [('freeze',m,None) for m in METHODS]
    steps += [('evaluate',m,'test') for m in METHODS]
    logs=OUT/'logs';logs.mkdir(parents=True,exist_ok=True)
    for mode,method,split in steps:
        update_status(dict(state='running',mode=mode,method=method,split=split))
        cmd=[sys.executable,str(Path(__file__).resolve()),'--mode',mode,'--device',device]
        if method:cmd+=['--method',method]
        if split:cmd+=['--split',split]
        logfile=logs/('_'.join(x for x in [mode,method,split] if x)+'.log')
        with logfile.open('a') as f:p=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        if p.returncode:
            update_status(dict(state='failed',mode=mode,method=method,split=split,log=str(logfile)))
            raise RuntimeError('Representation pipeline step failed; see '+str(logfile))
    table=[]
    for method in METHODS:
        m=read(OUT/method/'test_metrics.json');ap=m['groups']['AP'];cf=m['groups']['CF'];p=m['path']
        table.append(dict(method=METHODS[method][0],AP_n=ap['n'],AP_RMS=ap['difference_rms'],AP_RMS_ci95=json.dumps(ap['difference_rms_ci']),
            AP_VR=ap['variance_reduction'],AP_VR_ci95=json.dumps(ap['variance_reduction_ci']),
            CF_n=cf['n'],CF_correction_RMS=cf['correction_rms'],CF_correction_RMS_ci95=json.dumps(cf['correction_rms_ci']),
            CF_correct=cf['paired_correct'],CF_accuracy_ci95=json.dumps(cf['paired_accuracy_ci']),
            CF_ER=cf['mean_evidence_retention'],CF_ER_ci95=json.dumps(cf['mean_evidence_retention_ci']),
            path_correct=p['correct'],path_n=p['n'],path_accuracy=p['accuracy'],path_accuracy_ci95=json.dumps(p['accuracy_ci']),
            evaluation_seconds=m['seconds'],evaluation_peak_gib=m['peak_gib'],SEA_fallbacks=m['SEA_nonfinite_coordinate_fallbacks']))
    with (OUT/'section_4_5_representation_e1.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    completion=dict(state='complete',methods=list(METHODS.values()),seconds=time.time()-started,
                    dataset_pairs=2540,training_endpoints=4637,ASTER_training_runs=0,NLI4CT_runs=0,MedPIC_runs=0)
    write(OUT/'completion.json',completion);update_status(completion)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--mode',required=True,choices=['prepare','fit-leace','extract-truth','fit-iti','fit-sea','diagnose-sea','inspect-truth','evaluate','freeze','pipeline','controller'])
    p.add_argument('--device',default='cuda:0');p.add_argument('--limit',type=int,default=32)
    p.add_argument('--method',choices=list(METHODS),default='leace_pair');p.add_argument('--split',choices=['dev','test'],default='dev')
    a=p.parse_args();torch.set_num_threads(4)
    {'prepare':plan,'fit-leace':fit_leace,'extract-truth':lambda:extract_truth(a.device),'fit-iti':fit_iti,'fit-sea':fit_sea,'diagnose-sea':lambda:diagnose_sea(a.limit),'inspect-truth':inspect_truth,'evaluate':lambda:evaluate(a.method,a.split,a.device),'freeze':lambda:freeze(a.method),'pipeline':lambda:pipeline(a.device),'controller':lambda:pipeline(a.device)}[a.mode]()
