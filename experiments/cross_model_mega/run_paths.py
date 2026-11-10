"""Frozen four-method MEGA traversal. Each model supplies its own coefficients and cache."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse,gzip,json,time,traceback
from pathlib import Path
import numpy as np
from .fit_controlled import REPO,OLD,DATA,OUT,NAMES,dump
from .cached_fit_helpers import rows,selected_feature
from experiments.aster_closed_form.data import canonical_guide,prompt

def append(p,r):
    with p.open('a',encoding='utf8') as f:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n');f.flush()

def score_selected(engine,text,selected):
    """Canonical final-token native-cache replay, copying only selected heads."""
    import torch
    context=engine.prepare(text);bylayer={};captured={};handles=[]
    for h in selected:bylayer.setdefault(int(h)//engine.n_heads,[]).append(int(h)%engine.n_heads)
    def hook_for(layer,indices):
        def hook(module,args):
            value=args[0]
            captured[layer]=value[0,-1].reshape(engine.n_heads,engine.head_dim)[indices].detach()
        return hook
    before=time.perf_counter();engine._active=True
    try:
        for layer,indices in bylayer.items():handles.append(engine.layers[layer].self_attn.o_proj.register_forward_pre_hook(hook_for(layer,indices)))
        with torch.no_grad():
            embeds=engine.model.get_input_embeddings()(context.input_ids[:,-1:]).detach()
            output=engine.model.model(inputs_embeds=embeds,attention_mask=torch.ones_like(context.input_ids),
              past_key_values=engine._cache(context),use_cache=True,
              cache_position=torch.tensor([context.length-1],device=engine.device),
              position_ids=torch.tensor([[context.length-1]],device=engine.device))
            ell=float(engine._score_hidden(output.last_hidden_state[0,-1]))
            positions={(layer,head):j for layer,indices in bylayer.items() for j,head in enumerate(indices)}
            hh=torch.stack([captured[int(h)//engine.n_heads][positions[int(h)//engine.n_heads,int(h)%engine.n_heads]] for h in selected])
            feat=hh.float().cpu().numpy().reshape(-1).astype('float16')
        engine.counts['final_forwards']+=1;engine.counts['final_seconds']+=time.perf_counter()-before
        return dict(ell=ell,prediction=int(ell>0),tokens=context.length,selected_heads=feat)
    finally:
        for handle in handles:handle.remove()
        engine._active=False

class Frozen:
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as f:self.p={k:f[k] for k in f.files}
        self.heads=self.p['heads'].astype(int)
    def scores(self,ell,feat):
        z=(np.asarray(feat,np.float32).reshape(-1)-self.p['mu'])/self.p['scale']
        return {m:float(ell+z@self.p[m+'_coef']+self.p[m+'_b']) for m in NAMES}

def run(model,gpu,shard=0,shards=1,limit=None):
    import torch
    from .engine import Engine
    from .prefix_cache import attach_case_prefix,validate_case_prefix
    torch.set_num_threads(2);torch.cuda.set_device(gpu)
    out=OUT/model;work=Path('/home/gcx/aster_cross_model_mega_20261111')/model/('shard'+str(shard))
    work.mkdir(parents=True,exist_ok=True);(work/'features').mkdir(exist_ok=True)
    old=OLD/'results'/model/'mega';frozen=Frozen(out/'fitted.npz')
    controlled=[dict(c,evaluation_group='controlled_'+c['split']) for c in rows(DATA/'path_cases.jsonl') if c['split'] in ('dev','test')]
    official=[dict(c,evaluation_group='official_test') for c in rows(Path('/home/gcx/aster_next_20261109/inventory/official_all_path_cases.jsonl'))]
    cases=(controlled+official)[shard::shards]
    if limit is not None:cases=cases[:limit]
    results=rows(work/'path_results.jsonl');written={(r['case_id'],r['method']) for r in results}
    own={r['id']:r for r in rows(work/'nodes.jsonl')}
    prior={r['id']:r for r in rows(old/'paths/nodes.jsonl') if r.get('status')=='ok'}
    if model=='medgemma':
        for file in sorted((Path('/home/gcx/aster_cr_repair_v2_run_20261108')/model/'paths').glob('features_*.jsonl')):
            for r in rows(file):
                if r.get('status')=='ok':prior.setdefault(r['id'],r)
    oldresults={(r['case_id'],r['method']):r for s in ('dev','test') for r in rows(old/'paths'/f'{s}.jsonl') if r['method'] in NAMES}
    pending=[c for c in cases if any((c['case_id'],m) not in written for m in NAMES)]
    if not pending:
        
        if not (work/'done.json').exists():dump(work/'done.json',dict(state='complete',model=model,cases=len(cases),new_forwards=0,already_complete=True))
        return
    engine=None;new_forwards=0;reused=0;start=time.time();natural_written={r['id'] for r in rows(work/'natural_nodes.jsonl')}
    completed=0;validated=False
    for case in pending:
        tree=canonical_guide(case['guideline']);nodes={n['id']:n for n in tree['nodes']};root=tree['nodes'][0]['id']
        # Reuse previously completed paths for the three unchanged methods.
        for m in NAMES:
            key=case['case_id'],m
            if key in written:continue
            if case['evaluation_group'].startswith('controlled_') and key in oldresults:
                rr=dict(oldresults[key],evaluation_group=case['evaluation_group'],reused_result=True,model=model)
                append(work/'path_results.jsonl',rr);written.add(key)
        names=[m for m in NAMES if (case['case_id'],m) not in written]
        texts=None;cache={}
        def node(gid):
            nonlocal engine,texts,new_forwards,reused,validated
            key=case['case_id']+':'+gid
            if gid in cache:return cache[gid]
            rec=own.get(key) or prior.get(key)
            if rec is not None and rec.get('status')=='ok' and Path(rec['feature']).exists():
                with np.load(rec['feature'],allow_pickle=False) as a:feat=selected_feature(a,frozen.heads)
                scores=frozen.scores(rec['ell'],feat);reused+=1
            else:
                if engine is None:engine=Engine(model,device='cuda:'+str(gpu),metadata_path=out/'runtime_model_metadata.json')
                if texts is None:
                    texts={n['id']:prompt(case['guideline'],case.get('definitions',[]),case['patient'],n['id']) for n in tree['nodes']}
                    try:
                        if not validated:
                            validate_case_prefix(engine,texts,out/'prefix_validation.json');validated=True
                        else:attach_case_prefix(engine,texts)
                    except ValueError as exc:
                        attach_case_prefix(engine,{})
                        if not validated:
                            dump(out/'prefix_validation.json',dict(state='unavailable',error=str(exc)));validated=True
                rec=dict(id=key,case_id=case['case_id'],source_id=case['source_id'],gid=gid)
                try:
                    result=score_selected(engine,texts[gid],frozen.heads)
                    feature=work/'features'/(str(len(own))+'.npz')
                    np.savez_compressed(feature,selected_heads=result.pop('selected_heads'))
                    rec.update(result,feature=str(feature),status='ok')
                    with np.load(feature) as a:scores=frozen.scores(rec['ell'],a['selected_heads'])
                except ValueError as exc:
                    rec.update(status='unavailable',error=str(exc));scores=None
                append(work/'nodes.jsonl',rec);own[key]=rec;new_forwards+=1
            cache[gid]=scores
            return scores
        # Official natural node evaluation covers every gold-path node even if a method diverges early.
        if case['evaluation_group']=='official_test':
            for gold in case['gold_path']:
                key=case['case_id']+':'+gold['gid']
                if key in natural_written:continue
                score=node(gold['gid'])
                append(work/'natural_nodes.jsonl',dict(id=key,case_id=case['case_id'],source_id=case['source_id'],gid=gold['gid'],
                  gold_label=gold['label'],evaluation_group=case['evaluation_group'],scores=score,status='ok' if score else 'unavailable'))
                natural_written.add(key)
        states={m:dict(gid=root,path=[],error=None) for m in names}
        while True:
            active={m:s['gid'] for m,s in states.items() if s['error'] is None and s['gid'] in nodes}
            if not active:break
            for gid in dict.fromkeys(active.values()):
                scores=node(gid)
                for m,requested in active.items():
                    if requested!=gid:continue
                    state=states[m]
                    if scores is None:state['error']='unavailable_score';continue
                    if len(state['path'])>len(nodes):state['error']='cycle';continue
                    label=int(scores[m]>0);state['path'].append(dict(gid=gid,label=label))
                    state['gid']=nodes[gid]['yes' if label else 'no']
        for m,state in states.items():
            correct=state['error'] is None and state['path']==case['gold_path'] and state['gid']==case['gold_leaf']
            append(work/'path_results.jsonl',dict(model=model,method=m,case_id=case['case_id'],source_id=case['source_id'],
              evaluation_group=case['evaluation_group'],correct=bool(correct),leaf_correct=state['gid']==case['gold_leaf'],
              predicted_path=state['path'],gold_path=case['gold_path'],predicted_leaf=state['gid'],gold_leaf=case['gold_leaf'],
              error=state['error'],status='ok' if state['error'] is None else 'unavailable',reused_result=False))
            written.add((case['case_id'],m))
        completed+=1
        if completed%20==0 or completed==len(pending):
            dump(work/'status.json',dict(state='running',model=model,gpu=gpu,cases_completed=completed,total_pending=len(pending),
              new_forwards=new_forwards,reused_nodes=reused,seconds=time.time()-start,
              peak_gib=torch.cuda.max_memory_allocated(gpu)/2**30,last_case=case['case_id']))
    dump(work/'done.json',dict(state='complete',model=model,gpu=gpu,cases=len(cases),new_forwards=new_forwards,reused_nodes=reused,
      seconds=time.time()-start,peak_gib=torch.cuda.max_memory_allocated(gpu)/2**30,counts=engine.counts if engine else {}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--limit',type=int)
    a=p.parse_args()
    try:
        # Concurrent controllers/resumes share a per-model lock; only one may write this shard.
        import fcntl
        lockroot=Path('/home/gcx/aster_cross_model_mega_20261111')/a.model
        lockroot.mkdir(parents=True,exist_ok=True)
        with (lockroot/'.execution.lock').open('w') as execution_lock:
            fcntl.flock(execution_lock,fcntl.LOCK_EX)
            run(a.model,a.gpu,a.shard,a.shards,a.limit)
    except BaseException as exc:
        out=OUT/a.model;out.mkdir(parents=True,exist_ok=True)
        dump(out/('failed_shard'+str(a.shard)+'.json'),dict(state='failed',error=str(exc),traceback=traceback.format_exc()))
        raise
