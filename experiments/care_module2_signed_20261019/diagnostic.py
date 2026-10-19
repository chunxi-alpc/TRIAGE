#!/usr/bin/env python3
"""Gold-applicability diagnostic, server execution only; no model weight training."""
from __future__ import annotations
import argparse, contextlib, csv, gzip, hashlib, importlib.metadata, json, math, os, sys, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import rotate_half
HERE=Path(__file__).resolve().parent
METHODS=['B4_positive','B4_signed','d_app_signed','score_bias_signed']
TAUS=[0.,.01,.05,.1]
REVISION='906bfd4b4dc7f14ee4320094d8b41684abff8539'
LOCATIONS=[(2,0),(20,5),(9,0),(35,0)]
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def save(p,x):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def read(p):return json.loads(Path(p).read_text())
def rows(p):
    op=gzip.open if str(p).endswith('.gz') else open
    with op(p,'rt') as f:return [json.loads(x) for x in f if x.strip()]
def tokenize_span(tok,text,span):
    enc=tok(text,return_offsets_mapping=True,add_special_tokens=False,return_tensors='pt')
    offsets=enc.pop('offset_mapping')[0].tolist()
    idx=[i for i,(s,e) in enumerate(offsets) if e>span[0] and s<span[1]]
    assert idx and idx==list(range(idx[0],idx[-1]+1))
    a,b=offsets[idx[0]][0],offsets[idx[-1]][1]
    assert text[a:span[0]].strip()=='' and text[span[1]:b].strip()==''
    return enc,idx,offsets
def causal_inputs(enc,device):
    ids=enc['input_ids'].to(device);n=ids.shape[1]
    mask=torch.full((1,1,n,n),torch.finfo(torch.bfloat16).min,device=device,dtype=torch.bfloat16)
    return dict(input_ids=ids,attention_mask=torch.triu(mask,diagonal=1),use_cache=False,logits_to_keep=1)
def score(model,tok,row,context=None):
    enc,_,_=tokenize_span(tok,row['prompt'],row['spans']['action'])
    with context if context is not None else contextlib.nullcontext():
        logits=model(**causal_inputs(enc,model.device)).logits[0,-1].float()
    ids=[tok.encode(' '+s,add_special_tokens=False) for s in 'ABCD']
    assert all(len(x)==1 for x in ids)
    vals=logits[[x[0] for x in ids]]
    return vals.softmax(-1).cpu().tolist(),vals.cpu().tolist()
def collect(model,tok,row,locations):
    enc,idx,_=tokenize_span(tok,row['prompt'],row['spans']['action'])
    full={};handles=[]
    for L in sorted({x[0] for x in locations}):
        for name in ['q','k']:
            def hook(mod,args,out,L=L,name=name):full.setdefault(L,{})[name]=out[0].detach().clone()
            handles.append(getattr(model.model.layers[L].self_attn,name+'_norm').register_forward_hook(hook))
    try:probs,logits=score(model,tok,row)
    finally:
        for h in handles:h.remove()
    return full,idx,probs,logits
@contextlib.contextmanager
def edit_keys(model,location,idx,changed,expected):
    L,h=location;calls=[0]
    def hook(mod,args,out):
        calls[0]+=1
        assert torch.equal(out[0,idx,h],expected),'pre-edit keys changed: stale cache or earlier hook'
        x=out.clone();x[0,idx,h]=changed;return x
    module=model.model.layers[L].self_attn.k_norm;before=len(module._forward_hooks)
    handle=module.register_forward_hook(hook)
    try:yield
    finally:
        handle.remove()
        assert len(module._forward_hooks)==before and calls[0]==1
@contextlib.contextmanager
def bias_scores(model,location,idx,bias):
    L,h=location;calls=[0]
    def hook(mod,args,kw):
        calls[0]+=1;mask=kw['attention_mask'];assert mask.ndim==4
        mask=mask.expand(1,32,-1,-1).float().clone()
        for t in idx:mask[0,4*h:4*h+4,t:,t]+=bias
        kw['attention_mask']=mask.to(model.dtype)
        return args,kw
    module=model.model.layers[L].self_attn;before=len(module._forward_pre_hooks)
    handle=module.register_forward_pre_hook(hook,with_kwargs=True)
    try:yield
    finally:
        handle.remove()
        assert len(module._forward_pre_hooks)==before and calls[0]==1
def score_geometry(model,capture,idx,location):
    L,h=location;q=capture[L]['q'];k=capture[L]['k'][idx,h];n,nq,d=q.shape
    cos,sin=model.model.rotary_emb(torch.zeros((1,n,model.config.hidden_size),device=q.device,dtype=q.dtype),torch.arange(n,device=q.device)[None])
    cos,sin=cos[0],sin[0]
    rotate=lambda x,c,s:x*c+rotate_half(x)*s
    qr=rotate(q,cos[:,None],sin[:,None]).float()[:,4*h:4*h+4]
    before=rotate(k,cos[idx],sin[idx]).float()
    valid=torch.stack([torch.arange(n,device=q.device)>=t for t in idx],dim=-1)[None].expand(4,-1,-1)
    def deltas(changed):
        dr=rotate(changed,cos[idx],sin[idx]).float()-before
        return (torch.einsum('tqd,sd->qts',qr,dr)/math.sqrt(d))[valid]
    return k,deltas
def calibrate(k,direction,deltas,target):
    if target==0:return k.clone(),dict(target_rms=0.,actual_rms=0.,scale=0.,relative_error=0.)
    direction=direction.float();assert float(direction.norm())>1e-10
    direction=direction/direction.norm()*k.float().norm()
    def at(a):
        ch=(k.float()+a*direction).to(k.dtype);ds=deltas(ch)
        return ch,ds,float(ds.square().mean().sqrt())
    lo,hi=0.,.01
    for _ in range(30):
        if at(hi)[2]>=target:break
        hi*=2
    else:raise RuntimeError('cannot bracket score perturbation')
    best=None
    for _ in range(24):
        mid=(lo+hi)/2;ch,ds,rms=at(mid);candidate=(abs(rms-target),mid,ch.clone(),ds.clone(),rms)
        if best is None or candidate[0]<best[0]:best=candidate
        if rms<target:lo=mid
        else:hi=mid
    err,scale,ch,ds,rms=best
    return ch,dict(target_rms=target,actual_rms=rms,relative_error=err/target,scale=scale,
        delta_score_mean=float(ds.mean()),delta_score_negative_fraction=float((ds<0).float().mean()),
        relative_key_norm=float((ch.float()-k.float()).norm()/k.float().norm()))
def pair_metrics(rs):
    pp=defaultdict(list)
    for r in rs:pp[r['pair_id']].append(r['correct'])
    assert pp and all(len(x)==2 for x in pp.values())
    return float(np.mean([all(x) for x in pp.values()])),float(np.mean([r['correct'] for r in rs]))
def record(r,method,tau,role,fold,probs,logits,audit):
    g='ABCD'.index(r['gold_answer']);p='ABCD'[int(np.argmax(probs))]
    return dict(case_id=r['case_id'],pair_id=r['pair_id'],group_id=r['group_id'],rule_id=r['rule_id'],
        applicable=r['applicable'],gold_answer=r['gold_answer'],prediction=p,correct=int(p==r['gold_answer']),
        misuse=int(not r['applicable'] and p==r['inapplicable_action']),candidate_probabilities=probs,candidate_logits=logits,
        gold_probability_margin=probs[g]-max(v for i,v in enumerate(probs) if i!=g),
        gold_logit_margin=logits[g]-max(v for i,v in enumerate(logits) if i!=g),
        method=method,tau=tau,role=role,fold=fold,disturbance=audit)
def audit_data(args,tok,data):
    from support import classify
    assert len(data)==384 and len({r['case_id'] for r in data})==384 and len({r['group_id'] for r in data})==12
    pairs=defaultdict(list);audit=[]
    for r in data:
        assert classify(r['kind'],r['value'])==r['gold_label']==r['option_map'][r['gold_answer']]
        assert r['applicable']==(r['gold_answer']==r['inapplicable_action'])
        assert r['prompt'][slice(*r['spans']['action'])]==r['target_action']
        assert r['spans']['action'][1]<r['prompt'].index('\nOptions:')
        enc,idx,_=tokenize_span(tok,r['prompt'],r['spans']['action']);assert enc['input_ids'].shape[1]<=1024
        item=dict(case_id=r['case_id'],indices=idx,token_ids=enc['input_ids'][0,idx].tolist())
        audit.append(item);pairs[r['pair_id']].append((r,item))
    for pp in pairs.values():
        assert len(pp)==2 and {r['applicable'] for r,_ in pp}=={True,False}
        assert pp[0][1]['token_ids']==pp[1][1]['token_ids'] and pp[0][0]['option_map']==pp[1][0]['option_map']
    return audit
def setup(args):
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42);torch.backends.cuda.matmul.allow_tf32=False
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in {'0','1','2','3','6','7'}
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    manifest=read(args.artifacts/'results/manifest.json');assert manifest['model_revision']==REVISION
    weights={x.name:digest(x) for x in args.model.glob('*.safetensors')}
    assert weights=={k:v['sha256'] for k,v in manifest['weights'].items()}
    data_path=args.artifacts/'data/clinical/candidates.jsonl'
    assert digest(data_path)==read(args.artifacts/'results/obs1_clinical/manifest.json')['data_sha256']
    args.output.mkdir(parents=True,exist_ok=False)
    save(args.output/'provenance.json',dict(model_revision=REVISION,weights=weights,data_sha256=digest(data_path),
        source_sha256=digest(__file__),old_source_sha256=read(HERE/'pr3_source_hashes.json'),support_sha256=digest(HERE/'support.py'),
        physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],versions={k:importlib.metadata.version(k) for k in ['torch','transformers','numpy']},
        no_generation_cache=True,gold_applicability_oracle=True))
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    c=model.config;assert (c.num_hidden_layers,c.num_attention_heads,c.num_key_value_heads,c.head_dim)==(36,32,8,128)
    data=rows(data_path);save(args.output/'data_span_audit.json',audit_data(args,tok,data))
    return model,tok,data
def main_fold(args,model,tok,data):
    fold=args.fold;old=args.archive/f'clinical_fold{fold}';split=read(old/f'fold{fold}/split.json')
    original=read(args.artifacts/f'results/obs1_clinical/fold{fold}/manifest.json')
    for role in ['train','dev','test']:assert split[role]==sorted(original[role+'_families'])
    assert not(set(split['train'])&set(split['dev']) or set(split['train'])&set(split['test']) or set(split['dev'])&set(split['test']))
    save(args.output/'split.json',split)
    selection=read(old/f'fold{fold}/selection.json')['B4'];cid=selection['config']
    metadata=read(old/f'fold{fold}/operators.json')[cid];location=tuple(metadata['location']);rank=metadata['rank']
    cached=torch.load(old/'baseline_target_keys.pt',weights_only=False)
    from support import training_matrices,contrastive_basis
    train=[r for r in data if r['group_id'] in split['train']]
    fresh={};cache_errors=[]
    for r in train:
        cap,idx,_,_=collect(model,tok,r,[location]);k=cap[location[0]]['k'][idx,location[1]].float().cpu().numpy()
        cache_errors.append(float(np.max(np.abs(k-cached[r['case_id']][location]))));fresh[r['case_id']]={location:k}
    assert max(cache_errors)==0.,'training cache mismatch'
    raw,signal,background,controls=training_matrices(train,fresh,location)
    basis=contrastive_basis(signal,background,rank);P=torch.from_numpy(basis@basis.T).float()
    oldP=torch.load(old/f'fold{fold}/operators.pt',weights_only=True)[cid]
    assert float((P-oldP).abs().max())<1e-6
    d=torch.from_numpy(signal.mean(0)).float();assert float(d.norm())>1e-8
    torch.save(dict(B4_projector=P,d_app=d),args.output/'operators.pt')
    save(args.output/'training_audit.json',dict(train_case_ids=sorted(fresh),controls=controls,maximum_cache_error=max(cache_errors),
        B4_operator_error=float((P-oldP).abs().max()),symmetry_error=float((P-P.T).abs().max()),
        idempotence_error=float((P@P-P).abs().max()),old_B4_selection=selection,location=location,rank=rank,d_app_norm=float(d.norm())))
    out=[];stream=(args.output/'predictions.jsonl').open('w',buffering=1)
    def append(x):out.append(x);stream.write(json.dumps(x,allow_nan=False)+'\n')
    try:
        for role in ['dev','test']:
            for r in [r for r in data if r['group_id'] in split[role]]:
                cap,idx,bp,bl=collect(model,tok,r,[location]);k,ds=score_geometry(model,cap,idx,location)
                append(record(r,'baseline',0.,role,fold,bp,bl,{}));s=1. if r['applicable'] else -1.
                for method in METHODS:
                    direction=k.float()@P.to(k.device)
                    if method=='B4_signed':direction*=s
                    elif method=='d_app_signed':direction=d.to(k.device)[None].expand_as(k)*s
                    for tau in TAUS:
                        if tau==0:probs,logits,audit=bp,bl,dict(target_rms=0.,actual_rms=0.)
                        elif method=='score_bias_signed':
                            probs,logits=score(model,tok,r,bias_scores(model,location,idx,s*tau))
                            audit=dict(target_rms=tau,actual_rms=abs(float(torch.tensor(tau,dtype=model.dtype))),relative_error=abs(float(torch.tensor(tau,dtype=model.dtype))-tau)/tau,delta_score_mean=float(torch.tensor(s*tau,dtype=model.dtype)),delta_score_negative_fraction=float(s<0))
                        else:
                            changed,audit=calibrate(k,direction,ds,tau)
                            assert audit['relative_error']<.15,('score calibration failed',r['case_id'],method,tau,audit)
                            probs,logits=score(model,tok,r,edit_keys(model,location,idx,changed,k))
                        append(record(r,method,tau,role,fold,probs,logits,audit))
            if role=='dev':
                candidates=[]
                for tau in TAUS:
                    mm=[pair_metrics([r for r in out if r['role']=='dev' and r['method']==m and r['tau']==tau]) for m in METHODS]
                    candidates.append((float(np.mean([x[0] for x in mm])),float(np.mean([x[1] for x in mm])),-tau,tau))
                best=max(candidates)
                own={m:max((*pair_metrics([r for r in out if r['role']=='dev' and r['method']==m and r['tau']==tau]),-tau,tau) for tau in TAUS)[-1] for m in METHODS}
                save(args.output/'selection.json',dict(common_tau=best[-1],common_dev_pair_accuracy=best[0],method_specific_tau=own,
                    selection='dev pair accuracy, then case accuracy, then smaller tau; common tau maximizes mean across four methods',
                    candidates=candidates,test_labels_used=False,head_rank_source='frozen PR3 B4 dev selection'))
                print('DEV_COMPLETE',fold,flush=True)
    finally:stream.close()
    assert len(out)==(64+128)*17
def validate(args,model,tok,data):
    errors=[];old_errors=[];zero_errors=[]
    stream=(args.output/'reproduction.jsonl').open('w',buffering=1)
    for f in range(3):
        old=args.archive/f'clinical_fold{f}';split=read(old/f'fold{f}/split.json');sel=read(old/f'fold{f}/selection.json')['B4']
        meta=read(old/f'fold{f}/operators.json')[sel['config']];location=tuple(meta['location'])
        P=torch.load(old/f'fold{f}/operators.pt',weights_only=True)[sel['config']].to(model.device)
        cached=torch.load(old/'baseline_target_keys.pt',weights_only=False)
        archived=rows(old/f'fold{f}/predictions.jsonl')
        lookup={(r['case_id'],r['config'],r['mode']):r for r in archived if r['role']=='test'}
        for r in [r for r in data if r['group_id'] in split['test']]:
            cap,idx,bp,bl=collect(model,tok,r,LOCATIONS)
            err=max(float(np.max(np.abs(cap[L]['k'][idx,h].float().cpu().numpy()-cached[r['case_id']][L,h]))) for L,h in LOCATIONS)
            assert err==0.;errors.append(err)
            base=lookup[r['case_id'],'baseline','baseline'];assert bp==base['candidate_probabilities']
            k=cap[location[0]]['k'][idx,location[1]];u=k.float()@P
            delta=sel['epsilon']*k.float().norm()/u.norm()*u if sel['epsilon'] else torch.zeros_like(u)
            changed=(k.float()+delta).to(k.dtype)
            pp,ll=score(model,tok,r,edit_keys(model,location,idx,changed,k))
            expected=lookup[r['case_id'],sel['config'],'amplify']['candidate_probabilities']
            e=max(abs(x-y) for x,y in zip(pp,expected));assert e==0.;old_errors.append(e)
            stream.write(json.dumps(dict(case_id=r['case_id'],fold=f,baseline_probabilities=bp,B4_probabilities=pp,cache_error=err,reproduction_error=e))+'\n')
            if len(errors)==1:
                zz,_=score(model,tok,r,bias_scores(model,location,idx,0.));zero_errors.append(max(abs(x-y) for x,y in zip(zz,bp)))
                assert zero_errors[-1]==0.
    stream.close()
    save(args.output/'validation.json',dict(cases=len(errors),key_cache_max_error=max(errors),old_B4_max_probability_error=max(old_errors),
        zero_bias_max_error=max(zero_errors),all_hooks_removed=True,original_B4_has_online_sign_control=False,original_B4_positive_projector_amplification=True))
def main():
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=[0,1,2]);p.add_argument('--validate',action='store_true')
    p.add_argument('--artifacts',type=Path,default=Path('/home/gcx/care_20261016/CARE_Codex_Pilot'))
    p.add_argument('--legacy',type=Path,default=Path('/home/gcx/care_module2_followup_20261017'))
    p.add_argument('--archive',type=Path,default=Path('/home/gcx/care_module2_followup_20261017/results/module2_followup/parallel_v3'))
    p.add_argument('--model',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    assert args.validate or args.fold is not None
    args.model=args.model or args.artifacts/'models/Qwen3-4B-Base';start=time.time()
    model,tok,data=setup(args);torch.cuda.reset_peak_memory_stats()
    if args.validate:validate(args,model,tok,data)
    else:main_fold(args,model,tok,data)
    save(args.output/'status.json',dict(status='completed_real_inference',seconds=time.time()-start,
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
        physical_gpu=os.environ['CUDA_VISIBLE_DEVICES']))
if __name__=='__main__':
    with torch.inference_mode():main()

