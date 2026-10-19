#!/usr/bin/env python3
"""Real Qwen3-4B-Base CARE pilot; no simulated observations."""
from __future__ import annotations
import argparse, contextlib, csv, importlib.metadata, inspect, json, os, pathlib, random, subprocess, sys, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, rotate_half
from transformers.models.qwen3 import modeling_qwen3 as qwen_impl
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'external/SEKA'))
from src.model.projection_builder_base import ProjectionBuilderBase
from src.model.seka_llm import SEKALLM
from pastalib.pasta import PASTA
MODEL_ID='Qwen/Qwen3-4B-Base'
STATES=['RA','RN','IA','IN','RA2','RN2']
SEED=42
def dump(path,obj):
    path=pathlib.Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str))
def log(message):
    print(time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),message,flush=True)
def tokenize_span(tok,text,span):
    enc=tok(text,return_offsets_mapping=True,add_special_tokens=False,return_tensors='pt')
    offsets=enc.pop('offset_mapping')[0].tolist()
    idx=[i for i,(s,e) in enumerate(offsets) if e>span[0] and s<span[1]]
    assert idx and idx==list(range(idx[0],idx[-1]+1))
    a,b=offsets[idx[0]][0],offsets[idx[-1]][1]
    assert text[a:span[0]].strip()=='' and text[span[1]:b].strip()==''
    return enc,idx,offsets
def causal_inputs(enc,device):
    ids=enc['input_ids'].to(device); n=ids.shape[1]
    mask=torch.full((1,1,n,n),torch.finfo(torch.bfloat16).min,device=device,dtype=torch.bfloat16)
    return {'input_ids':ids,'attention_mask':torch.triu(mask,diagonal=1),'use_cache':False,'logits_to_keep':1}
class CareBuilder(ProjectionBuilderBase):
    """Official SVD and projectors; adapted prompts and streaming hooks only."""
    def __init__(self,model,tok,rows):
        self.model=model; self.tokenizer=tok; self.rows=rows
        self.layers=list(range(36)); self.device='cuda'; self.model_path='Qwen3-4B-Base'
        self.top_pct=.9; self.min_diff=0.; self.feature=None
    def iter_examples(self): return iter(self.rows)
    def get_triplets(self,ex):
        return [(ex['patient']+'\nGuideline: '+ex['modified_rule'],
            'Which route does the supplied decision protocol select?',
            'select Route Cedar',"What is the patient's age in years?")]
    def build(self,out):
        buffers=[[[[] for _ in range(8)] for _ in range(36)] for _ in range(3)]
        audit=[]
        for ni,r in enumerate(self.rows):
            ctx,rel,ans,irr=self.get_triplets(r)[0]
            texts=[f'Context: {ctx} ',f'Question: {rel}\nContext: {ctx}',f'Question: {irr}\nContext: {ctx}']
            token_ids=[]
            for c,text in enumerate(texts):
                span=[text.index(ans),text.index(ans)+len(ans)]
                enc,idx,_=tokenize_span(self.tokenizer,text,span)
                token_ids.append(enc['input_ids'][0,idx].tolist())
                captured={}; handles=[]
                for L in self.layers:
                    def h(mod,ins,out,L=L): captured[L]=out[0,idx].float().cpu().clone()
                    handles.append(self.model.model.layers[L].self_attn.k_norm.register_forward_hook(h))
                try: self.model(**causal_inputs(enc,self.model.device))
                finally:
                    for h in handles: h.remove()
                for L in self.layers:
                    for kv in range(8): buffers[c][L][kv].append(captured[L][:,kv,:])
                if ni==0 and c==1:
                    # Match the explicit causal mask used for every CARE forward.
                    # The official helper otherwise chooses the mask-free SDPA kernel,
                    # whose BF16 rounding can differ after many decoder layers.
                    def force_same_mask(mod,args,kw):
                        kw['attention_mask']=causal_inputs(enc,self.model.device)['attention_mask']
                        kw['logits_to_keep']=1
                        return args,kw
                    handle=self.model.register_forward_pre_hook(force_same_mask,with_kwargs=True)
                    try: ref=self.extract_keys(self.model,self.tokenizer,text,idx,[0,26,35],None)
                    finally: handle.remove()
                    error=max((ref[j].cpu()-captured[L][:,kv,:]).abs().max().item()
                        for j,(L,kv) in enumerate((L,kv) for L in [0,26,35] for kv in range(8)))
                    assert error<1e-6,error
                    dump(ROOT/'results/checks/official_key_extraction.json',{'max_abs_error':error,'tolerance':1e-6})
                audit.append({'group_id':r['group_id'],'condition':c,'text':text,'token_indices':idx,'token_ids':token_ids[-1]})
            assert token_ids[0]==token_ids[1]==token_ids[2]
            if ni%12==0: log(f'projection extraction {ni+1}/{len(self.rows)}')
        dump(ROOT/'results/obs2/projection_span_audit.json',audit)
        self._compute_and_save_traditional(*buffers,36,8,str(out))
        torch.save({'H':buffers[0],'H_relevant':buffers[1],'H_irrelevant':buffers[2]},out/'builder_keys.pt')
class Runner:
    def __init__(self,model,tok,proj):
        self.model=model; self.tok=tok; self.proj=proj; self.layers=list(range(26,36))
        self.seka=object.__new__(SEKALLM)
        self.seka.model=model; self.seka.tok=tok; self.seka._hooks=[]
        self.seka.pos_pt=str(proj/'last10_pos.pt'); self.seka.neg_pt=str(proj/'last10_neg.pt')
        self.seka.layers='last10'; self.seka.amplify_pos=.8; self.seka.amplify_neg=.2; self.seka.feature_function=None
        self.answer_ids=[]
        for x in 'ABCD':
            ids=tok.encode(' '+x,add_special_tokens=False)
            assert len(ids)==1,(x,ids)
            self.answer_ids.append(ids[0])
    def forward(self,row,method='baseline',strength=0.,text_key='prompt',capture=False):
        text=row[text_key]
        if text_key=='prompt': span=row['spans']['action']
        else:
            needle='select Route Amber' if text_key=='original_prompt' else 'Options:'
            p=text.index(needle); span=[p,p+len(needle)]
        enc,idx,offsets=tokenize_span(self.tok,text,span)
        inp=causal_inputs(enc,self.model.device); n=inp['input_ids'].shape[1]
        assert n<=1024
        captured={}; handles=[]
        rotary_original=None
        if capture:
            for L in range(36):
                def kh(mod,ins,out,L=L):
                    captured.setdefault(L,{})['span_key']=out[0,idx].float().cpu().clone()
                    if L>=26: captured[L]['key']=out[0].float().cpu().clone()
                handles.append(self.model.model.layers[L].self_attn.k_norm.register_forward_hook(kh))
                if L>=26:
                    def qh(mod,ins,out,L=L): captured.setdefault(L,{})['query']=out[0,-1].float().cpu().clone()
                    handles.append(self.model.model.layers[L].self_attn.q_norm.register_forward_hook(qh))
            if row['split']=='test' and row['state'] in ['RA','RN']:
                rotary_original=qwen_impl.apply_rotary_pos_emb
                counter=[0]
                def capture_rotary(*a,**kw):
                    qr,kr=rotary_original(*a,**kw)
                    L=counter[0]; counter[0]+=1
                    if L>=26:
                        captured.setdefault(L,{})['query_actual_rope']=qr[0,:,-1].float().cpu().clone()
                        captured[L]['key_actual_rope']=kr[0].transpose(0,1).float().cpu().clone()
                    return qr,kr
                qwen_impl.apply_rotary_pos_emb=capture_rotary
        if method=='seka':
            sm=torch.zeros((1,n),dtype=torch.bool,device=self.model.device); sm[:,idx]=True
            self.seka.attach_projection(steer_mask_tensor=sm,amplify_pos=strength,amplify_neg=.25*strength,silence=True)
        elif method=='pasta':
            pasta=PASTA(self.model,self.tok,head_config=self.layers,alpha=float(np.exp(-strength)),scale_position='exclude')
            from functools import partial
            for L in self.layers:
                fn=partial(pasta.edit_multisection_attention,head_idx=list(range(32)),
                    token_ranges=[torch.tensor([[idx[0],idx[-1]+1]])],input_len=n)
                handles.append(self.model.model.layers[L].self_attn.register_forward_pre_hook(fn,with_kwargs=True))
        try:
            out=self.model(**inp); logits=out.logits[0,-1].float()
            chosen=logits[self.answer_ids]; probs=chosen.softmax(-1).cpu().numpy()
            full_probs=logits.softmax(-1)[self.answer_ids].cpu().numpy()
            prediction='ABCD'[int(chosen.argmax())]
        finally:
            if rotary_original is not None: qwen_impl.apply_rotary_pos_emb=rotary_original
            self.seka.remove_projection()
            for h in handles: h.remove()
        meta={'prediction':prediction,'candidate_probabilities':probs.tolist(),'full_vocabulary_candidate_mass':float(full_probs.sum()),
            'n_tokens':n,'target_token_indices':idx,'target_token_ids':enc['input_ids'][0,idx].tolist(),
            'target_offsets':[offsets[i] for i in idx],'decision_query_position':n-1}
        return meta,captured
def check_pasta(model,tok):
    pasta=PASTA(model,tok,head_config=[26],alpha=.1,scale_position='exclude')
    n=8; mask=torch.triu(torch.full((1,1,n,n),-1e4,dtype=torch.float32),diagonal=1)
    _,kw=pasta.edit_multisection_attention(model.model.layers[26].self_attn,(),{'attention_mask':mask},
        head_idx=[0,1,2,3],token_ranges=[torch.tensor([[2,4]])],input_len=n)
    altered=kw['attention_mask']; ref=mask.expand(1,32,n,n).clone()
    ref[:,0:4,:,:2]+=np.log(.1); ref[:,0:4,:,4:]+=np.log(.1)
    assert torch.allclose(altered,ref,atol=1e-6,rtol=0)
    torch.manual_seed(SEED); scores=torch.randn((1,32,n,n))
    a=(scores+mask).softmax(-1); weighted=a.clone()
    weighted[:,0:4,:,:2]*=.1; weighted[:,0:4,:,4:]*=.1; weighted/=weighted.sum(-1,keepdim=True)
    error=((scores+altered).softmax(-1)-weighted).abs().max().item(); assert error<1e-6
    dump(ROOT/'results/checks/pasta_equivalence.json',{'implementation':'official PASTA unchanged',
        'mask_max_error':float((altered-ref).abs().max()),'softmax_max_error':error,'tolerance':1e-6})
def rope(x,cos,sin): return x*cos+rotate_half(x)*sin
def write_o3(row,captured,meta,pos,neg,model,writer,record):
    idx=meta['target_token_indices']; n=meta['n_tokens']; d=128
    positions=torch.arange(n,device=model.device).unsqueeze(0)
    cos,sin=model.model.rotary_emb(torch.zeros((1,n,2560),device=model.device,dtype=torch.float32),positions)
    cos=cos[0].cpu().bfloat16().float(); sin=sin[0].cpu().bfloat16().float()
    rawdir=ROOT/'results/obs3/full_qk'; rawdir.mkdir(exist_ok=True)
    np.savez_compressed(rawdir/(row['case_id']+'.npz'),
        keys=np.stack([captured[L]['key'].numpy() for L in range(26,36)]),
        queries=np.stack([captured[L]['query'].numpy() for L in range(26,36)]),
        keys_actual_rope=np.stack([captured[L]['key_actual_rope'].numpy() for L in range(26,36)]),
        queries_actual_rope=np.stack([captured[L]['query_actual_rope'].numpy() for L in range(26,36)]),
        cos=cos.numpy(),sin=sin.numpy(),target_indices=idx,query_position=n-1)
    record['q_pre'].append(np.stack([captured[L]['query'].numpy() for L in range(26,36)]))
    record['k_pre_target'].append(np.stack([captured[L]['span_key'].numpy() for L in range(26,36)]))
    record['token_positions'].append(idx); record['query_positions'].append(n-1); record['case_ids'].append(row['case_id'])
    record['cos_target'].append(cos[idx].numpy()); record['sin_target'].append(sin[idx].numpy())
    record['cos_query'].append(cos[-1].numpy()); record['sin_query'].append(sin[-1].numpy())
    for L in range(26,36):
        keys=captured[L]['key']; q=captured[L]['query']
        q_rope=rope(q.bfloat16(),cos[-1].bfloat16(),sin[-1].bfloat16()).float()
        assert torch.equal(q_rope,captured[L]['query_actual_rope'])
        k0bf=rope(keys.bfloat16(),cos[:,None,:].bfloat16(),sin[:,None,:].bfloat16()).float()
        assert torch.equal(k0bf,captured[L]['key_actual_rope'])
        actual_logits=torch.einsum('qd,tqd->qt',q_rope,k0bf.repeat_interleave(4,dim=1))/np.sqrt(d)
        actual_p0=actual_logits.softmax(-1)
        k_rope=rope(keys,cos[:,None,:],sin[:,None,:])
        logits=torch.einsum('qd,tqd->qt',q_rope,k_rope.repeat_interleave(4,dim=1))/np.sqrt(d)
        p0=logits.softmax(-1)
        variants={'positive_only':pos[L-26], 'official_posneg':(pos[L-26]+.25*neg[L-26])/2}
        for variant,P in variants.items():
            u=torch.einsum('thd,hdk->thk',keys[idx],P)
            u_rope=rope(u,cos[idx,None,:],sin[idx,None,:])
            z=torch.einsum('qd,tqd->qt',q_rope,u_rope.repeat_interleave(4,dim=1))/np.sqrt(d)
            for weight_type in ['unit','continuous']:
                distance=abs(row['patient_value']-row['threshold'])
                c=1. if weight_type=='unit' else distance/(distance+.1*row['threshold'])
                r=1. if weight_type=='unit' else .75
                omega=(1 if row['applicable'] else -1)*c*r; lam=.8
                edited=keys.clone(); edited[idx]+=lam*omega*u
                after_rope=rope(edited,cos[:,None,:],sin[:,None,:])
                after=torch.einsum('qd,tqd->qt',q_rope,after_rope.repeat_interleave(4,dim=1))/np.sqrt(d)
                da=after[:,idx]-logits[:,idx]; expected=lam*omega*z
                error=float((da-expected).abs().max()); assert error<=1e-4,(L,error)
                p1=after.softmax(-1)
                edited_bf=keys.bfloat16().clone(); edited_bf[idx]=(keys[idx]+lam*omega*u).bfloat16()
                krbf=rope(edited_bf,cos[:,None,:].bfloat16(),sin[:,None,:].bfloat16()).float()
                after_actual=torch.einsum('qd,tqd->qt',q_rope,krbf.repeat_interleave(4,dim=1))/np.sqrt(d)
                da_bf=after_actual-actual_logits; actual_p1=after_actual.softmax(-1)
                for qh in range(32):
                    for j,t in enumerate(idx):
                        writer.writerow({'group_id':row['group_id'],'case_id':row['case_id'],'layer':L,'kv_head':qh//4,
                            'query_head':qh,'token_idx':t,'omega':omega,'z':float(z[qh,j]),'a_before':float(logits[qh,t]),
                            'a_after':float(after[qh,t]),'delta_a':float(da[qh,j]),'lambda':lam,'split':row['split'],'model':MODEL_ID,
                            'projection_variant':variant,'weight_type':weight_type,'confidence_c':c,'relevance_r':r,
                            'p_before':float(p0[qh,t]),'p_after':float(p1[qh,t]),'delta_p':float(p1[qh,t]-p0[qh,t]),
                            'formula_abs_error':float(abs(da[qh,j]-expected[qh,j])),'delta_a_bf16':float(da_bf[qh,t]),
                            'a_before_bf16':float(actual_logits[qh,t]),'a_after_bf16':float(after_actual[qh,t]),
                            'p_before_bf16':float(actual_p0[qh,t]),'p_after_bf16':float(actual_p1[qh,t])})
def pred_record(row,method,strength,meta,prior=None,variant='modified'):
    pred=meta['prediction']; gold=row['gold_answer'] if variant=='modified' else row['original_action']
    return {'group_id':row['group_id'],'case_id':row['case_id'],'method':method,'strength':strength,'variant':variant,
        'subset':'app' if row['applicable'] else 'non','state':row['state'],'correct':int(pred==gold),
        'misapplied':int(not row['applicable'] and pred==row['inapplicable_action']),
        'prior_reversion':int(prior is not None and pred==prior),'prior_eligible':False,
        'synthetic_prior_disagreement':int(prior is not None and prior!=gold),'split':row['split'],'model':MODEL_ID,'seed':SEED,
        'source_type':row['source_type'],'validated_clinical_conflict':False,'gold_answer':gold,**meta}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',default='models/Qwen3-4B-Base'); p.add_argument('--smoke',action='store_true'); args=p.parse_args()
    os.chdir(ROOT); start=time.time()
    for d in ['obs1','obs2','obs3','checks','logs','projections']: (ROOT/'results'/d).mkdir(parents=True,exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    tok=AutoTokenizer.from_pretrained(args.model)
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    cfg=model.config
    assert (cfg.num_hidden_layers,cfg.num_attention_heads,cfg.num_key_value_heads,cfg.head_dim)==(36,32,8,128)
    dump('results/checks/actual_model_config.json',cfg.to_dict())
    pathlib.Path('results/checks/qwen3_attention_source.py').write_text(inspect.getsource(Qwen3Attention.forward))
    rows=[json.loads(x) for x in pathlib.Path('data/derived/pilot.jsonl').read_text().splitlines()]
    rows=sorted(rows,key=lambda r:(r['group_id'],STATES.index(r['state'])))
    train=[r for r in rows if r['split']=='train' and r['state'] in ['RA','RN']]
    if args.smoke:
        gids=[sorted({r['group_id'] for r in train})[0],
              sorted({r['group_id'] for r in rows if r['split']=='test'})[0]]
        train=[r for r in train if r['group_id'] in gids]; rows=[r for r in rows if r['group_id'] in gids]
    check_pasta(model,tok)
    log(f'model ready; parameters={sum(p.numel() for p in model.parameters())}; groups={len(set(r["group_id"] for r in rows))}')
    proj=ROOT/('results/projections_smoke' if args.smoke else 'results/projections'); proj.mkdir(exist_ok=True)
    CareBuilder(model,tok,train).build(proj)
    for name in ['pos','neg']:
        dat=torch.load(proj/f'Qwen3-4B-Base_{name}_proj.pt',weights_only=True)
        assert dat['layers']==list(range(36))
        torch.save({'layers':list(range(26,36)),'proj':dat['proj'][26:]},proj/f'last10_{name}.pt')
    pos=torch.load(proj/'last10_pos.pt',weights_only=True)['proj']; neg=torch.load(proj/'last10_neg.pt',weights_only=True)['proj']
    run=Runner(model,tok,proj); keys=[]; audit=[]; raw=[]
    fields=['group_id','case_id','layer','kv_head','query_head','token_idx','omega','z','a_before','a_after','delta_a','lambda','split','model',
        'projection_variant','weight_type','confidence_c','relevance_r','p_before','p_after','delta_p','formula_abs_error','delta_a_bf16',
        'a_before_bf16','a_after_bf16','p_before_bf16','p_after_bf16']
    records={k:[] for k in ['q_pre','k_pre_target','token_positions','query_positions','case_ids','cos_target','sin_target','cos_query','sin_query']}
    logfile=open('results/obs3/logits.csv','w',newline=''); writer=csv.DictWriter(logfile,fieldnames=fields); writer.writeheader()
    cached={}
    for i,row in enumerate(rows):
        meta,cap=run.forward(row,capture=True)
        keys.append(np.stack([cap[L]['span_key'].mean(0).numpy() for L in range(36)]))
        raw.append(np.stack([cap[L]['span_key'].numpy() for L in range(36)]))
        audit.append({'case_id':row['case_id'],'group_id':row['group_id'],'state':row['state'],**meta})
        cached[row['case_id']]=meta
        if row['split']=='test' and row['state'] in ['RA','RN']: write_o3(row,cap,meta,pos,neg,model,writer,records)
        if i%60==0: log(f'keys/base forwards {i+1}/{len(rows)}')
    logfile.close()
    gids=sorted(set(r['group_id'] for r in rows)); splits=[next(r['split'] for r in rows if r['group_id']==g) for g in gids]
    keyarr=np.stack(keys).reshape(len(gids),6,36,8,128); tokenarr=np.stack(raw).reshape(len(gids),6,36,-1,8,128)
    for i in range(0,len(rows),6): assert len({tuple(a['target_token_ids']) for a in audit[i:i+6]})==1
    np.savez_compressed('results/obs2/key_embeddings.npz',keys=keyarr[:,:4],same_side_keys=keyarr[:,4:],
        token_keys=tokenarr,group_ids=gids,layer_ids=np.arange(36),states=np.array(STATES[:4]),split=splits)
    dump('results/obs2/token_alignment.json',audit)
    if records['case_ids']: np.savez_compressed('results/obs3/qk_raw.npz',**{k:np.array(v) for k,v in records.items()})
    predictions=[]; prior_rows=[]; relevant=[r for r in rows if r['state'] in ['RA','RN']]; priors={}
    for row in relevant:
        pm,_=run.forward(row,text_key='closedbook_prompt'); prior=pm['prediction']; priors[row['case_id']]=prior
        paraphrase=dict(row); paraphrase['closedbook_prompt']=row['closedbook_prompt'].replace(
            'Which route does the supplied decision protocol select?','Which option should be selected for this patient?')
        pp,_=run.forward(paraphrase,text_key='closedbook_prompt')
        prior_rows.append({'case_id':row['case_id'],'group_id':row['group_id'],'split':row['split'],
            'prior_prediction':prior,'paraphrase_prediction':pp['prediction'],'stable':prior==pp['prediction'],
            'gold_answer':row['gold_answer'],'prior_disagreement':prior!=row['gold_answer'],
            'clinical_source_verified':False,'conflict_eligible':False,'candidate_probabilities':pm['candidate_probabilities']})
        predictions.append(pred_record(row,'baseline',0,cached[row['case_id']],prior))
        om,_=run.forward(row,text_key='original_prompt'); predictions.append(pred_record(row,'baseline',0,om,prior,variant='original'))
    grid=[.25,.5,.8,1.]
    for method in ['seka','pasta']:
        for strength in grid:
            for row in relevant:
                if row['split']!='dev' and not args.smoke: continue
                meta,_=run.forward(row,method,strength); predictions.append(pred_record(row,method,strength,meta,priors[row['case_id']]))
            log(f'dev {method} strength={strength} done')
    import pandas as pd
    df=pd.DataFrame(predictions); selected={}
    for method in ['seka','pasta']:
        scores=[]
        for strength in grid:
            d=df[(df.method==method)&(df.strength==strength)&(df.split=='dev')]
            if args.smoke: d=df[(df.method==method)&(df.strength==strength)]
            a=float(d[d.subset=='app'].correct.mean()); b=float(d[d.subset=='non'].correct.mean())
            scores.append((2*a*b/(a+b) if a+b else 0.,-strength,strength))
        selected[method]=max(scores)[2]
        for row in relevant:
            if row['split']=='dev' and not args.smoke: continue
            meta,_=run.forward(row,method,selected[method]); predictions.append(pred_record(row,method,selected[method],meta,priors[row['case_id']]))
        log(f'{method} chosen strength={selected[method]} complete')
    pd.DataFrame(predictions).to_csv('results/obs1/predictions.csv',index=False)
    pd.DataFrame(prior_rows).to_csv('results/obs1/prior_screen.csv',index=False)
    dump('results/obs1/selected_hyperparameters.json',{'criterion':'development H_Acc, ties smaller strength','strengths':selected,
        'seka_variant':'official positive+negative: (lambda P_pos + 0.25 lambda P_neg)/2',
        'pasta_variant':'official PASTA exclude alpha=exp(-strength); explicit causal mask',
        'intervention_layers':list(range(26,36)),'kv_heads':list(range(8)),'query_heads':list(range(32)),
        'selection_strategy':'single-token conditional probability for space+A/B/C/D; identical fixed candidate order'})
    metadata_dir=pathlib.Path(args.model)/'.cache/huggingface/download'; revisions=[]
    if metadata_dir.exists(): revisions=sorted({f.read_text().splitlines()[0] for f in metadata_dir.rglob('*.metadata')})
    manifest={'model':MODEL_ID,'model_revision':revisions,'model_path':str(pathlib.Path(args.model).resolve()),
        'seka_commit':subprocess.check_output(['git','-C','external/SEKA','rev-parse','HEAD'],text=True).strip(),
        'seka_unmodified':subprocess.check_output(['git','-C','external/SEKA','status','--porcelain'],text=True).strip()=='',
        'seed':SEED,'versions':{x:importlib.metadata.version(x) for x in ['torch','transformers','numpy','scipy','scikit-learn','matplotlib','pandas']},
        'gpu':torch.cuda.get_device_name(),'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'dtype':'bfloat16',
        'attention':'sdpa with explicit causal 4D mask for all methods','seconds':time.time()-start,
        'peak_memory_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'peak_memory_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,
        'groups':len(gids),'cases':len(rows),'clinical_conflict_eligible':0,'synthetic_only':True,
        'o3_formula_tolerance':1e-4,'o3_near_zero_eps':1e-6,'projection_training_groups':sorted(set(r['group_id'] for r in train)),
        'config':cfg.to_dict(),'smoke':args.smoke}
    dump('results/manifest.json',manifest)
    log(f'COMPLETE elapsed={time.time()-start:.1f}s peak={manifest["peak_memory_allocated_GiB"]:.2f}GiB')
if __name__=='__main__':
    with torch.inference_mode(): main()

