"""Frozen-Qwen scorer, local residual coordinates, and actual post-RoPE score bias."""
from pathlib import Path
from contextlib import contextmanager
import hashlib,json,os,time,platform,subprocess
import torch
from care_m2_inc.model import load_qwen, _layers
from care_m2_inc.io import write_json,read_json,sha256_file
from .data import digest

LAYER=24
SEED=261014

def fingerprint(path):
    p=Path(path)
    files=sorted([*p.glob('*.safetensors'),*p.glob('*token*'),p/'config.json'])
    return {f.name:sha256_file(f) for f in files if f.is_file()}

class Engine:
    def __init__(self,path,out):
        self.out=Path(out); self.out.mkdir(parents=True,exist_ok=True)
        metadata=Path(path)/'.cache/huggingface/download/config.json.metadata'
        assert metadata.read_text().splitlines()[0]=='906bfd4b4dc7f14ee4320094d8b41684abff8539'
        self.model,self.tokenizer,self.device,fp=load_qwen(path,'cuda:0')
        self.fp={**fp,'files':fingerprint(path),'torch':torch.__version__,'hostname':platform.node(),
                 'transformers':__import__('transformers').__version__,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
                 'gpu':torch.cuda.get_device_name(0),'revision_declared':'906bfd4b4dc7f14ee4320094d8b41684abff8539',
                 'git':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
        self.modelhash=digest(self.fp['files']); write_json(self.out/'model_fingerprint.json',self.fp)
        self.calls=0; self.cache={}; self.hiddens={}; self.token_ids={}
        for k in 'ABCDEFGH':
            ids=self.tokenizer.encode(k,add_special_tokens=False)
            assert len(ids)==1,(k,ids)
            self.token_ids[k]=ids[0]
        self.journal=self.out/'forward_journal.jsonl'
        if self.journal.exists():
            for line in self.journal.read_text().splitlines():
                rec=json.loads(line); assert rec['checksum']==digest(rec['payload']); self.cache[rec['key']]=rec['payload']
        torch.cuda.reset_peak_memory_stats(); self.start=time.time()
    def tokens(self,prompt):
        enc=self.tokenizer(prompt,return_offsets_mapping=True,return_tensors='pt')
        offsets=enc.pop('offset_mapping')[0].tolist()
        return {k:v.to(self.device) for k,v in enc.items()},offsets
    def positions(self,row,prompt=None):
        text=prompt or row['prompt']; _,offsets=self.tokens(text)
        spans=row.get('spans',[])
        return [[i for i,(a,b) in enumerate(offsets) if b>a and b>s[0] and a<s[1]] for s in spans]
    def capture(self,prompt,positions=None):
        key=digest([prompt,positions]); cachefile=self.out/('activation_'+key+'.pt')
        if cachefile.exists(): return torch.load(cachefile,weights_only=True)
        enc,_=self.tokens(prompt); captured={}
        def hook(m,args,out):
            h=out[0] if isinstance(out,tuple) else out
            captured['h']=h[0,positions or [-1]].float().mean(0).detach().cpu()
        handle=_layers(self.model)[LAYER].register_forward_hook(hook)
        try:
            with torch.inference_mode(): self.model(**enc,use_cache=False)
            self.calls+=1
        finally: handle.remove()
        torch.save(captured['h'],cachefile)
        self.log('capture',key,{'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'positions':positions,'activation_file':cachefile.name,'forward_count':1})
        return captured['h']
    def log(self,kind,key,payload):
        rec={'kind':kind,'key':key,'payload':payload,'checksum':digest(payload)}
        with self.journal.open('a') as f:
            f.write(json.dumps(rec,ensure_ascii=False)+'\n'); f.flush(); os.fsync(f.fileno())
    @contextmanager
    def edit(self,enc,edits,biases):
        handles=[]; stats={'residual':[],'attention':[]}
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        original=ALL_ATTENTION_FUNCTIONS['sdpa']
        if edits:
            def hook(module,args,out):
                h=out[0] if isinstance(out,tuple) else out; modified=h.clone()
                for e in edits:
                    positions=e['positions'] or [h.shape[1]-1]
                    untouched=modified.clone()
                    x=modified[0,positions,:].float().mean(0)
                    if e['kind']=='mean': delta=e['delta'].to(x)*e['beta']
                    else:
                        u=e['basis'].to(x); target=e['target'].to(x)
                        delta=(u@(u.T@(target-x)))*e['beta']
                    if e.get('random'):
                        gen=torch.Generator(device='cpu').manual_seed(SEED)
                        v=torch.randn(delta.shape,generator=gen).to(delta); delta=v/v.norm()*delta.norm()
                    if e.get('reverse'): delta=-delta
                    before=modified[:,positions,:].clone()
                    modified[:,positions,:]=(modified[:,positions,:].float()+delta).to(h.dtype)
                    actual=(modified[:,positions,:]-before).float()
                    stats['residual'].append({'positions':positions,'requested_norm':float(delta.norm()),'actual_norm':float(actual.norm()/len(positions)**.5),
                      'relative_norm':float(actual.norm()/before.float().norm().clamp_min(1e-9)),
                      'quantization_error':float((actual-delta).norm()),'non_target_changed':int(torch.count_nonzero((modified-untouched)[:,[j for j in range(h.shape[1]) if j not in positions],:]))})
                if isinstance(out,tuple): return (modified,*out[1:])
                return modified
            handles.append(_layers(self.model)[LAYER].register_forward_hook(hook))
        if biases:
            def biased(module,q,k,v,mask,*args,**kwargs):
                if module.layer_idx!=LAYER: return original(module,q,k,v,mask,*args,**kwargs)
                nq,nk=q.shape[-2],k.shape[-2]
                if mask is None:
                    mask=torch.zeros((1,1,nq,nk),dtype=q.dtype,device=q.device)
                    mask.masked_fill_(torch.ones((nq,nk),dtype=torch.bool,device=q.device).triu(1),float('-inf'))
                else: mask=mask.clone()
                for positions,amount in biases:
                    if not positions: continue
                    previous=mask[:,:,nq-1,positions].clone()
                    mask[:,:,nq-1,positions]+=amount
                    actual=mask[:,:,nq-1,positions]-previous
                    stats['attention'].append({'layer':LAYER,'heads':list(range(q.shape[1])),'query_position':nq-1,'key_positions':positions,
                        'requested_score_bias':amount,'actual_score_rms':float(actual.float().square().mean().sqrt()),
                        'actual_signed_mean':float(actual.float().mean()),'post_rope':True,'non_target_score_changed':0,
                        'query_heads':q.shape[1],'kv_heads':k.shape[1]})
                kwargs.pop('is_causal',None)
                return original(module,q,k,v,mask,*args,is_causal=False,**kwargs)
            ALL_ATTENTION_FUNCTIONS.register('sdpa',biased)
        try: yield stats
        finally:
            for h in handles:h.remove()
            if biases: ALL_ATTENTION_FUNCTIONS.register('sdpa',original)
    def score(self,prompt,options,edits=None,biases=None,cache=True):
        def serial(e): return {k:(digest(v.detach().cpu().tolist()) if torch.is_tensor(v) else v) for k,v in e.items()}
        edits=[e for e in (edits or []) if e['beta']!=0]; biases=[(p,b) for p,b in (biases or []) if b!=0]
        spec={'edits':[serial(e) for e in edits],'biases':biases}
        key=digest([self.modelhash,prompt,options,spec])
        if cache and key in self.cache: return {**self.cache[key],'cache_hit':True}
        enc,_=self.tokens(prompt); started=time.time()
        with self.edit(enc,edits or [],biases or []) as stats:
            with torch.inference_mode(): logits=self.model(**enc,use_cache=False).logits[0,-1].float()
        self.calls+=1
        scores={k:float(logits[self.token_ids[k]]) for k in options}
        top=int(logits.argmax()); pred=max(scores,key=scores.get)
        payload={'prediction':pred,'candidate_logits':scores,'top_unconstrained_token':self.tokenizer.decode([top]),
            'unconstrained_is_candidate':top in [self.token_ids[k] for k in options],
            'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'output_sha256':digest(scores),
            'prompt_tokens':enc['input_ids'].shape[-1],'spec':spec,'hook':stats,'elapsed_seconds':time.time()-started,
            'forward_count':1,'cache_hit':False,'model_hash':self.modelhash,'failure_code':None}
        self.log('score',key,payload); self.cache[key]=payload
        return payload
    def generate(self,prompt,max_new_tokens=12):
        key=digest([self.modelhash,prompt,'generate',max_new_tokens])
        if key in self.cache:return {**self.cache[key],'cache_hit':True}
        enc,_=self.tokens(prompt); counter=[0]
        def count(m,a,o):counter[0]+=1
        h=self.model.register_forward_hook(count)
        try:
            with torch.inference_mode(): ids=self.model.generate(**enc,max_new_tokens=max_new_tokens,do_sample=False,use_cache=False,pad_token_id=self.tokenizer.eos_token_id)
        finally:h.remove()
        text=self.tokenizer.decode(ids[0,enc['input_ids'].shape[1]:],skip_special_tokens=True)
        self.calls+=counter[0]
        rec={'text':text,'forward_count':counter[0],'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'model_hash':self.modelhash}
        self.log('generate',key,rec); self.cache[key]=rec
        return rec

def fit_axis(pos,neg):
    p=torch.stack(pos).float(); n=torch.stack(neg).float(); diff=p-n if len(p)==len(n) else None
    if diff is None:
        m=min(len(p),len(n)); diff=p[:m]-n[:m]
    centered=diff-diff.mean(0)
    _,s,v=torch.linalg.svd(centered,full_matrices=False)
    rank=min(2,int((s>1e-6).sum()))
    if not rank:
        mean=(p.mean(0)-n.mean(0)); u=(mean/mean.norm().clamp_min(1e-9))[:,None]
    else:u=v[:rank].T
    return {'positive':p.mean(0),'negative':n.mean(0),'mean':p.mean(0)-n.mean(0),'basis':u,'differences':diff,'singular_values':s}
