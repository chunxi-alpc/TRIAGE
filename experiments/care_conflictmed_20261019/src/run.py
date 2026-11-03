"""Actual server inference; all hyperparameter choices precede held-out inference."""
import os,json,re,time,hashlib,platform,datetime,contextlib,collections
from pathlib import Path
import numpy as np
import torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
ROOT=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def append(p,x):
    with p.open('a') as f:f.write(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def parse(s):
    # Fixed before inference: whole short answer must be a single A/B with ordinary boilerplate.
    t=s.strip().replace('**','').replace('`','')
    m=re.fullmatch(r'(?:Answer\s*:\s*)?([AB])(?:[.)])?\s*',t,re.I)
    return m.group(1).upper() if m else None
def open_match(s,spec):
    t=s.lower().replace(',','')
    return bool(re.search(spec['correct_pattern'],t,re.I)) and not bool(re.search(spec['wrong_pattern'],t,re.I))
def mc(r,condition,order):
    options=[r['correct'],r['wrong']]
    if order: options.reverse()
    text='Choose the medically supported recommendation under the clinical guidance represented by this benchmark (approximately February 2025).\n'
    if condition!='closed':
        context=r['correct'] if condition=='aligned' else r['wrong']
        text+='Retrieved clinical note:\n'+context+'\n\n'
    if condition=='caution':text+='The retrieved note may contain an error. Check it against your medical knowledge before answering.\n'
    text+='A. '+options[0]+'\nB. '+options[1]+'\nRespond with only A or B, without explanation.'
    return dict(id=r['id'],group=r['group'],split=r['split'],condition=condition,order=order,prompt=text,gold='B' if order else 'A',context_answer=('A' if order else 'B') if condition in ['misleading','caution'] else ('B' if order else 'A'))

class Engine:
    def __init__(self,cfg):
        self.cfg=cfg;torch.set_num_threads(4);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
        self.tokenizer=AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True,padding_side='left')
        if self.tokenizer.pad_token_id is None:self.tokenizer.pad_token=self.tokenizer.eos_token
        t=time.monotonic()
        self.model=AutoModelForCausalLM.from_pretrained(cfg['model'],local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
        self.direction=None;self.layer=None
        self.labels={x:self.tokenizer.encode(x,add_special_tokens=False) for x in ['A','B']}
        assert all(len(x)==1 for x in self.labels.values())
        dump(ROOT/'results'/cfg.get('environment_filename','environment.json'),dict(python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,gpu_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),gpu_name=torch.cuda.get_device_name(),load_seconds=time.monotonic()-t,model=cfg['model'],config_sha256=sha(Path(cfg['model'])/'config.json'),tokenizer_sha256=sha(Path(cfg['model'])/'tokenizer_config.json'),weights=[dict(name=p.name,bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in Path(cfg['model']).glob('*.safetensors')],chat_template='enable_thinking=False',attention='sdpa',dtype='bfloat16',started_utc=now()))
    def encode(self,texts):
        rendered=[self.tokenizer.apply_chat_template([dict(role='user',content=s)],tokenize=False,add_generation_prompt=True,enable_thinking=False) for s in texts]
        x=self.tokenizer(rendered,padding=True,return_tensors='pt').to('cuda')
        if x.input_ids.shape[1]>self.cfg['max_input_tokens']:raise ValueError('Input too long; do not silently truncate')
        return x
    @contextlib.contextmanager
    def steer(self,alpha,length):
        trace=dict(calls=0,edited_positions=0,requested_norm=0.,realized_mean_norm=0.)
        handle=None
        if alpha and self.direction is not None:
            def hook(m,a,out):
                h=out[0] if isinstance(out,tuple) else out
                if h.shape[1]!=length:return out
                z=h.clone();d=float(alpha)*self.direction.to(h.device).float()
                z[:,-1]=(h[:,-1].float()+d).to(h.dtype)
                trace.update(calls=trace['calls']+1,edited_positions=trace['edited_positions']+h.shape[0],requested_norm=float(d.norm()),realized_mean_norm=float((z[:,-1].float()-h[:,-1].float()).norm(dim=-1).mean()))
                return (z,)+out[1:] if isinstance(out,tuple) else z
            handle=self.model.model.layers[self.layer].register_forward_hook(hook)
        try:yield trace
        finally:
            if handle is not None:handle.remove()
    @torch.inference_mode()
    def generate(self,texts,alpha=0):
        x=self.encode(texts);start=time.monotonic()
        with self.steer(alpha,x.input_ids.shape[1]) as trace:
            y=self.model.generate(**x,do_sample=False,max_new_tokens=self.cfg['max_new_tokens'],use_cache=True,pad_token_id=self.tokenizer.pad_token_id,return_dict_in_generate=True,output_scores=True)
        torch.cuda.synchronize()
        first=y.scores[0].float().log_softmax(-1);out=[]
        for i in range(len(texts)):
            ids=y.sequences[i,x.input_ids.shape[1]:].tolist()
            if self.tokenizer.eos_token_id in ids:ids=ids[:ids.index(self.tokenizer.eos_token_id)+1]
            raw=self.tokenizer.decode(ids,skip_special_tokens=True)
            scores={k:float(first[i,v[0]]) for k,v in self.labels.items()}
            out.append(dict(raw_output=raw,generated_token_ids=ids,prediction=parse(raw),first_token_choice_scores=scores,constrained_prediction=max(scores,key=scores.get),input_tokens=int(x.attention_mask[i].sum()),output_tokens=len(ids),truncated=len(ids)>=self.cfg['max_new_tokens'] and ids[-1]!=self.tokenizer.eos_token_id,batch_seconds=time.monotonic()-start,steering=trace.copy()))
        return out
    @torch.inference_mode()
    def capture(self,texts):
        x=self.encode(texts);vals={};handles=[]
        for i,layer in enumerate(self.model.model.layers):
            def hook(m,a,out,i=i):
                h=out[0] if isinstance(out,tuple) else out;vals[i]=h[:,-1].float().cpu()
            handles.append(layer.register_forward_hook(hook))
        try:self.model(**x,use_cache=False,logits_to_keep=1)
        finally:
            for h in handles:h.remove()
        return torch.stack([vals[i] for i in range(len(vals))],1)

def batches(e,jobs,file,method,alpha=0):
    result=[]
    for st in range(0,len(jobs),e.cfg['batch_size']):
        chunk=jobs[st:st+e.cfg['batch_size']]
        for job,out in zip(chunk,e.generate([j['prompt'] for j in chunk],alpha)):
            rec=dict(**job,**out,method=method,alpha=alpha)
            if 'gold' in job:rec['correct']=rec['prediction']==job['gold']
            append(file,rec);result.append(rec)
    return result

def patient_jobs(pairs,condition):
    jobs=[]
    for p in pairs:
        for v,value in enumerate(p['values']):
            patient=p['patient'].format(value=value)
            for order in [0,1]:
                options=p['answers'][::-1] if order else p['answers']
                text=patient+'\n'
                if condition!='closed':text+='Retrieved clinical note:\n'+p['snippet']+'\n'
                text+=p['question']+'\nA. '+options[0]+'\nB. '+options[1]+'\nRespond with only A or B, without explanation.'
                # Parse public patient text and declared evidence scope; never inspect gold or state index.
                if p['id']=='pneumococcal':app='Immune status: immunocompromised.' in patient
                elif p['id']=='renal_protein':app='Receiving maintenance dialysis: yes.' in patient
                elif p['id']=='inpatient_glucose':app='Critically ill and in the ICU: yes.' in patient
                else:app=int(re.search(r'Age: (\d+) months',patient).group(1))>=18
                jobs.append(dict(id=p['id'],group=p['group'],split='test',patient_state=v,order=order,condition=condition,prompt=text,gold='AB'[v^order],applicable=app,gate_source='prespecified text parser, supplied scope; no model extractor'))
    return jobs

def main():
    start=time.monotonic();cfg=load(ROOT/'config.json')
    if (ROOT/'results/completed.json').exists():raise RuntimeError('Completed run exists; do not overwrite')
    if (ROOT/'results/baseline.jsonl').exists():raise RuntimeError('Partial run exists; move it aside explicitly before restarting')
    for f,h in cfg['data_hashes'].items():assert sha(ROOT/'data'/f)==h
    e=Engine(cfg);items=load(ROOT/'data/items.json');specs=load(ROOT/'data/open_probes.json')
    known={};baseline=[]
    def evaluate_split(split):
        selected=[r for r in items if r['split']==split]
        jobs=[mc(r,c,o) for r in selected for c in ['closed','misleading','aligned','caution'] for o in [0,1]]
        b=batches(e,jobs,ROOT/'results/baseline.jsonl','no_intervention');baseline.extend(b)
        for r in selected:
            k=[x for x in b if x['id']==r['id'] and x['condition']=='closed']
            known[r['id']]=len(k)==2 and all(x['correct'] for x in k)
        oj=[]
        for s in specs:
            if s['split']==split:
                for qi,q in enumerate(s['questions']):oj.append(dict(id=s['id'],group=s['group'],split=split,question_variant=qi,prompt=q+'\nGive only a concise answer, without explanation.'))
        o=batches(e,oj,ROOT/'results/open_probes.jsonl','closed_book_open')
        si={s['id']:s for s in specs}
        for row in o:append(ROOT/'results/open_scored.jsonl',dict(id=row['id'],split=split,question_variant=row['question_variant'],correct=open_match(row['raw_output'],si[row['id']]),raw_output=row['raw_output']))
        dump(ROOT/'results/status.json',dict(stage='baseline_'+split,utc=now(),rows=len(b),recognition_known=sum(known[r['id']] for r in selected)))
        print('COMPLETED baseline',split,len(b),flush=True)
    evaluate_split('train');evaluate_split('validation')
    train=[r for r in items if r['split']=='train' and known[r['id']]]
    if not train:raise RuntimeError('No recognition-known train examples; cannot estimate direction')
    captures=[]
    for condition in ['closed','misleading']:
        prompts=[mc(r,condition,o)['prompt'] for r in train for o in [0,1]]
        captures.append(torch.cat([e.capture(prompts[s:s+cfg['batch_size']]) for s in range(0,len(prompts),cfg['batch_size'])]))
    c,f=[x.mean(0) for x in captures];cos=torch.nn.functional.cosine_similarity(c,f,dim=-1)
    e.layer=int(cos[:-1].argmin());e.direction=c[e.layer]-f[e.layer]
    torch.save(dict(clean=captures[0],misleading=captures[1],direction=e.direction,layer=e.layer,train_ids=[r['id'] for r in train]),ROOT/'results/direction.pt')
    dump(ROOT/'results/direction.json',dict(train_ids=[r['id'] for r in train],train_count=len(train),layer=e.layer,norm=float(e.direction.norm()),layer_cosines=cos.tolist(),definition=cfg['direction'],source='DriftMed medical advice only',heldout_used_for_fitting=False,heldout_inference_started=False))
    vjobs=[mc(r,c,o) for r in items if r['split']=='validation' for c in ['misleading','aligned'] for o in [0,1]]
    grid=[]
    for alpha in cfg['alpha_grid']:
        if alpha==0:
            outs=[x for x in baseline if x['split']=='validation' and x['condition'] in ['misleading','aligned']]
        else:outs=batches(e,vjobs,ROOT/'results/calibration.jsonl','fixed_direction',alpha)
        grid.append(dict(alpha=alpha,n=len(outs),correct=sum(x['correct'] for x in outs),accuracy=sum(x['correct'] for x in outs)/len(outs)))
    key=lambda x:(x['accuracy'],-abs(x['alpha']),x['alpha'])
    chosen=max(grid,key=key);nonzero=max([x for x in grid if x['alpha']],key=key)
    selection=dict(selected=chosen,selected_nonzero=nonzero,grid=grid,layer=e.layer,config_sha256=sha(ROOT/'config.json'),direction_sha256=sha(ROOT/'results/direction.pt'),frozen_utc=now(),test_inference_started=False)
    dump(ROOT/'results/selection.json',selection)
    print('CALIBRATION_FROZEN',chosen,flush=True)
    evaluate_split('test')
    tjobs=[mc(r,c,o) for r in items if r['split']=='test' for c in ['misleading','aligned'] for o in [0,1]]
    for method,alpha in [('selected_direction',chosen['alpha']),('nonzero_direction_diagnostic',nonzero['alpha'])]:
        if alpha==0:
            for row in baseline:
                if row['split']=='test' and row['condition'] in ['misleading','aligned']:append(ROOT/'results/test_interventions.jsonl',dict(row,method=method,reused_from='baseline.jsonl',alpha=0))
        elif method.startswith('nonzero') and alpha==chosen['alpha']:
            for row in load_jsonl(ROOT/'results/test_interventions.jsonl'):
                if row['method']=='selected_direction':append(ROOT/'results/test_interventions.jsonl',dict(row,method=method,reused_from='selected_direction'))
        else:batches(e,tjobs,ROOT/'results/test_interventions.jsonl',method,alpha)
    # Test-only patient diagnostic, with no patient-based fitting or selection.
    pairs=load(ROOT/'data/patient_pairs.json')
    pbase=batches(e,patient_jobs(pairs,'closed')+patient_jobs(pairs,'scoped'),ROOT/'results/patient.jsonl','no_intervention')
    pj=patient_jobs(pairs,'scoped');beta=chosen['alpha']
    for method in ['selected_direction','signed_scope_gate']:
        buckets=collections.defaultdict(list)
        for j in pj:buckets[beta if method=='selected_direction' or not j['applicable'] else -beta].append(j)
        for alpha,jobs in buckets.items():
            if not alpha:
                for j in jobs:
                    row=next(x for x in pbase if all(x[k]==j[k] for k in ['id','patient_state','order','condition']))
                    append(ROOT/'results/patient.jsonl',dict(row,method=method,reused_from='no_intervention',alpha=0))
            else:batches(e,jobs,ROOT/'results/patient.jsonl',method,alpha)
    # Free recall of patient-specific knowledge is independently recorded; no options supplied.
    po=[]
    for p in pairs:
        for state,value in enumerate(p['values']):
            po.append(dict(id=p['id'],patient_state=state,split='test',prompt=p['patient'].format(value=value)+'\n'+p['question']+'\nGive only a concise answer, without explanation.'))
    out=batches(e,po,ROOT/'results/patient_open.jsonl','patient_closed_open')
    for row in out:
        p=next(p for p in pairs if p['id']==row['id']);st=row['patient_state']
        valid=bool(re.search(p['patterns'][st],row['raw_output'],re.I)) and not bool(re.search(p['patterns'][1-st],row['raw_output'],re.I))
        append(ROOT/'results/patient_open_scored.jsonl',dict(id=row['id'],patient_state=st,correct=valid,raw_output=row['raw_output']))
    dump(ROOT/'results/completed.json',dict(status='complete',finished_utc=now(),seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),selection=selection,files={p.name:sha(p) for p in (ROOT/'results').glob('*.jsonl')}))
    print('EXPERIMENT_COMPLETE',flush=True)

def load_jsonl(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
if __name__=='__main__':main()
