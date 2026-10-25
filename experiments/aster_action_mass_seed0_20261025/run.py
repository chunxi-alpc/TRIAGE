"""Fresh seed=0 dev-only GPU inference. Every candidate retained."""
import argparse,hashlib,json,os,time,traceback,sys
from pathlib import Path
import torch
from action_engine import ActionEngine
from mass_operator import action_scores
from improve_branch import judge,judge_payload,condition_key
HERE=Path(__file__).resolve().parent
CORE=('NC','AC','IC','SW')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,obj):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(p)
def configs():
 out=[('original',{}),('mass_zero_control',{'mode':'mass','lam':0.,'layers':'all'}),('action_crop',{'form':'action_crop'})]
 for layers in ('all','last_half'):
  for mode in ('gate','hard'):
   out.append((f'branch_{mode}_{layers}',dict(mode=mode,lam=2.,layers=layers,gate_phase='all')))
  for strength in (1.,2.):out.append((f'action_bias_{strength}_{layers}',dict(mode='action_bias',lam=strength,layers=layers)))
  for strength in (.25,.5,1.):out.append((f'mass_{strength}_{layers}',dict(mode='mass',lam=strength,layers=layers)))
 return out

def gpu_tests(e,c,out):
 from mass_operator import attention
 torch.manual_seed(0);q=torch.randn(1,4,257,16,device='cuda',dtype=torch.bfloat16)
 k=torch.randn(1,2,257,16,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
 m=torch.zeros(1,1,257,257,device='cuda').masked_fill(torch.ones(257,257,device='cuda',dtype=torch.bool).triu(1),float('-inf'))
 ref=torch.nn.functional.scaled_dot_product_attention(q,k.repeat_interleave(2,1),v.repeat_interleave(2,1),attn_mask=m.to(q.dtype))
 got=attention(q,k,v,m,.25,list(range(257)),[1/257]*257)
 err=float((ref-got).abs().max());assert err<.04,err
 # Long causal context with an explicit masked key, using query chunks.
 ql=torch.randn(1,4,4097,16,device='cuda',dtype=torch.bfloat16)
 kl=torch.randn(1,2,4097,16,device='cuda',dtype=torch.bfloat16);vl=torch.randn_like(kl)
 ml=torch.ones(1,1,4097,4097,device='cuda',dtype=torch.bool).tril();ml[...,7]=False
 long_out=attention(ql,kl,vl,ml,.25,[i%4-1 for i in range(4097)],[.2,.3,.5],alpha=.5)
 assert torch.isfinite(long_out).all()
 del ql,kl,vl,ml,long_out

 ids,roles=e.render(c);act=torch.tensor([bool(r and r['kind'] in ('action','option')) for r in roles])
 changed=list(ids)
 token=e.tok.encode(' neutral',add_special_tokens=False)[-1]
 for i,a in enumerate(act):
  if a:changed[i]=token
 def capture(tokens,mode):
  cap={};e.forward(tokens,roles,len(ids),mode,capture=cap)
  return {l:x[...,~act,:] for l,x in cap.items()}
 a=capture(ids,'blind');b=capture(changed,'blind')
 errors={str(l):float((a[l]-b[l]).abs().max()) for l in a};assert max(errors.values())<1e-5,errors
 del a,b
 a=capture(ids,'none');b=capture(changed,'none');positive=max(float((a[l]-b[l]).abs().max()) for l in a)
 assert positive>1e-5
 atomic(out/'gpu_tests.json',{'bf16_gqa_max_abs_error':err,'blind_nonaction_K_error_each_layer':errors,'unmasked_positive_control_max_error':positive,'tokens':len(ids),'synthetic_long_context_tokens':4097,'passed':True})

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--name',choices=['qwen','llama'],required=True)
 ap.add_argument('--root',type=Path,default=Path('/home/gcx/aster_action_mass_seed0_20261025'))
 ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
 assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('0','1')
 out=a.root/(a.name+('_pilot' if a.limit else ''));out.mkdir(parents=True,exist_ok=False)
 start=time.time();data=a.root/'dev.jsonl';rows=[json.loads(x) for x in data.read_text().splitlines()]
 assert all([r['rid'] for r in c['rules']]==list(range(len(c['rules']))) for c in rows)
 assert len(rows)==800 and len({c['id'] for c in rows})==200
 assert all([r['rid'] for r in c['rules']]==list(range(len(c['rules']))) for c in rows)
 assert all(c['split']=='dev' and c['seed']==0 and c['variant'] in CORE for c in rows)
 rows.sort(key=lambda c:(c['id'],CORE.index(c['variant'])))
 if a.limit:rows=rows[:a.limit]
 model=Path('/home/gcx/TRIAGE/models/modelscope')/('Qwen3-8B' if a.name=='qwen' else 'Meta-Llama-3.1-8B-Instruct')
 manifest={'name':a.name,'model':str(model),'model_config_sha256':sha(model/'config.json'),'data_sha256':sha(data),'seed':0,'rows':len(rows),'cases':len({c['id'] for c in rows}),'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'configs':configs(),'code_sha256':{p.name:sha(p) for p in HERE.glob('*.py')},'legacy_code_sha256':{f:sha(HERE.parent/'aster_e0_e8_20261024'/f) for f in ('engine.py','improve_branch.py')},'baseline_pr31':'cf3298d','baseline_pr30':'1dae58a14a8242c620e8c5810be432ae198949cf','estimate_GiB_per_gpu':[20,35],'estimate_wall_hours':[2,6],'provenance':'fresh GPU; exact-payload condition reuse within this run only','torch':torch.__version__}
 atomic(out/'manifest.json',manifest)
 e=ActionEngine(str(model));gpu_tests(e,{k:v for k,v in rows[0].items() if not k.startswith('gold')},out)
 cache={};calls=0;max_tokens=0
 with (out/'predictions.jsonl').open('x') as f,(out/'condition_cache.jsonl').open('x') as cf:
  for i,c in enumerate(rows):
   t=time.time();safe={k:v for k,v in c.items() if k in ('patient','rules','options')}
   ck=condition_key(safe);reused=ck in cache;condition_seconds=0.
   if not reused:
    judge_start=time.time();p,assess=judge(e,judge_payload(safe));cache[ck]=(p,assess);calls+=len(p);condition_seconds=time.time()-judge_start
    cf.write(json.dumps({'key':ck,'p':p,'assessments':assess,'seconds':time.time()-judge_start})+'\n');cf.flush()
   p,assess=cache[ck];actions,s=action_scores(safe,p)
   winner=actions[max(range(len(s)),key=lambda j:s[j])]
   option_s=[s[actions.index(opt)] if opt in actions else 0. for opt in safe['options']]
   represented=sum(option_s);answers={};times={};scores={}
   for name,kw in configs():
    kw=dict(kw)
    if name=='action_crop':kw['selected']=winner
    if name!='original':kw['p']=p
    st=time.time();pred,logits=e.answer(safe,**kw);torch.cuda.synchronize();times[name]=time.time()-st;calls+=1
    answers[name]=pred;scores[name]=logits
   base=torch.tensor(scores['original']).softmax(0).tolist()
   normalized=[x/represented for x in option_s] if represented>0 else base
   answers['direct_map']=max(range(len(normalized)),key=lambda j:normalized[j])
   scores['direct_map']=normalized
   for w in (.25,.5,.75):
    name=f'fusion_{w}';mix=[(1-w)*b+w*x for b,x in zip(base,normalized)]
    scores[name]=mix;answers[name]=max(range(len(mix)),key=lambda j:mix[j])
   for layer in ('all','last_half'):
    name=f'mass_0.5_{layer}_conf0.6';src=f'mass_0.5_{layer}' if max(s)>=.6 else 'original'
    answers[name]=answers[src];scores[name]=scores[src]
   tokens=len(e.render(safe)[0]);max_tokens=max(max_tokens,tokens)
   r={k:c[k] for k in ('id','tree','seed','variant','split','gold_option')}
   r.update(p=p,S=s,option_S=option_s,condition_seconds=condition_seconds,rule_to_action=[actions.index(rule['action']) for rule in safe['rules']],option_to_action=[actions.index(opt) if opt in actions else -1 for opt in safe['options']],condition_key=ck,condition_reused=reused,answers=answers,scores=scores,seconds=times,total_seconds=time.time()-t,tokens=tokens,represented_mass=represented,orphan_options=sum(x==0 for x in option_s),rules=len(p),actions=len(s),covered=represented>0)
   f.write(json.dumps(r)+'\n');f.flush()
   atomic(out/'status.json',{'completed':i+1,'total':len(rows),'complete':i+1==len(rows),'elapsed_seconds':time.time()-start,'peak_GiB':torch.cuda.max_memory_allocated()/2**30,'max_tokens':max_tokens,'forward_call_units':calls,'gpu_hours':(time.time()-start)/3600})
 atomic(out/'DONE.json',{'complete':True,'elapsed_seconds':time.time()-start})
if __name__=='__main__':
 try:main()
 except BaseException:traceback.print_exc();raise
