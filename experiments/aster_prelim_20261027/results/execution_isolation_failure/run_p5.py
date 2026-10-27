"""P5 gold-information execution controls, conditional on all P4 tests passing."""
import argparse,collections,hashlib,json,os,time
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from attention_engine import Engine
from run_p1 import dump,sha
BASE=Path(__file__).resolve().parent
PROJECTIONS={'qwen':'/home/gcx/TRIAGE-worktrees/aster-seka-execution-20261025/experiments/aster_seka_execution/runtime/projection_full200/Qwen3-8B_pos_proj.pt','llama':'/home/gcx/aster_e0_e8_20261024/private/green_llama/Meta-Llama-3.1-8B-Instruct_pos_proj.pt'}

class P5Engine(Engine):
 def __init__(self,model,tok,projection,green_layers):
  self.green=0.;self.green_layers=set(green_layers);self.highlight=[];self.projection_alpha=0.;self.collect=False;self.masses=[]
  super().__init__(model,tok)
  artifact=torch.load(projection,map_location='cpu',weights_only=True)
  for i,P in zip(artifact['layers'],artifact['proj']):
   attn=model.model.layers[i].self_attn;module=attn.k_norm if hasattr(attn,'k_norm') else attn.k_proj;P=P.to('cuda',dtype=torch.float32)
   def hook(mod,args,result,P=P):
    if not self.projection_alpha:return
    b,t=result.shape[:2];view=result.view(b,t,P.shape[0],P.shape[1]);mask=self.state['roles']&(self.state['branches']>=0)&(self.state['branches']!=self.state['selected'])
    result=view.clone();keys=view[0,mask].float();delta=torch.einsum('nhd,hdk->nhk',keys,P);result[0,mask]=(keys-self.projection_alpha*delta).to(view.dtype);return result
   # Preserve the module output shape (Qwen k_norm 4D, Llama k_proj 3D).
   def wrapped(mod,args,result,hook=hook):
    shape=result.shape;z=hook(mod,args,result);return z.reshape(shape) if z is not None else None
   self.handles.append(module.register_forward_hook(wrapped))
 def mask(self,module,args,kw):
  answer=super().mask(module,args,kw)
  if answer is not None:args,kw=answer
  if self.green and module.layer_idx in self.green_layers:
   mask=kw['attention_mask'].clone();mask[0,0,-1,self.highlight]+=self.green;kw['attention_mask']=mask;return args,kw
  return answer
 def set_highlight(self,packed,row,selected):
  rule=next(q for q in row['rules'] if int(q['bid'])-1==selected)
  prefix=self.tok.encode(f"Branch {rule['bid']}: IF {rule['condition']}\nAction: ",add_special_tokens=False)
  ids=packed['ids'];starts=[i for i in range(len(ids)-len(prefix)+1) if ids[i:i+len(prefix)]==prefix];assert len(starts)==1
  start=starts[0];self.highlight=list(range(start,start+len(prefix)+len(self.tok.encode(rule['action'],add_special_tokens=False))))

def main():
 p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--model',required=True);a=p.parse_args()
 assert json.loads((BASE/'results/p4/gate_status.json').read_text())['P4']=='PASS','P4 has not passed; P5 prohibited'
 out=BASE/'private'/('p5_'+a.name);out.mkdir(exist_ok=False);start=time.time();torch.set_num_threads(4);torch.manual_seed(20261027);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True);model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,torch_dtype=torch.float32,attn_implementation='sdpa',device_map={'':'cuda:0'}).eval()
 choice=json.loads((BASE/'results/p3/gate_status.json').read_text())['window_choices'][a.name]['selected_layers'];nl=len(model.model.layers)
 green_layers=list(range(nl)) if choice=='all' else list(range(int(choice.split('_')[1])-1,int(choice.split('_')[2])))
 eng=P5Engine(model,tok,PROJECTIONS[a.name],green_layers)
 rows=[json.loads(s) for s in (BASE/'private/cases.jsonl').read_text().splitlines()]
 # All 200 SW/AC cases plus deterministic 50 NC/IC cases and available CS.
 chosen={}
 for v in ['NC','IC']:
  chosen[v]={r['id'] for r in sorted([r for r in rows if r['variant']==v],key=lambda r:hashlib.sha256(('p5:'+r['id']).encode()).hexdigest())[:50]}
 rows=[r for r in rows if r['variant'] not in chosen or r['id'] in chosen[r['variant']]]
 sw=[r for r in rows if r['variant']=='SW'];mass_ids=set()
 for tree in sorted({r['tree'] for r in sw}):mass_ids.update(r['id'] for r in sorted([r for r in sw if r['tree']==tree],key=lambda r:r['id'])[:5])
 configs=[('gold_prompt',{}),('gold_rule',{}),('gold_prune',{}),('gold_prune_isolated',{})]+[(f'gate_{lam:g}',{'lam':lam}) for lam in [0,.5,1,2,4,8,16]]+[('gate_hard',{'hard':True}),('gate_8_green',{'lam':8,'green':1})]+[(f'projection_{g:g}',{'projection_alpha':g}) for g in [.5,1]]
 manifest={'stage':'P5','dtype':'float32','TF32':False,'name':a.name,'status':'running','gpu':os.environ['CUDA_VISIBLE_DEVICES'],'counts':dict(collections.Counter(r['variant'] for r in rows)),'configs':[k for k,_ in configs],'script_sha256':sha(__file__),'engine_sha256':sha(BASE/'attention_engine.py'),'protocol_sha256':sha(BASE/'p5_protocol.json'),'projection_path':PROJECTIONS[a.name],'projection_sha256':sha(PROJECTIONS[a.name]),'projection_source':'fixed pre-existing official SEKA synthetic QA calibration artifact; no new calibration data evaluated','projection_operator':'k <- k - gamma P k, gamma in [0,1], using official pre-RoPE k_norm/k_proj placement; not a full published SEKA benchmark reproduction','green':'bounded +1 attention score on gold rule ID/condition/action in P3-selected layers','gate_probability_floor':1e-4,'attention_mass_cases':len(mass_ids),'attention_mass_scope':'20 fixed SW cases, FP32 recomputation from actual query/key and final-query mask, not full-kernel attention output','test_data_used':False}
 dump(out/'manifest.json',manifest)
 original_sdpa=ALL_ATTENTION_FUNCTIONS['sdpa']
 def measured(module,q,k,v,mask,**kw):
  answer=original_sdpa(module,q,k,v,mask,**kw)
  if eng.collect:
   kk=k.repeat_interleave(q.shape[1]//k.shape[1],dim=1);scores=torch.matmul(q[:,:,-1:].float(),kk.float().transpose(-2,-1))*kw.get('scaling',q.shape[-1]**-.5)
   if mask is not None:scores+=mask[:,:,-1:,:].float()
   weights=scores.softmax(-1)[0,:,0];roles=eng.state['roles'];branches=eng.state['branches'];selected=eng.state['selected'];app=roles&(branches==selected);comp=roles&(branches>=0)&(branches!=selected)
   aa=weights[:,app].sum(-1);cc=weights[:,comp].sum(-1);eng.masses.append({'layer':module.layer_idx+1,'applicable_mass':aa.mean().item(),'competitor_mass':cc.mean().item(),'applicable_conditional_share':(aa/(aa+cc).clamp_min(1e-30)).mean().item()})
  return answer
 ALL_ATTENTION_FUNCTIONS.register('sdpa',measured)
 try:
  with (out/'records.jsonl').open('w',buffering=1) as f:
   for index,r in enumerate(rows):
    selected=int(r['gold'])-1;rule=next(q for q in r['rules'] if q['bid']==r['gold']);gold=chr(65+r['options'].index(rule['action']));labels=[chr(65+i) for i in range(len(r['options']))]
    for name,config in configs:
     extra=''
     if name=='gold_prompt':extra=f"\nKnown applicable branch: {r['gold']}. Use the supplied action for this branch."
     if name=='gold_rule':extra=f"\nApplicable rule: IF {rule['condition']} THEN {rule['action']}."
     packed=eng.pack(r,task='action',extra_patient=extra);eng.green=config.get('green',0);eng.projection_alpha=config.get('projection_alpha',0);eng.set_highlight(packed,r,selected);eng.collect=r['id'] in mass_ids and r['variant']=='SW' and name not in ['gold_prompt','gold_rule','gold_prune','gold_prune_isolated'];eng.masses=[]
     torch.cuda.synchronize();t=time.time()
     logits=eng.score(packed,'execute',selected=selected,prune=name in ['gold_prune','gold_prune_isolated'],hard=config.get('hard',False),lam=config.get('lam',0),plain=name in ['gold_prompt','gold_rule','gold_prune'])
     torch.cuda.synchronize();elapsed=time.time()-t;pred,prob=eng.choose(logits,labels)
     z={'model':a.name,'id':r['id'],'tree':r['tree'],'variant':r['variant'],'method':name,'gold_option':gold,'prediction':pred,'correct':pred==gold,'option_probabilities':prob,'seconds':elapsed,'attention_mass':eng.masses};f.write(json.dumps(z)+'\n')
    if (index+1)%20==0:dump(out/'status.json',{'cases_completed':index+1,'cases_total':len(rows),'seconds':time.time()-start,'peak_GiB':torch.cuda.max_memory_allocated()/2**30});print(json.dumps({'completed':index+1,'total':len(rows)}),flush=True)
 finally:ALL_ATTENTION_FUNCTIONS.register('sdpa',original_sdpa)
 manifest.update(status='completed',wall_seconds=time.time()-start,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30);dump(out/'manifest.json',manifest);dump(out/'DONE.json',{'status':'completed'})
if __name__=='__main__':main()
