"""Matched comparison: the identical decision-conditioned signal is applied at the input or attention output."""
import argparse,contextlib,copy,json,random,re,time
from pathlib import Path
import torch
from .data import OUT,SEED,read,write,deployment,nodes
from .engine import Engine,copy_cache
from .intervention import ConditionalIntervention
from .train import train_case,load_module,save_checkpoint
from .matched_data import DEST
from experiments.aster_bottleneck_20261031.runtime import attention_control
from experiments.aster_manuscript_clinical_20261030.native_memory import query_chunked_attention

LAYER=18
@contextlib.contextmanager
def capture(layer,box):
 def save(_module,_args,out):box.append(out[:,-1:,:])
 h=layer.register_forward_hook(save)
 try:yield
 finally:h.remove()

class SoftContext(Engine):
 def training_decision(self,context,history,gid,layer,module,blind,training=True):
  prefix='PATH: '+(', '.join(history)+', ' if history else '')+'R'+gid[1:]
  ids=self.builder.tokenizer.encode(prefix,add_special_tokens=False)
  assert len(ids)>=2
  oproj=self.model.model.layers[layer].self_attn.o_proj
  before=[]
  with torch.no_grad(),attention_control(self.back),capture(oproj,before):
   clean=self.back._forward(torch.tensor([ids]),cache=copy_cache(context['result'].past_key_values),start=context['length'],use_cache=True)
  position=context['length']+len(ids)-2
  state=before[0].detach()
  # Same full readout, same blind tensor, same trainable module. Only the injection site changes.
  with torch.set_grad_enabled(training):
   edited,_,gate=module(state,blind.to(state.device).reshape_as(state))
   delta=edited.float()-state.float()
   def soft_slot(_module,_args,embedding):
    anchor=embedding[:,:1,:]
    scale=anchor.detach().float().norm()/state.float().norm().clamp_min(1e-8)
    return torch.cat(((anchor.float()+scale*delta).to(embedding.dtype),embedding[:,1:,:]),dim=1)
   after=[]
   handle=self.model.get_input_embeddings().register_forward_hook(soft_slot)
   try:
    with attention_control(self.back),capture(oproj,after):
     result=self.back._forward(torch.tensor([ids[-2:]]),cache=copy_cache(clean.past_key_values,position),start=position,use_cache=True)
   finally:handle.remove()
   # Keep the edit penalty in the same space as the internal arm.
   mag=(after[0].float()-state.float()).square().sum()/state.float().square().sum().clamp_min(1e-8)
   return result.logits[0,-1,[self.yes,self.no]].float(),clean.logits[0,-1,[self.yes,self.no]].float().detach(),{'magnitude':mag,'gate':gate}

def build(arm):
 torch.set_num_threads(4);torch.manual_seed(SEED)
 e=(SoftContext if arm=='soft' else Engine)()
 torch.manual_seed(SEED)
 m=ConditionalIntervention(e.width,variant='no_js').to(e.device)
 assert sum(p.numel() for p in m.parameters())==32777
 return e,m

def cases():return read(OUT/'train.jsonl')+read(OUT/'age_train.jsonl')

def pilot():
 e,m=build('soft')
 picked={r['case_id'] for r in json.loads((OUT/'pilot.json').read_text())['cases']}
 results=[]
 with query_chunked_attention(e.model,256):
  for c in cases():
   if c['case_id'] not in picked:continue
   torch.cuda.reset_peak_memory_stats();t=time.time()
   m.zero_grad(set_to_none=True)
   train_case(e,c,m,LAYER,accumulation=1)
   grad=torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True)
   assert float(grad)>0 and all(p.grad is None for p in e.model.parameters())
   torch.cuda.synchronize()
   results.append(dict(case_id=c['case_id'],nodes=len(nodes(c)),seconds=time.time()-t,peak_gb=torch.cuda.max_memory_allocated()/1e9,gradient=float(grad)))
 write(DEST/'pilot.json',results)

def train():
 e,m=build('soft');opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01)
 samples=cases()
 with query_chunked_attention(e.model,256):
  for epoch in [1,2,3]:
   ck=DEST/f'soft_epoch{epoch}.pt'
   if ck.exists():
    d=torch.load(ck,map_location=e.device,weights_only=False);m.load_state_dict(d['model']);opt.load_state_dict(d['optimizer']);continue
   order=list(samples);random.Random(SEED+epoch).shuffle(order)
   m.train();opt.zero_grad(set_to_none=True)
   for i,c in enumerate(order):
    loss=train_case(e,c,m,LAYER)
    if (i+1)%8==0 or i+1==len(order):
     if (i+1)==len(order) and len(order)%8:
      for p in m.parameters():
       if p.grad is not None:p.grad.mul_(8/(len(order)%8))
     torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True)
     opt.step();opt.zero_grad(set_to_none=True)
    write(DEST/'soft_status.json',dict(epoch=epoch,cases_done=i+1,total=len(order),last_loss=loss['loss']))
   tmp=ck.with_suffix('.tmp');save_checkpoint(tmp,m,opt,epoch,LAYER,'no_js',{});tmp.replace(ck)
 del e,m,opt
 import gc
 gc.collect();torch.cuda.empty_cache()

def eval_arm(arm):
 e,_=build(arm)
 ck=OUT/'no_js_layer18/epoch3.pt' if arm=='internal' else DEST/'soft_epoch3.pt'
 m,layer=load_module(e,ck)
 output=DEST/(arm+'_predictions.jsonl')
 done={(r['pair_id'],r['state'],r['view']) for r in read(output)} if output.exists() else set()
 with torch.no_grad(),query_chunked_attention(e.model,256),output.open('a') as handle:
  for p in read(DEST/'pairs.jsonl'):
   for state in ['yes','no']:
    for view in ['original','sibling_swap']:
     if (p['pair_id'],state,view) in done:continue
     guide=copy.deepcopy(p['guideline'])
     if view=='sibling_swap':
      group=next(g for g in guide['groups'] if g['group_id']==p['gid'])
      r0,r1=group['rules'];a,b=[re.fullmatch(r'(L\d+[：:])(.*)',r['action'],re.S) for r in [r0,r1]]
      r0['action'],r1['action']=a[1]+b[2],b[1]+a[2]
     case={'case_id':p['pair_id']+':'+state,'source_id':p['source_id'],'patient':p[state],'guideline':guide,'question':p['question']}
     z=e.blind(case,p['gid'])['states'][layer]
     logits,_,diag=e.training_decision(e.context(case),p['history'],p['gid'],layer,m,z,training=False)
     pred='yes' if int(logits.argmax())==0 else 'no'
     row=dict(pair_id=p['pair_id'],source=p['source_id'],state=state,view=view,prediction=pred,correct=pred==state,logits=logits.tolist())
     handle.write(json.dumps(row)+'\n');handle.flush()

def summary():
 from scipy.stats import binomtest
 import numpy as np
 all_rows={}
 for arm in ['internal','soft']:
  p=DEST/(arm+'_predictions.jsonl')
  if not p.exists():return
  all_rows[arm]={(r['pair_id'],r['state'],r['view']):r['correct'] for r in read(p)}
 pairs=read(DEST/'pairs.jsonl')
 if any(len(d)!=4*len(pairs) for d in all_rows.values()):return
 out={}
 for metric,states,views in [('four_correct',['yes','no'],['original','sibling_swap']),('fact_pair_original',['yes','no'],['original']),('fact_pair_swapped',['yes','no'],['sibling_swap']),('both_views_yes',['yes'],['original','sibling_swap']),('both_views_no',['no'],['original','sibling_swap'])]:
  counts=[0,0,0,0];diff=[]
  for p in pairs:
   a,b=[all(d[p['pair_id'],s,v] for s in states for v in views) for d in all_rows.values()]
   counts[0 if a and b else 1 if a else 2 if b else 3]+=1;diff.append(int(a)-int(b))
  both,io,so,neither=counts
  ci=np.quantile(np.random.default_rng(SEED).choice(diff,(10000,len(diff)),replace=True).mean(axis=1),[.025,.975]).tolist()
  out[metric]=dict(n=len(pairs),internal=both+io,soft=both+so,both=both,internal_only=io,soft_only=so,neither=neither,net_proportion=sum(diff)/len(diff),paired_source_bootstrap_ci95=ci,exact_mcnemar_p=binomtest(io,io+so).pvalue if io+so else 1.)
 write(DEST/'paired_results.json',out)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('task',choices=['pilot','train_soft','internal','soft','summary']);a=p.parse_args()
 if a.task=='pilot':pilot()
 elif a.task=='train_soft':train();eval_arm('soft');summary()
 elif a.task=='summary':summary()
 else:eval_arm(a.task);summary()
