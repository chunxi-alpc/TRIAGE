"""Frozen E1 final-judgment interventions; no fitting or head/path selection."""
import argparse,collections,difflib,time
import numpy as np
import torch
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
from experiments.aster_cr.engine import Engine
from .common import *
def groups(heads,n):
 d=collections.defaultdict(list)
 for h in heads:d[h//n].append(h%n)
 return dict(d)
def capture(eng,ctx,union):
 hs=[];data={}
 for i in range(eng.n_layers):
  d=data[i]={}
  for name in ('q','k','v'):
   def hook(m,a,o,d=d,name=name):d[name]=o.detach().clone()
   hs.append(getattr(eng.layers[i].self_attn,name+'_proj').register_forward_hook(hook))
  def poshook(m,a,kw,d=d):d['pos']=tuple(x.detach() for x in kw['position_embeddings'])
  hs.append(eng.layers[i].self_attn.register_forward_pre_hook(poshook,with_kwargs=True))
 try:r=eng.run(ctx,capture=True)
 finally:
  for h in hs:h.remove()
 for i,js in groups(union,eng.n_heads).items():
  att=eng.layers[i].self_attn;d=data[i]
  q=att.q_norm(d['q'].view(1,1,eng.n_heads,eng.head_dim)).transpose(1,2)
  k=att.k_norm(d['k'].view(1,1,-1,eng.head_dim)).transpose(1,2)
  q,k=apply_rotary_pos_emb(q,k,*d['pos'])
  keys=torch.cat((ctx.prefix.layers[i].keys,k),dim=2)
  vals=torch.cat((ctx.prefix.layers[i].values,d['v'].view(1,1,-1,eng.head_dim).transpose(1,2)),dim=2)
  ratio=eng.n_heads//keys.shape[1];d['a']={};d['values']={}
  for j in js:
   kv=j//ratio
   d['a'][j]=torch.softmax((q[0,j,0].float()@keys[0,kv].float().T)*att.scaling,dim=-1)
   d['values'][j]=vals[0,kv]
 r['data']=data
 return r
def freeze_qk(eng,base):
 hs=[]
 for i,layer in enumerate(eng.layers):
  for name in ('q','k'):
   def hook(m,a,o,value=base['data'][i][name]):return value
   hs.append(getattr(layer.self_attn,name+'_proj').register_forward_hook(hook))
 return hs
def run_intervention(eng,ctx,base,donor,heads,other,mode,align,readout):
 frozen=mode in ('output_qk_frozen','ov_values_qk_frozen','routing_only_qk_frozen')
 hs=freeze_qk(eng,base) if frozen else []
 bh=base['heads'].reshape(-1,eng.head_dim).to(eng.device,dtype=torch.bfloat16)
 dh=donor['heads'].reshape(-1,eng.head_dim).to(eng.device,dtype=torch.bfloat16)
 replacements={};local2=0.;projected2=0.
 idxr=torch.tensor([x for x,y in align],device=eng.device)
 idxd=torch.tensor([y for x,y in align],device=eng.device)
 for i,js in groups(heads,eng.n_heads).items():
  ds=[];patch={}
  for j in js:
   h=i*eng.n_heads+j
   delta=dh[h].float()-bh[h].float()
   if mode=='ov_values_qk_frozen':
    bd=base['data'][i];dd=donor['data'][i]
    delta=bd['a'][j][idxr]@(dd['values'][j][idxd].float()-bd['values'][j][idxr].float())
   elif mode=='routing_only_qk_frozen':
    bd=base['data'][i];dd=donor['data'][i]
    a=bd['a'][j].clone();a[idxr]=dd['a'][j][idxd];a/=a.sum()
    delta=(a-bd['a'][j])@bd['values'][j].float()
   patch[h]=(bh[h].float()+delta).to(torch.bfloat16)
   ds.append(delta);local2+=float(delta.square().sum())
  w=eng.layers[i].self_attn.o_proj.weight
  if mode in ('projection_swap','projection_native'):
   projected=torch.zeros(w.shape[0],device=eng.device,dtype=torch.float32)
   for j,delta in zip(js,ds):
    h=i*eng.n_heads+j;alt=other[h]%eng.n_heads if mode=='projection_swap' else j
    own=w[:,j*eng.head_dim:(j+1)*eng.head_dim].float()
    altw=w[:,alt*eng.head_dim:(alt+1)*eng.head_dim].float()
    altw=altw*(own.norm()/altw.norm().clamp_min(1e-12))
    projected+=altw@delta
   def ohook(m,a,o,delta=projected):return o+delta.to(o.dtype).view(1,1,-1)
   hs.append(eng.layers[i].self_attn.o_proj.register_forward_hook(ohook))
   projected2+=float(projected.square().sum())
  else:
   replacements.update(patch)
   projected=sum(w[:,j*eng.head_dim:(j+1)*eng.head_dim].float()@d for j,d in zip(js,ds))
   projected2+=float(projected.square().sum())
 try:
  r=eng.run(ctx,replacements=replacements,capture=True)
  # Engine captures the input before applying its replacement hook. Readout must use effective head values.
  effective=r['heads'].reshape(-1,eng.head_dim)
  if replacements:
   keys=list(replacements)
   effective[keys]=torch.stack([replacements[h] for h in keys]).detach().float().cpu()
 finally:
  for h in hs:h.remove()
 return dict(ell=r['ell'],aster_score=readout.score(r['ell'],r['heads']),
             local_delta_l2=local2**.5,projected_delta_l2=projected2**.5)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=0);args=ap.parse_args()
 torch.set_num_threads(4);torch.set_grad_enabled(False)
 cfg=read(OUT/'configs/e9.json');selected=cfg['selected_heads'];random=cfg['random_heads']
 pairs=rows(OUT/'configs/e9_pairs.jsonl');pairs=pairs[:args.limit] if args.limit else pairs
 target=OUT/('e9_smoke.jsonl' if args.limit else 'e9_predictions.jsonl')
 done={(r['pair_id'],r['direction'],r['group'],r['intervention']) for r in rows(target)} if target.exists() else set()
 if len(done)==len(pairs)*2*2*len(cfg['interventions']):return
 lookup={r['uid']:r for r in rows(OUT/'configs/e9_inputs.jsonl')} if (OUT/'configs/e9_inputs.jsonl').exists() else inputs()
 eng=Engine('qwen');rd=Readout(eng.device)
 union=sorted(set(selected+random));swap=dict(zip(selected,random));swap.update(zip(random,selected))
 started=time.time()
 for pi,p in enumerate(pairs):
  if all((p['pair_id'],d,g,m) in done for d in ('forward','reverse') for g in ('selected','random') for m in cfg['interventions']):continue
  prompts=[lookup[p[x]]['prompt'] for x in ('left','right')]
  contexts=[eng.prepare(s) for s in prompts];bases=[capture(eng,c,union) for c in contexts]
  token_ids=[c.input_ids[0].tolist() for c in contexts]
  match=difflib.SequenceMatcher(a=token_ids[0],b=token_ids[1],autojunk=False)
  align=[(b.a+j,b.b+j) for b in match.get_matching_blocks() for j in range(b.size)]
  for side in (0,1):
   a,b=bases[side],bases[1-side];ctx=contexts[side]
   alignment=align if side==0 else [(y,x) for x,y in align]
   label=p['left_label' if side==0 else 'right_label']
   common=dict(physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),pair_id=p['pair_id'],source_id=p['source_id'],kind=p['kind'],
    direction='forward' if side==0 else 'reverse',recipient_uid=p['left' if side==0 else 'right'],
    donor_uid=p['right' if side==0 else 'left'],label=label,base_ell=a['ell'],donor_ell=b['ell'],
    base_aster_score=rd.score(a['ell'],a['heads']),tokens=ctx.length,
    aligned_tokens=len(alignment),alignment_fraction=len(alignment)/ctx.length)
   for gn,hh in [('selected',selected),('random',random)]:
    for mode in cfg['interventions']:
     key=(p['pair_id'],common['direction'],gn,mode)
     if key in done:continue
     result=run_intervention(eng,ctx,a,b,hh,swap,mode,alignment,rd)
     append(target,dict(common,group=gn,intervention=mode,**result))
   # Only an implementation identity control, not another mechanism experiment.
   if pi==0:
    r=eng.run(ctx,replacements={h:a['heads'].reshape(-1,eng.head_dim)[h] for h in selected},capture=False)
    append(OUT/'e9_identity_controls.jsonl',dict(pair_id=p['pair_id'],direction=common['direction'],baseline=a['ell'],self_patch=r['ell']))
  del contexts,bases
  write(OUT/'e9_status.json',dict(state='running',done=pi+1,total=len(pairs),seconds=time.time()-started))
 append(OUT/'e9_runs.jsonl',dict(pairs=len(pairs),interventions=cfg['interventions'],seconds=time.time()-started,counts=eng.counts))
 write(OUT/'e9_status.json',dict(state='complete',pairs=len(pairs),seconds=time.time()-started,
  peak_gib=torch.cuda.max_memory_allocated()/2**30,counts=eng.counts))
if __name__=='__main__':
 import fcntl
 with (OUT/'e9.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  main()
