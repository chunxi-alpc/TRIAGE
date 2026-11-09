
"""Resumable official-test dynamic paths; frozen methods share each node forward."""
import os,sys,json,time,argparse,traceback,importlib.util
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.path.insert(0,'/home/gcx/TRIAGE-worktrees/aster-cr-20261108')
from experiments.aster_cr.engine import Engine
from experiments.aster_closed_form.data import canonical_guide,prompt
from frozen import FrozenMethods
ROOT=Path('/home/gcx/aster_next_20261109')
def read_rows(path,repair=False):
 path=Path(path)
 if not path.exists():return []
 records=[]
 with path.open('rb+' if repair else 'rb') as f:
  while True:
   start=f.tell();line=f.readline()
   if not line:break
   if not line.strip():continue
   try:records.append(json.loads(line))
   except json.JSONDecodeError:
    if repair and not f.read(1):f.truncate(start);break
    raise
 return records
def dump(path,obj):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def append(path,obj):
 with Path(path).open('a',encoding='utf-8') as f:
  f.write(json.dumps(obj,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
def score_selected(engine,text,selected):
 """Same canonical prepare/final replay, only copy the selected heads to CPU."""
 import torch
 started=time.perf_counter();context=engine.prepare(text);bylayer={};captured={};residual={};handles=[]
 for h in selected:bylayer.setdefault(int(h)//engine.n_heads,[]).append(int(h)%engine.n_heads)
 def head_hook(layer,indices):
  def hook(module,args):
   value=args[0]
   if value.shape!=(1,1,engine.n_heads*engine.head_dim):raise RuntimeError('Unexpected final-token shape')
   captured[layer]=value[0,0].reshape(engine.n_heads,engine.head_dim)[indices].detach()
  return hook
 def residual_hook(module,args,out):
  value=out[0] if isinstance(out,tuple) else out
  residual['h']=value[0,-1].detach()
 before=time.perf_counter();engine._active=True
 try:
  for layer,indices in bylayer.items():handles.append(engine.layers[layer].self_attn.o_proj.register_forward_pre_hook(head_hook(layer,indices)))
  handles.append(engine.layers[-1].register_forward_hook(residual_hook))
  with torch.no_grad():
   embeds=engine.model.get_input_embeddings()(context.input_ids[:,-1:]).detach()
   output=engine.model.model(inputs_embeds=embeds,attention_mask=torch.ones_like(context.input_ids),
    past_key_values=engine._cache(context),use_cache=True,
    cache_position=torch.tensor([context.length-1],device=engine.device),
    position_ids=torch.tensor([[context.length-1]],device=engine.device))
   value=engine._score_hidden(output.last_hidden_state[0,-1]);ell=float(value)
   positions={(layer,head):j for layer,indices in bylayer.items() for j,head in enumerate(indices)}
   hh=torch.stack([captured[int(h)//engine.n_heads][positions[int(h)//engine.n_heads,int(h)%engine.n_heads]] for h in selected])
   heads=hh.float().cpu().numpy().reshape(-1).astype('float16')
   h=residual['h'].float().cpu().numpy().astype('float16')
  engine.counts['final_forwards']+=1;engine.counts['final_seconds']+=time.perf_counter()-before
  return dict(ell=ell,prediction=int(ell>0),tokens=context.length,selected_heads=heads,residual=h,
    inference_seconds=time.perf_counter()-started)
 finally:
  for handle in handles:handle.remove()
  engine._active=False
def run(gpu,shard,shards,limit=None,validate=False):
 import torch
 torch.set_num_threads(2);torch.cuda.set_device(gpu)
 out=ROOT/'results'/('shard'+str(shard));out.mkdir(parents=True,exist_ok=True)
 (out/'features').mkdir(exist_ok=True)
 cases=read_rows(ROOT/'inventory/path_cases.jsonl')[shard::shards]
 if limit is not None:cases=cases[:limit]
 adapters=FrozenMethods();names=list(adapters.names)
 prompt_cfg=adapters.manifest.get('evidence_alignment')
 prompt_names={};enhance=None;prompt_variant='evidence_alignment_v1'
 if prompt_cfg and prompt_cfg.get('enabled',True):
  module_path=Path(prompt_cfg.get('module',ROOT/'diagnosis/evidence_alignment.py'))
  spec=importlib.util.spec_from_file_location('fixed_evidence_alignment',module_path)
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  enhance=getattr(module,prompt_cfg.get('function','enhance_prompt'))
  prompt_variant=prompt_cfg.get('id','evidence_alignment_v1')
  prompt_names[prompt_cfg.get('name','evidence_alignment_fit_calibration')]=float(prompt_cfg['b'])
  if prompt_cfg.get('include_raw',False):prompt_names[prompt_cfg.get('raw_name','evidence_alignment_raw')]=0.
  names.extend(prompt_names)
 variants={name:(prompt_variant if name in prompt_names else 'canonical') for name in names}
 existing=read_rows(out/'results.jsonl',repair=True);written=defaultdict(set)
 for r in existing:written[r['case_id']].add(r['method'])
 node_records={r['id']:r for r in read_rows(out/'nodes.jsonl',repair=True)}
 # Prior canonical path cache is indexed by exact case/node identity.
 prior={}
 prior_root=Path('/home/gcx/aster_cr_repair_v2_run_20261108/qwen')
 case_ids={c['case_id'] for c in cases}
 for p in (prior_root/'paths').glob('features_*.jsonl'):
  for r in read_rows(p):
   if r.get('status')=='ok' and r['case_id'] in case_ids:prior[r['id']]=r
 pending=[c for c in cases if set(names)-written[c['case_id']]]
 if not pending:
  dump(out/'done.json',dict(state='complete',gpu=gpu,shard=shard,cases=len(cases),methods=names,already_complete=True));return
 start=time.time();engine=Engine('qwen',device='cuda:'+str(gpu));model_loaded=time.time()-start
 new_nodes=0;reused_nodes=0;node_scores={}
 def node(case,gid,variant='canonical',text_override=None):
  nonlocal new_nodes,reused_nodes
  key=case['case_id']+':'+gid+(('|'+variant) if variant!='canonical' else '')
  if key in node_scores:return node_scores[key]
  if key in node_records:
   rec=node_records[key];reused_nodes+=1
  elif variant=='canonical' and key in prior:
   rec=dict(prior[key],id=key,status='ok',reused_from='repair_v2_canonical_cache')
   append(out/'nodes.jsonl',rec);node_records[key]=rec;reused_nodes+=1
  else:
   text=text_override or prompt(case['guideline'],case['definitions'],case['patient'],gid)
   rec=dict(id=key,case_id=case['case_id'],source_id=case['source_id'],gid=gid,input_variant=variant)
   try:
    if variant=='canonical':
     data=score_selected(engine,text,adapters.heads);feature=out/'features'/(key+'.npz')
     temp=feature.with_suffix('.npz.tmp')
     with temp.open('wb') as f:np.savez_compressed(f,selected_heads=data.pop('selected_heads'),residual=data.pop('residual'))
     temp.replace(feature);rec.update(data,feature=str(feature),status='ok')
    else:
     data=engine.score(enhance(text),capture=False)
     rec.update(data,status='ok')
   except ValueError as exc:rec.update(status='unavailable',error=str(exc))
   append(out/'nodes.jsonl',rec);node_records[key]=rec;new_nodes+=1
  if rec.get('status')!='ok':result=None
  elif variant=='canonical':result=adapters.predict_file(rec)
  else:result={name:float(rec['ell']+bias) for name,bias in prompt_names.items()}
  node_scores[key]=result
  return result
 if validate:
  c=pending[0];gid=canonical_guide(c['guideline'])['nodes'][0]['id'];text=prompt(c['guideline'],c['definitions'],c['patient'],gid)
  old=engine.score(text,capture=True)
  actual=score_selected(engine,text,adapters.heads)
  oldheads=old['heads'].numpy().reshape(-1,engine.head_dim)[adapters.heads].reshape(-1)
  err=dict(ell=abs(float(old['ell'])-actual['ell']),heads=float(np.max(np.abs(oldheads-actual['selected_heads'].astype('float32')))),
   residual=float(np.max(np.abs(old['residual'].numpy()-actual['residual'].astype('float32')))))
  dump(out/'canonical_validation.json',dict(errors=err,passed=max(err.values())<=1e-5))
  if max(err.values())>1e-5:raise RuntimeError('Selected-head scoring did not match canonical scoring: '+str(err))
  key=c['case_id']+':'+gid;feature=out/'features'/(key+'.npz')
  with feature.open('wb') as f:np.savez_compressed(f,selected_heads=actual.pop('selected_heads'),residual=actual.pop('residual'))
  rec=dict(actual,id=key,case_id=c['case_id'],source_id=c['source_id'],gid=gid,feature=str(feature),status='ok',validation_reused=True)
  node_records[key]=rec;append(out/'nodes.jsonl',rec)
 for ordinal,case in enumerate(cases):
  if set(names)<=written[case['case_id']]:continue
  tree=canonical_guide(case['guideline']);nodes={n['id']:n for n in tree['nodes']};root=tree['nodes'][0]['id']
  states={name:dict(gid=root,path=[],error=None) for name in names}
  while True:
   active={name:s['gid'] for name,s in states.items() if s['error'] is None and s['gid'] in nodes}
   if not active:break
   for variant,gid in dict.fromkeys((variants[name],gid) for name,gid in active.items()):
    scores=node(case,gid,variant)
    for name,requested in active.items():
     if requested!=gid or variants[name]!=variant:continue
     state=states[name]
     if scores is None:state['error']='unavailable_score';continue
     if len(state['path'])>len(nodes):state['error']='cycle';continue
     label=int(scores[name]>0);state['path'].append(dict(gid=gid,label=label))
     state['gid']=nodes[gid]['yes' if label else 'no']
  for name,state in states.items():
   if name in written[case['case_id']]:continue
   leaf_correct=state['gid']==case['gold_leaf'];correct=state['error'] is None and state['path']==case['gold_path'] and leaf_correct
   first_error=next((p['gid'] for i,p in enumerate(state['path']) if i>=len(case['gold_path']) or p!=case['gold_path'][i]),None)
   record=dict(case_id=case['case_id'],source_id=case['source_id'],source_key=case.get('source_key'),method=name,
    correct=bool(correct),leaf_correct=bool(leaf_correct),predicted_path=state['path'],gold_path=case['gold_path'],
    predicted_leaf=state['gid'],gold_leaf=case['gold_leaf'],split_group=case.get('history_group','unspecified'),
    current_source_role=case.get('current_source_role'),input_variant=variants[name],status='ok' if state['error'] is None else 'unavailable',
    error=state['error'],first_error=first_error,node_forward_count=len(state['path']),gpu=gpu,shard=shard)
   append(out/'results.jsonl',record);written[case['case_id']].add(name)
  node_scores={k:v for k,v in node_scores.items() if not k.startswith(case['case_id']+':')}
  dump(out/'status.json',dict(state='running',gpu=gpu,shard=shard,cases_completed=sum(set(names)<=written[c['case_id']] for c in cases),
   total_cases=len(cases),new_node_forwards=new_nodes,reused_nodes=reused_nodes,seconds=time.time()-start,
   model_load_seconds=model_loaded,peak_gib=torch.cuda.max_memory_allocated(gpu)/2**30,last_case=case['case_id']))
 dump(out/'done.json',dict(state='complete',gpu=gpu,shard=shard,cases=len(cases),methods=names,new_node_forwards=new_nodes,
  reused_nodes=reused_nodes,seconds=time.time()-start,model_load_seconds=model_loaded,counts=engine.counts,peak_gib=torch.cuda.max_memory_allocated(gpu)/2**30))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--gpu',required=True,type=int);p.add_argument('--shard',required=True,type=int);p.add_argument('--shards',type=int,default=2);p.add_argument('--limit',type=int);p.add_argument('--validate',action='store_true')
 a=p.parse_args()
 try:run(a.gpu,a.shard,a.shards,a.limit,a.validate)
 except Exception as exc:
  out=ROOT/'results'/('shard'+str(a.shard));out.mkdir(parents=True,exist_ok=True)
  dump(out/'failed.json',dict(state='failed',error=str(exc),traceback=traceback.format_exc()));raise

