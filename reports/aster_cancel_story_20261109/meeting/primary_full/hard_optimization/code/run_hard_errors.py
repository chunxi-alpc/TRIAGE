import sys
sys.path.insert(0,'/home/gcx/aster_next_20261109/inference')

"""Error-case-only evaluation of new frozen methods, with lazy missing-node forwards."""
import os,sys,json,time,argparse,traceback
from pathlib import Path
from collections import defaultdict
import numpy as np
from run import Engine,canonical_guide,prompt,score_selected,read_rows,dump,append
from hard_adapter import HardMethods as FrozenMethods
BASE=Path('/home/gcx/aster_next_20261109')
LEGACY=Path('/home/gcx/aster_cr_repair_v2_run_20261108/qwen')
LEGACY_META=Path('/home/gcx/aster_cancel_story_20261109/meeting/real_completion/metadata.json')
DEFAULT_METHODS=['raw','old_linear','optimized_balanced_fit_calibration','fit_bias_only','optimized_strict_fit_calibration','trace_only_balanced_fit_calibration']
def iter_rows(path):
 path=Path(path)
 if not path.exists():return
 with path.open('rb') as f:
  while True:
   line=f.readline()
   if not line:break
   if not line.strip():continue
   try:yield json.loads(line)
   except (json.JSONDecodeError,UnicodeDecodeError):
    if f.read(1):raise
    break
def local_rows(path):
 path=Path(path)
 if not path.exists():return []
 values=[]
 with path.open('rb+') as f:
  while True:
   start=f.tell();line=f.readline()
   if not line:break
   if not line.strip():continue
   try:values.append(json.loads(line))
   except (json.JSONDecodeError,UnicodeDecodeError):
    if f.read(1):raise
    f.truncate(start);break
 return values
def shard_cases(path,shard,shards,limit=None):
 result=[];ordinal=0
 with Path(path).open('rb') as f:
  for line in f:
   if not line.strip():continue
   if ordinal%shards==shard:
    result.append(json.loads(line))
    if limit is not None and len(result)>=limit:break
   ordinal+=1
 return result
def evaluation_group(case):return case.get('evaluation_group',case.get('history_group',case.get('dataset_split',case.get('split','unspecified'))))
def tree_nodes(case):
 if case.get('task_type')=='binary':
  gid=case.get('gid','G0')
  return {gid:dict(id=gid,yes=case.get('positive_leaf','L1'),no=case.get('negative_leaf','L0'))},gid
 tree=canonical_guide(case['guideline']);nodes={n['id']:n for n in tree['nodes']}
 return nodes,tree['nodes'][0]['id']
def node_prompt(case,gid):
 if case.get('task_type')=='binary' and case.get('prompt_text') is not None:return case['prompt_text']
 if case.get('node_prompts') and gid in case['node_prompts']:return case['node_prompts'][gid]
 return prompt(case['guideline'],case.get('definitions',[]),case['patient'],gid)
def normalize_metadata(record,case,shard,gpu):
 r=dict(record)
 r.update(case_id=case['case_id'],source_id=case['source_id'],source_key=case.get('source_key'),
  dataset=case.get('dataset','MEGA-CDP'),dataset_split=case.get('dataset_split'),split=case.get('split'),
  split_group=evaluation_group(case),evaluation_group=evaluation_group(case),
  current_source_role=case.get('current_source_role'),input_variant='canonical',shard=shard,gpu=gpu)
 return r
def reuse_files(roots,filename):
 found=set()
 for raw in roots:
  root=Path(raw)
  for candidate in [root/filename,*root.glob('shard*/'+filename),*root.glob('results/shard*/'+filename)]:
   if candidate.is_file():found.add(candidate.resolve())
 return sorted(found)
def collect_prior(case_ids,reuse_roots,own_nodes):
 prior=dict(own_nodes);origins={}
 files=reuse_files(reuse_roots,'nodes.jsonl')
 for file in files:
  for r in iter_rows(file):
   if r.get('case_id') not in case_ids or r.get('input_variant','canonical')!='canonical' or '|' in str(r.get('id','')):continue
   if r.get('status')=='ok' and r.get('feature') and not Path(r['feature']).exists():continue
   if r.get('status') not in ('ok','unavailable'):continue
   if r['id'] not in prior:prior[r['id']]=r;origins[r['id']]=str(file)
 # Existing full fit/dev/test path caches are immutable canonical inputs.
 for file in sorted((LEGACY/'paths').glob('features_*.jsonl')):
  for r in iter_rows(file):
   if r.get('case_id') not in case_ids or r.get('status')!='ok' or r['id'] in prior:continue
   if not r.get('feature') or not Path(r['feature']).exists():continue
   prior[r['id']]=r;origins[r['id']]=str(file)
 # Natural full-node feature records fill any path-cache gaps without parsing prompt text.
 if LEGACY_META.exists():
  meta=json.loads(LEGACY_META.read_text());wanted={}
  for v in meta['views']:
   if v.get('case_id') in case_ids and v.get('variant')=='full' and v.get('original'):
    gid=str(v['group_id']).rsplit('-',1)[-1];key=v['case_id']+':'+gid
    if key not in prior:wanted[v['view_id']]=(key,v['case_id'],gid)
  del meta
  if wanted:
   for file in sorted((LEGACY/'scores').glob('*.jsonl')):
    for r in iter_rows(file):
     if r.get('view_id') not in wanted or r.get('status')!='ok':continue
     key,cid,gid=wanted[r['view_id']]
     if key in prior or not r.get('feature') or not Path(r['feature']).exists():continue
     prior[key]=dict(r,id=key,case_id=cid,gid=gid);origins[key]=str(file)
 return prior,origins,len(files)
def run(args):
 out=Path(args.out_root)/('shard'+str(args.shard));out.mkdir(parents=True,exist_ok=True)
 (out/'features').mkdir(exist_ok=True)
 cases=shard_cases(args.cases,args.shard,args.shards,args.limit);bycase={c['case_id']:c for c in cases};case_ids=set(bycase)
 names=[]
 for token in args.methods:
  for name in token.split(','):
   if name and name not in names:names.append(name)
 manifest=json.loads(Path(args.manifest).read_text());valid={'raw','fit_bias_only','old_linear'}|set(manifest['methods'])
 bad=set(names)-valid
 if bad:raise ValueError('Only canonical frozen methods are supported: '+str(sorted(bad)))
 written=defaultdict(set);existing=local_rows(out/'results.jsonl')
 for r in existing:
  if r['case_id'] in bycase and r['method'] in names:written[r['case_id']].add(r['method'])
 imported=0
 for file in reuse_files(args.reuse_root,'results.jsonl'):
  if file== (out/'results.jsonl').resolve():continue
  for r in iter_rows(file):
   cid=r.get('case_id');name=r.get('method')
   if cid not in bycase or name not in names or name in written[cid]:continue
   if r.get('input_variant','canonical')!='canonical':continue
   record=normalize_metadata(r,bycase[cid],args.shard,args.gpu)
   record.update(reused_result=True,reused_from_file=str(file),original_shard=r.get('shard'),original_gpu=r.get('gpu'))
   if not args.dry_run:append(out/'results.jsonl',record)
   written[cid].add(name);imported+=1
 pending=[c for c in cases if set(names)-written[c['case_id']]]
 own_nodes={r['id']:r for r in local_rows(out/'nodes.jsonl')}
 prior,origins,node_files=collect_prior(case_ids,args.reuse_root,own_nodes) if pending else (own_nodes,{},0)
 info=dict(gpu=args.gpu,shard=args.shard,shards=args.shards,assigned_cases=len(cases),pending_cases=len(pending),
  completed_cases=len(cases)-len(pending),methods=names,imported_method_records=imported,cached_canonical_nodes=len(prior),
  reuse_roots=args.reuse_root,cases_manifest=str(args.cases),out_root=str(args.out_root),prompt_forwards_enabled=False)
 dump(out/('dry_run.json' if args.dry_run else 'resume.json'),info)
 if args.dry_run:print(json.dumps(info));return
 if not pending:
  dump(out/'done.json',dict(state='complete',already_complete=True,**info));return
 import torch
 torch.set_num_threads(2);torch.cuda.set_device(args.gpu)
 adapters=FrozenMethods(args.manifest);started=time.time();engine=None;model_loaded=None
 new_nodes=0;reused_nodes=0;node_scores={}
 def node(case,gid):
  nonlocal engine,model_loaded,new_nodes,reused_nodes
  key=case['case_id']+':'+gid
  if key in node_scores:return node_scores[key]
  if key in prior:
   rec=dict(prior[key]);reused_nodes+=1
   if key not in own_nodes:
    rec.update(id=key,case_id=case['case_id'],gid=gid,source_id=case['source_id'],input_variant='canonical',reused_from_file=origins.get(key,'existing_node_cache'))
    append(out/'nodes.jsonl',rec);own_nodes[key]=rec
  else:
   if engine is None:
    load_start=time.time();engine=Engine('qwen',device='cuda:'+str(args.gpu));model_loaded=time.time()-load_start
   rec=dict(id=key,case_id=case['case_id'],source_id=case['source_id'],gid=gid,input_variant='canonical')
   try:
    data=score_selected(engine,node_prompt(case,gid),adapters.heads);feature=out/'features'/(key.replace('/','_')+'.npz')
    temp=feature.with_suffix('.npz.tmp')
    with temp.open('wb') as f:np.savez_compressed(f,selected_heads=data.pop('selected_heads'),residual=data.pop('residual'))
    temp.replace(feature);rec.update(data,feature=str(feature),status='ok')
   except ValueError as exc:rec.update(status='unavailable',error=str(exc))
   append(out/'nodes.jsonl',rec);own_nodes[key]=rec;prior[key]=rec;new_nodes+=1
  result=adapters.predict_file(rec) if rec.get('status')=='ok' else None
  node_scores[key]=result
  return result
 for case in pending:
  # Completed methods stay complete; only unfinished paths may request more nodes.
  active_names=[name for name in names if name not in written[case['case_id']]]
  nodes,root=tree_nodes(case);states={name:dict(gid=root,path=[],error=None) for name in active_names}
  while True:
   active={name:s['gid'] for name,s in states.items() if s['error'] is None and s['gid'] in nodes}
   if not active:break
   for gid in dict.fromkeys(active.values()):
    scores=node(case,gid)
    for name,requested in active.items():
     if requested!=gid:continue
     state=states[name]
     if scores is None:state['error']='unavailable_score';continue
     if len(state['path'])>len(nodes):state['error']='cycle';continue
     label=int(scores[name]>0);state['path'].append(dict(gid=gid,label=label))
     state['gid']=nodes[gid]['yes' if label else 'no']
  for name,state in states.items():
   leaf_correct=state['gid']==case['gold_leaf']
   correct=state['error'] is None and state['path']==case['gold_path'] and leaf_correct
   first=next((p['gid'] for i,p in enumerate(state['path']) if i>=len(case['gold_path']) or p!=case['gold_path'][i]),None)
   record=normalize_metadata(dict(method=name,correct=bool(correct),leaf_correct=bool(leaf_correct),predicted_path=state['path'],
    gold_path=case['gold_path'],predicted_leaf=state['gid'],gold_leaf=case['gold_leaf'],error=state['error'],first_error=first,
    node_forward_count=len(state['path']),status='ok' if state['error'] is None else 'unavailable'),case,args.shard,args.gpu)
   append(out/'results.jsonl',record);written[case['case_id']].add(name)
  node_scores.clear()
  dump(out/'status.json',dict(state='running',gpu=args.gpu,shard=args.shard,cases_completed=sum(set(names)<=written[c['case_id']] for c in cases),
   total_cases=len(cases),methods=names,new_node_forwards=new_nodes,reused_nodes=reused_nodes,imported_method_records=imported,
   seconds=time.time()-started,model_load_seconds=model_loaded,peak_gib=torch.cuda.max_memory_allocated(args.gpu)/2**30,
   last_case=case['case_id'],prompt_forwards_enabled=False))
 dump(out/'completed_case_ids.json',[c['case_id'] for c in cases if set(names)<=written[c['case_id']]])
 dump(out/'done.json',dict(state='complete',gpu=args.gpu,shard=args.shard,shards=args.shards,cases=len(cases),methods=names,new_node_forwards=new_nodes,
  reused_nodes=reused_nodes,imported_method_records=imported,seconds=time.time()-started,model_load_seconds=model_loaded,
  counts=engine.counts if engine else {},peak_gib=torch.cuda.max_memory_allocated(args.gpu)/2**30,prompt_forwards_enabled=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cases',required=True);p.add_argument('--out-root',required=True);p.add_argument('--manifest',default=str(BASE/'frozen_methods.json'))
 p.add_argument('--gpu',type=int,required=True);p.add_argument('--shard',type=int,required=True);p.add_argument('--shards',type=int,default=4)
 p.add_argument('--methods',nargs='+',default=DEFAULT_METHODS);p.add_argument('--reuse-root',action='append',default=[])
 p.add_argument('--limit',type=int);p.add_argument('--dry-run',action='store_true');args=p.parse_args()
 try:run(args)
 except Exception as exc:
  out=Path(args.out_root)/('shard'+str(args.shard));out.mkdir(parents=True,exist_ok=True)
  dump(out/'failed.json',dict(state='failed',error=str(exc),traceback=traceback.format_exc()));raise

