"""Append-only recovery of complete-input context failures, with original-model judging.
Original result files stay intact. Replays form an explicit replacement overlay only
for original context_overflow cells, never for output quality or parsing failures.
"""
import argparse,collections,fcntl,json,math,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
from types import SimpleNamespace
from batch_evaluate import history,read,write,evaluate_batch
from fullscale_run import METHODS,rows
PHASES={'confirm':('mega','confirm','mega_confirm'),'official_test':('mega','official_test','mega_official_test'),'med':('med','heldout','med_heldout')}
SCRIPTS=Path(__file__).resolve().parent

def alive(pid):
 try:return (Path('/proc')/str(pid)/'stat').read_text().split()[2]!='Z'
 except (OSError,IndexError):return False

def base_failures(root,phase):
 cases=collections.defaultdict(dict)
 for r in history(root/'serial'/phase):cases[r['case_id']][r['method']]=r
 return {c:rs for c,rs in cases.items() if len(rs)==len(METHODS) and all(r.get('technical_failure') and r.get('failure_reason')=='context_overflow' for r in rs.values())}

def completed(root,phase):
 cells=collections.defaultdict(set)
 for r in history(root/'context_replay'/phase):cells[r['case_id']].add(r['method'])
 return {c for c,ms in cells.items() if set(METHODS)<=ms}

def available_gpus(root):
 # Never borrow a card reserved by the active main stage, including brief idle gaps.
 state=read(root/'serial_status.json') or {};allocation=read(root/'scheduler'/(str(state.get('phase'))+'_allocation.json')) or {}
 reserved=set()
 for rank,gpu in enumerate(allocation.get('cards',[])):
  info=read(root/'scheduler'/(str(state.get('phase'))+'_'+str(rank)+'.json')) or {}
  status=read(Path(info.get('status_file','/nonexistent'))) or {}
  if alive(info.get('pid',0)):reserved.add(gpu)
 text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15)
 allowed=set((read(root/'resource_policy.json') or {}).get('allowed_gpus',[0,1,5,6]))
 return [g for g,m,u in ([int(x.strip()) for x in line.split(',')] for line in text.splitlines()) if g in allowed and g not in reserved and m<1024 and u<5]

class CaptureLength(Exception):pass

def prepare_cases(root,phase,failures):
 from transformers import AutoTokenizer
 from fullscale_engine import Engine
 phase_dir=root/'context_replay'/phase;phase_dir.mkdir(parents=True,exist_ok=True)
 manifest=read(phase_dir/'manifest.json') or {'cases':{},'created':time.time()}
 new=set(failures)-set(manifest['cases'])
 if not new:return manifest
 auditor=Engine.__new__(Engine);auditor.tok=AutoTokenizer.from_pretrained('/home/gcx/TRIAGE/models/modelscope/Qwen3-8B',local_files_only=True)
 lengths=[]
 def capture(text,offset=False):lengths.append(len(auditor.tok.encode(text,add_special_tokens=False)));raise CaptureLength()
 auditor.encode=capture
 dataset,split,tag=PHASES[phase]
 # Dataset split names are retained verbatim; use the original records as authority.
 splits={next(iter(failures[c].values()))['split'] for c in new}
 for c in rows(root,dataset,splits):
  cid=c['case_id']
  if cid not in new:continue
  try:auditor.task(c)
  except CaptureLength:pass
  n=lengths[-1];needed=n+(2048 if dataset=='mega' else 8)
  factor=max(2,2**math.ceil(math.log2(needed/32768)));capacity=int(32768*factor)
  item={'tokens':n,'capacity':capacity,'factor':factor,'split':c['split'],'dataset':dataset,'original_records':len(failures[cid])}
  directory=phase_dir/'cases'/cid;directory.mkdir(parents=True,exist_ok=True)
  write(directory/'case.json',c)
  write(directory/'original_records.json',list(failures[cid].values()))
  route=read(root/'serial'/phase/'batch_cache'/cid/'routing.json')
  if route is None:
   import re
   old=next(iter(failures[cid].values()));raw=old.get('raw_routing','');failure=old.get('route_failure');rmap={r['id']:0 for r in c['rules']}
   try:
    obj=json.loads(re.search(r'\{.*\}',raw,re.S).group());rmap={rid:{'T':1,'F':-1,'U':0}.get(str(obj.get(rid,'U')).upper(),0) for rid in rmap}
   except Exception:failure=failure or 'applicability_parse_failure'
   route={'route':rmap,'raw':raw,'failure':failure}
  write(phase_dir/'batch_cache'/cid/'routing.json',route)
  manifest['cases'][cid]=item
 write(phase_dir/'manifest.json',manifest)
 return manifest

def worker(a):
 import torch
 from transformers.models.qwen3.modeling_qwen3 import ALL_ATTENTION_FUNCTIONS
 from batch_engine import BatchEngine
 original=ALL_ATTENTION_FUNCTIONS['sdpa'];root=Path(a.run);job=Path(a.job);info=read(job/'job.json');phase=info['phase'];out=root/'context_replay'/phase/('worker_'+job.name);out.mkdir(parents=True,exist_ok=True)
 status=out/'status.json';write(status,{'status':'running','pid':os.getpid(),'started':time.time(),'cases':len(info['cases']),'capacity':info['capacity']})
 class ReplayEngine(BatchEngine):
  def generate_many(self,*args,**kwargs):
   ALL_ATTENTION_FUNCTIONS.register('sdpa',self.attention)
   return super().generate_many(*args,**kwargs)
  def prefill(self,prefix,position):
   if not getattr(self,'chunked_prefill',False):return super().prefill(prefix,position)
   from transformers.cache_utils import DynamicCache
   tokens=prefix['input_ids'][0].tolist();previous=getattr(self,'shared_prefill',None);common=0
   if previous:
    for old,new in zip(previous[0],tokens):
     if old!=new:break
     common+=1
   if common<128:common=0
   result=None;cache=DynamicCache.from_legacy_cache(tuple((k[:,:,:common],v[:,:,:common]) for k,v in previous[1])) if common else None
   self.batch_stats['common_prefix_tokens']=self.batch_stats.get('common_prefix_tokens',0)+common
   self.batch_stats['prefill_tokens']=self.batch_stats.get('prefill_tokens',0)+len(tokens)-common
   if common==len(tokens):return SimpleNamespace(past_key_values=cache)
   for start in range(common,position.shape[-1],2048):
    end=min(start+2048,position.shape[-1]);inputs={k:(v[:,start:end] if k=='input_ids' else v[:,:end]) for k,v in prefix.items()}
    result=self.model(**inputs,position_ids=position[:,start:end],past_key_values=cache,cache_position=torch.arange(start,end,device=self.model.device),use_cache=True,logits_to_keep=1);cache=result.past_key_values
   self.shared_prefill=(tokens,result.past_key_values.to_legacy_cache())
   return result
 try:
  model='/home/gcx/TRIAGE/models/modelscope/Qwen3-8B'
  judge=ReplayEngine(model);judge.original=original
  config={'max_position_embeddings':info['capacity'],'rope_scaling':{'rope_type':'yarn','factor':float(info['factor']),'original_max_position_embeddings':32768}}
  e=ReplayEngine(model,config_overrides=config);e.original=original;e.chunked_prefill=True;e.judge_engine=judge
  e.applicability_many=judge.applicability_many
  e.record_metadata={'supersedes_original_context_failure':True,'generation_context_profile':config,'judging_context_profile':'original_checkpoint_config','applicability_profile':'preserved_original_prediction','input_truncated':False,'context_replay_job':job.name}
  # A cheap cache-position check before the first real long-context case.
  if info.get('probe'):
   with torch.inference_mode():
    text=e.prompt('Continue this sequence: '+'1 2 3 4 5 6 7 8 9 10. '*130);x=e.tok(text,return_tensors='pt',add_special_tokens=False).to(e.model.device);prefix={k:v[:,:-1] for k,v in x.items()};position=prefix['attention_mask'].cumsum(-1)-1
    full=e.model(**prefix,position_ids=position,use_cache=True,logits_to_keep=1);reference=full.logits[:,-1].detach();del full
    chunk=e.prefill(prefix,position);difference=float((chunk.logits[:,-1]-reference).abs().max());same=bool(chunk.logits[:,-1].argmax()==reference.argmax());size=chunk.past_key_values.get_seq_length();assert size==position.shape[-1]
    write(job/'prefill_check.json',{'tokens':size,'same_top_token':same,'max_logit_difference':difference,'all_tokens_retained':True});del chunk,reference,x,prefix,position
   torch.cuda.empty_cache()
  args=SimpleNamespace(run=str(job),worker=0,world=1,generation_batch_size=1,judge_batch_size=8)
  artifact=torch.load(root/'serial/frozen.pt',map_location='cpu',weights_only=False)
  dataset,_,tag=PHASES[phase];evaluate_batch(e,args,out,artifact,dataset,{r['split'] for r in info['cases']},tag=tag)
  records=[r for r in history(root/'context_replay'/phase) if r.get('context_replay_job')==job.name]
  technical=sum(bool(r.get('technical_failure')) for r in records)
  write(status,{'status':'needs_attention' if technical else 'complete','pid':os.getpid(),'finished':time.time(),'cases':len(info['cases']),'records':len(records),'technical_failure_records':technical,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'capacity':info['capacity']})
 except Exception as ex:
  write(status,{'status':'needs_attention','pid':os.getpid(),'reason':str(ex),'traceback':traceback.format_exc(),'time':time.time()});raise

def coordinator(a):
 root=Path(a.run);base=root/'context_replay';base.mkdir(exist_ok=True);lock=(base/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 scheduler=base/'scheduler';scheduler.mkdir(exist_ok=True)
 while True:
  state=read(root/'serial_status.json') or {};phase=state.get('phase');summary={'time':time.time(),'pid':os.getpid(),'status':'watching','phases':{}}
  if phase not in PHASES:
   write(base/'status.json',summary)
   if phase=='complete':return
   time.sleep(30);continue
  failures=base_failures(root,phase);manifest=prepare_cases(root,phase,failures);done=completed(root,phase);active=set();jobs=[];blocked=[]
  for path in scheduler.glob(phase+'_*/job.json'):
   job=read(path);st=read(base/phase/('worker_'+path.parent.name)/'status.json') or {};running=alive(job.get('pid',0));jobs.append((path,job,st,running))
   if running:active.update(r['case_id'] for r in job['cases'])
   elif st.get('status')=='needs_attention':blocked.append(path.parent.name)
   elif st.get('status')!='complete' and any(r['case_id'] not in done for r in job['cases']):
    if job.get('recovery_attempts',0)>=2:blocked.append(path.parent.name)
    else:
     free=available_gpus(root)
     if not free:
      active.update(r['case_id'] for r in job['cases']);continue
     if job['gpu'] not in free:job['gpu']=free[0]
     env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(job['gpu'])
     with (path.parent/'worker.log').open('a') as log:
      proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'worker','--run',str(root),'--job',str(path.parent)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
     job.update(pid=proc.pid,recovery_attempts=job.get('recovery_attempts',0)+1);write(path,job);active.update(r['case_id'] for r in job['cases'])
  pending=set(manifest['cases'])-done-active
  unsupported={c for c in pending if manifest['cases'][c]['capacity']>131072}
  probe_complete=any(j.get('probe') and st.get('status')=='complete' for _,j,st,_ in jobs)
  probe_running=any(j.get('probe') and running for _,j,_,running in jobs)
  if not blocked and not unsupported:
   busy={j["gpu"] for _,j,_,running in jobs if alive(j.get("pid",0))}
   for gpu in [g for g in available_gpus(root) if g not in busy]:
    if not pending or (not probe_complete and probe_running):break
    ordered=sorted(pending,key=lambda c:manifest['cases'][c]['tokens'],reverse=True);capacity=manifest['cases'][ordered[0]]['capacity'];selected=[c for c in ordered if manifest['cases'][c]['capacity']==capacity][:4 if probe_complete else 1]
    job_dir=scheduler/(phase+'_'+str(time.time_ns()));job_dir.mkdir();dataset=PHASES[phase][0]
    with (job_dir/(dataset+'.jsonl')).open('w') as f:
     for cid in selected:f.write(json.dumps(read(base/phase/'cases'/cid/'case.json'),ensure_ascii=False)+'\n')
    job={'phase':phase,'gpu':gpu,'capacity':capacity,'factor':manifest['cases'][selected[0]]['factor'],'cases':[{'case_id':c,**manifest['cases'][c]} for c in selected],'probe':not probe_complete,'started':time.time()}
    write(job_dir/'job.json',job)
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
    with (job_dir/'worker.log').open('a') as log:
     proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'worker','--run',str(root),'--job',str(job_dir)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    job['pid']=proc.pid;write(job_dir/'job.json',job);active.update(selected);pending-=set(selected);probe_running=not probe_complete
    with (root/'pipeline_events.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),'event':'context_replay_started','phase':phase,'gpu':gpu,'cases':len(selected),'pid':proc.pid,'capacity':capacity})+'\n')
  for name in PHASES:
   m=read(base/name/'manifest.json') or {'cases':{}};finished=completed(root,name);summary['phases'][name]={'target_cases':len(m['cases']),'complete_cases':len(finished),'pending_cases':len(set(m['cases'])-finished),'original_records_preserved':sum(v['original_records'] for v in m['cases'].values())}
  policy=read(root/'resource_policy.json') or {};cohort=set(policy.get('priority_context_cases',[])) if policy.get('priority_phase')==phase else set()
  summary['priority_cohort']={'target_cases':len(cohort),'complete_cases':len(cohort & done),'remaining_cases':len(cohort-done),'allowed_gpus':policy.get('allowed_gpus',[0,1,5,6])}
  finished_count=0;gpu_seconds=0.
  for _,job,st,running in jobs:
   finished_count+=sum(r['case_id'] in done for r in job['cases'])
   gpu_seconds+=max(0.,(st.get('finished',time.time()) if not running else time.time())-job['started'])
  card_count=sum(alive(j.get('pid',0)) for _,j,_,_ in jobs)
  if finished_count and gpu_seconds:
   rate=finished_count/gpu_seconds*3600
   summary['known_queue_eta']={'completed_cases':finished_count,'cases_per_gpu_hour':round(rate,2),'remaining_cases':len(set(manifest['cases'])-done),'remaining_hours_at_current_cards':round((len(set(manifest['cases'])-done))/rate/max(1,card_count),2),'basis':'actual elapsed GPU-job time including model loading and in-flight cases; current discovered queue only; future overflow cases not included'}
  live=[{'gpu':j['gpu'],'pid':j['pid'],'cases':len(j['cases']),'capacity':j['capacity'],'elapsed_seconds':time.time()-j['started']} for _,j,st,running in jobs if running]
  summary.update(phase=phase,status='needs_attention' if blocked or unsupported else 'running' if active else 'idle',active_jobs=live,blocked_jobs=blocked,unsupported_cases=len(unsupported),probe_complete=probe_complete)
  write(base/'status.json',summary);time.sleep(30)

def barrier(root,phase):
 """No next clinical stage starts until this stage's replay denominator is closed."""
 root=Path(root)
 while True:
  failures=base_failures(root,phase);done=completed(root,phase);pending=set(failures)-done;status=read(root/'context_replay/status.json') or {}
  ending=[]
  for path in (root/'scheduler').glob(phase+'_*.json'):
   info=read(path) or {}
   if alive(info.get('pid',0)):ending.append(info['pid'])
  for path in (root/'context_replay/scheduler').glob(phase+'_*/job.json'):
   info=read(path) or {}
   if alive(info.get('pid',0)):ending.append(info['pid'])
  if not pending and not ending:break
  if status.get('status')=='needs_attention':raise RuntimeError('context replay requires engineering recovery; original records retained')
  if not alive(status.get('pid',0)):raise RuntimeError('context replay controller is not running')
  write(root/'serial_status.json',{'time':time.time(),'controller_pid':os.getpid(),'phase':phase,'status':'waiting_for_context_replay','remaining_context_cases':len(pending)})
  time.sleep(30)
 materialize(root,phase)

def materialize(root,phase):
 original={(r['case_id'],r['method']):r for r in history(root/'serial'/phase)};replayed=history(root/'context_replay'/phase)
 for r in replayed:
  key=(r['case_id'],r['method']);old=original.get(key)
  if not old or not old.get('technical_failure') or old.get('failure_reason')!='context_overflow':raise ValueError('replacement must supersede an original context_overflow cell')
  original[key]=r
 target=root/'analysis/serial'/phase/'worker_0';target.mkdir(parents=True,exist_ok=True);tag=PHASES[phase][2]
 with (target/(tag+'.jsonl')).open('w') as f:
  for r in original.values():f.write(json.dumps(r,ensure_ascii=False)+'\n')
 write(target.parent/'context_policy.json',{'original_records_retained':True,'replaced_context_failure_cells':len(replayed),'active_cells':len(original),'policy':'original successful cases plus explicit YaRN long-context replays; report runtime profiles separately'})

if __name__=='__main__':
 p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=['controller','worker']);p.add_argument('--run',required=True);p.add_argument('--job');a=p.parse_args()
 if a.mode=='worker':worker(a)
 else:coordinator(a)
