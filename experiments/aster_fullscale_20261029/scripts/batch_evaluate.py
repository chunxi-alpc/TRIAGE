"""Batched clinical inference and judging, preserving historical prediction cells."""
import json,re,time
from pathlib import Path
from fullscale_run import rows,METHODS,score
from batch_judge import BatchJudge
METRICS={'official_pcc','official_accuracy','official_failure','official_pathway_judge','official_accuracy_judge','official_judge_failed','parsed_native_output','mcqa_prediction','predicted_path','terminal_correct','path_prefix_fraction','judge_draws_requested','raw_path_judge_responses','raw_accuracy_judge_responses'}
def read(path):
 try:return json.loads(path.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return None
def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False)+'\n');temp.replace(path)
def history(phase):
 records=[]
 for path in phase.glob('worker_*/*.jsonl'):
  for line in path.open():
   try:records.append(json.loads(line))
   except json.JSONDecodeError:continue
 return records

def evaluate_batch(e,args,out,artifact,dataset,splits,limit=None,methods=METHODS,tag='evaluation'):
 phase=out.parent;old=history(phase);done={(r['case_id'],r['method']) for r in old};bycase={}
 for r in old:bycase.setdefault(r['case_id'],[]).append(r)
 path=out/(tag+'.jsonl');start=time.time();completed_cases=0;written=0
 # Preserve an interrupted trailing write separately before appending resumed cells.
 if path.exists():
  data=path.read_bytes();lines=data.splitlines(keepends=True)
  if lines:
   try:
    json.loads(lines[-1])
    if not lines[-1].endswith(b'\n'):path.write_bytes(data+b'\n')
   except json.JSONDecodeError:
    saved=path.with_name(path.name+'.partial_'+str(time.time_ns()));saved.write_bytes(lines[-1]);path.write_bytes(b''.join(lines[:-1]))
 judge=BatchJudge(e) if dataset=='mega' else None
 source=(c for i,c in enumerate(rows(args.run,dataset,set(splits))) if i%args.world==args.worker and not all((c['case_id'],m) in done for m in methods))
 with path.open('a') as log:
  while True:
   cases=[]
   for _ in range(args.generation_batch_size):
    try:cases.append(next(source))
    except StopIteration:break
   if not cases:break
   if limit:cases=cases[:max(0,limit-completed_cases)]
   if not cases:break
   batch_started=time.time();states=[];route_pending=[];route_indices=[]
   if hasattr(e,'prefix_cache'):e.prefix_cache.clear()
   for c in cases:
    cache=phase/'batch_cache'/c['case_id'];prior=bycase.get(c['case_id'],[]);route=read(cache/'routing.json');fatal=None
    if c.get('invalid_source'):fatal='invalid_source:'+c['invalid_source']
    if route is None and prior and 'out of memory' not in str(prior[0].get('route_failure','')).lower():
     raw=prior[0].get('raw_routing','');failure=prior[0].get('route_failure');rmap={r['id']:0 for r in c['rules']}
     try:
      obj=json.loads(re.search(r'\{.*\}',raw,re.S).group());rmap={rid:{'T':1,'F':-1,'U':0}.get(str(obj.get(rid,'U')).upper(),0) for rid in rmap}
     except Exception:failure=failure or 'applicability_parse_failure'
     route={'route':rmap,'raw':raw,'failure':failure};write(cache/'routing.json',route)
    st={'case':c,'cache':cache,'prior':prior,'route':route,'fatal':fatal,'generated':{},'scores':{}}
    states.append(st)
    if route is None and not fatal:route_indices.append(len(states)-1);route_pending.append(c)
   if route_pending:
    for i,(rmap,raw,failure) in zip(route_indices,e.applicability_many(route_pending,args.generation_batch_size)):
     st=states[i];st['route']={'route':rmap,'raw':raw,'failure':failure};write(st['cache']/'routing.json',st['route'])
   for st in states:
    c=st['case'];st['route']=st['route'] or {'route':{},'raw':'','failure':st['fatal']}
    try:
     if st['fatal']:raise ValueError(st['fatal'])
     st['text'],st['masks']=e.task(c)
    except Exception as ex:st['fatal']=str(ex)
    for previous in st['prior']:
     if previous.get('raw_output') and not previous['technical_failure']:
      st['scores'][previous['raw_output']]=[previous['correct'],previous['invalid_output'],{k:v for k,v in previous.items() if k in METRICS}]
   for method in methods:
    pending=[];texts=[];configs=[]
    for st in states:
     c=st['case']
     if (c['case_id'],method) in done:continue
     saved=read(st['cache']/'generated'/(method+'.json'))
     if saved is not None:st['generated'][method]=saved;continue
     route=st['route'];record={'case_id':c['case_id'],'guideline_id':c['group'],'dataset':dataset,'split':c['split'],'method':method,'correct':False,'technical_failure':False,'invalid_output':False,'intervention_applied':False,'route_failure':route['failure'],'raw_routing':route['raw'],'raw_output':'','batched':True}
     if st['fatal']:
      record.update(technical_failure=True,failure_reason=st['fatal']);st['generated'][method]=record;write(st['cache']/'generated'/(method+'.json'),record);continue
     selection=dict(artifact['selection'])
     if method=='reverse':selection['strength']=-selection['strength']
     routing=c['gold_route'] if method=='oracle' else route['route']
     config=e.config(selection,method,st['masks'],routing,artifact['directions'])
     record['intervention_applied']=method not in ('baseline','zero') and any(routing.values())
     pending.append((st,record));texts.append(st['text']);configs.append(config)
    def persist_generation(i,raw):
     st,record=pending[i]
     if isinstance(raw,Exception):record.update(technical_failure=True,failure_reason=str(raw))
     else:record['raw_output']=raw
     st['generated'][method]=record;write(st['cache']/'generated'/(method+'.json'),record)
    if pending:e.generate_many(texts,2048 if dataset=='mega' else 8,configs=configs,batch_size=args.generation_batch_size,on_result=persist_generation)
   requests=[];locations=[];draw_dirs=[]
   for st in states:
    unique={}
    for method,record in st['generated'].items():
     if record['technical_failure']:continue
     unique.setdefault(record['raw_output'],method)
    for raw,method in unique.items():
     if raw in st['scores']:continue
     cached=read(st['cache']/'scores'/(method+'.json'))
     if cached and cached['raw_output']==raw:st['scores'][raw]=cached['score'];continue
     requests.append((st['case'],raw));locations.append((st,method,raw));draw_dirs.append(st['cache']/'judge'/method)
   if requests:
    if hasattr(e,'prefix_cache'):e.prefix_cache.clear()
    outcomes=judge.score_many(requests,args.judge_batch_size,draw_dirs) if judge else [score(c,raw) for c,raw in requests]
    for (st,method,raw),value in zip(locations,outcomes):
     st['scores'][raw]=value;write(st['cache']/'scores'/(method+'.json'),{'raw_output':raw,'score':value})
   for st in states:
    for method in methods:
     if method not in st['generated']:continue
     record=dict(st['generated'][method])
     if not record['technical_failure']:
      record['correct'],record['invalid_output'],metrics=st['scores'][record['raw_output']];record.update(metrics)
     record['elapsed_case_seconds']=time.time()-batch_started;record['case_batch_size']=len(states)
     log.write(json.dumps(record,ensure_ascii=False)+'\n');log.flush();done.add((record['case_id'],method));written+=1
    completed_cases+=1
   write(out/(tag+'_progress.json'),{'completed_cases_this_invocation':completed_cases,'written_records_this_invocation':written,'wall_seconds':time.time()-start,'case_batch_size':args.generation_batch_size,'judge_batch_size':args.judge_batch_size,'batch_transport':e.batch_stats})
   if limit and completed_cases>=limit:break
 return path
