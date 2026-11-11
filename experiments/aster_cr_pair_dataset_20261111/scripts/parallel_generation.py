"""Parallel fit-only authors, explicit edit IO and one-time blind handoff; no model API."""
from runtime_paths import ROOT, DATA, RAW, REPO, SCRIPTS, SCHEMA, source_document_path
import argparse,collections,datetime,fcntl,json,pathlib,sys,os
from agent_a_sol_prepare import ROOT,FitCatalog,read_rows
from normalize_sol_candidates import normalize_one,normalize_file
BASE=ROOT/'generation_parallel'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.'+str(os.getpid())+'.tmp');q.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');q.replace(p)
def rows(p):return read_rows(p) if p.exists() else []
def jobpath(job):
 if job not in json.loads((BASE/'plan.json').read_text())['jobs']:raise ValueError('unknown job')
 return BASE/job
def claim(job,owner,role='author'):
 p=jobpath(job);lock=p/(role+'_claim.json')
 if lock.exists():
  old=json.loads(lock.read_text())
  if old['owner']!=owner:raise RuntimeError('already claimed by '+old['owner'])
 else:
  with lock.open('x') as f:json.dump(dict(owner=owner,role=role,claimed_at=now()),f)
 return dict(job=job,owner=owner,role=role)
def initialize():
 if (BASE/'plan.json').exists():return json.loads((BASE/'plan.json').read_text())
 tasks=read_rows(ROOT/'sol_generation_tasks.jsonl');by=collections.defaultdict(list)
 for t in tasks:by[t['source_id']].append(t)
 load=[0,0,0];sources=[[],[],[]]
 for sid,tt in sorted(by.items(),key=lambda x:(-len(x[1]),x[0])):
  w=min(range(3),key=lambda w:(load[w],w));sources[w].append(sid);load[w]+=len(tt)
 jobs={}
 for w in range(3):
  for kind in ['P','C','F']:
   job='w%d_%s'%(w,kind);p=BASE/job;p.mkdir(parents=True,exist_ok=True)
   selected=[dict(t,task_id=job+'_'+str(i)) for i,t in enumerate(tasks) if t['kind']==kind and t['source_id'] in sources[w]]
   with (p/'tasks.jsonl').open('x') as f:
    for t in selected:f.write(json.dumps(t,ensure_ascii=False)+'\n')
   info=dict(job=job,window=w,kind=kind,tasks=len(selected),sources=sorted(sources[w]),author_model='gpt-6.1-sol',author_effort='high',review_model='gpt-6.1-sol',review_effort='xhigh')
   save(p/'assignment.json',info);jobs[job]=info
 plan=dict(created_at=now(),tasks=len(tasks),windows=3,workers_per_window=3,jobs=jobs,one_generation_round=True,one_review_round=True,automatic_semantic_retry=False)
 save(BASE/'plan.json',plan);return plan
def packet(job,sid):
 p=jobpath(job);tasks=[t for t in read_rows(p/'tasks.jsonl') if t['source_id']==sid]
 if not tasks:raise ValueError('source outside assigned job')
 catalog=FitCatalog();documents={};guides={};cases={};items=[]
 done={r['candidate_id'] for r in rows(p/'author_edits.jsonl')}
 for t in tasks:
  meta=catalog.metadata(t['case_id'],t['node_id']);doc=meta['source_document'];gid=str(meta['guideline_id'])
  documents.setdefault(doc,source_document_path(doc).read_text(encoding='utf-8'))
  side=catalog.original(t['case_id'],t['node_id'])
  guides.setdefault(gid,dict(guide=side['guide'],definitions=side['definitions']))
  c=catalog.cases[t['case_id']]
  cases.setdefault(t['case_id'],dict(patient=side['patient'],gold_targets=[dict(node_id=g['gid'],label=g['label'],condition=catalog.original(t['case_id'],g['gid'])['condition']) for g in c['gold_path']]))
  items.append(dict(t,completed=t['task_id'] in done,program_rule=catalog.rules.get((t['case_id'],t['node_id']))))
 return dict(job=job,source_id=sid,documents=documents,guidelines=guides,cases=cases,tasks=items)
def persist(job,owner,authored):
 p=jobpath(job);claim(job,owner)
 if (p/'author_completed.json').exists():raise RuntimeError('author job is frozen')
 allowed={t['task_id']:t for t in read_rows(p/'tasks.jsonl')};done={r['candidate_id'] for r in rows(p/'author_edits.jsonl')}
 catalog=FitCatalog()
 from jsonschema import Draft202012Validator
 schema=Draft202012Validator(json.loads((SCHEMA).read_text()))
 for a in authored:
  task=allowed[a['candidate_id']]
  if a['candidate_id'] in done:raise ValueError('already authored: '+a['candidate_id'])
  if a['kind']!=task['kind'] or a['case_id']!=task['case_id']:raise ValueError('cross-job/task write')
  if a.get('available') is not False:
   schema.validate(normalize_one(a,catalog,p/'author_edits.jsonl',len(done)))
  a['author_agent']=owner;a['authored_at']=now()
  done.add(a['candidate_id'])
 with (p/'author_edits.jsonl').open('a',encoding='utf-8') as f:
  for a in authored:f.write(json.dumps(a,ensure_ascii=False)+'\n')
 data=rows(p/'author_edits.jsonl')
 result=dict(job=job,authored=len(data),total=len(allowed),available=sum(r.get('available') is not False for r in data),updated_at=now())
 save(p/'author_progress.json',result);return result
def finish(job,owner):
 p=jobpath(job);claim(job,owner)
 allowed={t['task_id'] for t in read_rows(p/'tasks.jsonl')};data=rows(p/'author_edits.jsonl')
 if {a['candidate_id'] for a in data}!=allowed or len(data)!=len(allowed):raise ValueError('unfinished author tasks')
 out=p/'generated_candidates.jsonl';ex=p/'conversion_exclusions.jsonl'
 if not out.exists():normalize_file(p/'author_edits.jsonl',out,ex)
 import export_blind_packets
 export_blind_packets.export('generated',file=out,folder=p/'blind_packets')
 result=dict(job=job,author=owner,model='gpt-6.1-sol',reasoning_effort='high',tasks=len(data),normalized=len(rows(out)),excluded=len(rows(ex)),completed_at=now())
 save(p/'author_completed.json',result);return result
def status():
 plan=json.loads((BASE/'plan.json').read_text());jobs={}
 for j in plan['jobs']:
  p=jobpath(j);a=rows(p/'author_edits.jsonl');b=rows(p/'review_results.jsonl')
  jobs[j]=dict(tasks=plan['jobs'][j]['tasks'],authored=len(a),available=sum(r.get('available') is not False for r in a),normalized=json.loads((p/'author_completed.json').read_text())['normalized'] if (p/'author_completed.json').exists() else 0,reviewed=len(b),review_counts=dict(collections.Counter(r['status'] for r in b)),author_complete=(p/'author_completed.json').exists(),review_complete=(p/'review_completed.json').exists())
 result=dict(updated_at=now(),planned_tasks=plan['tasks'],authored=sum(j['authored'] for j in jobs.values()),reviewed=sum(j['reviewed'] for j in jobs.values()),jobs=jobs)
 save(BASE/'progress.json',result);return result
def main():
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['init','claim','packet','persist','finish','status']);a.add_argument('--job');a.add_argument('--owner');a.add_argument('--source');a.add_argument('--part',choices=['all','source','cases','structure','tasks'],default='all');a.add_argument('--start',type=int,default=0);a.add_argument('--end',type=int);args=a.parse_args()
 if args.mode=='init':r=initialize()
 elif args.mode=='claim':r=claim(args.job,args.owner)
 elif args.mode=='status':r=status()
 elif args.mode=='persist':r=persist(args.job,args.owner,json.load(sys.stdin))
 elif args.mode=='finish':r=finish(args.job,args.owner)
 else:
  r=packet(args.job,args.source)
  p=jobpath(args.job)
  with (p/'read_operations.jsonl').open('a') as f:f.write(json.dumps(dict(source_id=args.source,part=args.part,start=args.start,end=args.end,time=now()))+'\n')
  if args.part=='source':
   s='\n'.join(doc+'\n'+s for doc,s in r['documents'].items());r=s[args.start:args.end]
  elif args.part=='cases':r=r['cases']
  elif args.part=='structure':r=r['guidelines']
  elif args.part=='tasks':r=r['tasks'][args.start:args.end]
 print(r if isinstance(r,str) else json.dumps(r,ensure_ascii=False))
if __name__=='__main__':main()
