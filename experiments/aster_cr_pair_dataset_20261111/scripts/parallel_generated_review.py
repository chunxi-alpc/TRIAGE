"""Blind one-round reviewer IO for independently assigned Sol/xhigh agents."""
from runtime_paths import ROOT, DATA, RAW, REPO, SCRIPTS, SCHEMA, source_document_path
import argparse,collections,json,pathlib,sys
from parallel_generation import ROOT,BASE,claim,jobpath,rows,save,now
CHECKS=['factor_matches_class','no_other_decisive_change','labels_correct','source_support','medical_plausibility','protected_content_preserved','no_hidden_second_factor']
def index(job):return json.loads((jobpath(job)/'blind_packets/index.json').read_text())
def load(job,i):return json.loads(pathlib.Path(index(job)[i]['path']).read_text())
def persist(job,owner,i,decisions):
 p=jobpath(job)
 if not (p/'author_completed.json').exists():raise RuntimeError('author job not frozen')
 claim(job,owner,'reviewer')
 if (p/'author_claim.json').exists() and json.loads((p/'author_claim.json').read_text())['owner']==owner:raise ValueError('author cannot review own job')
 if (p/'review_completed.json').exists():raise RuntimeError('review is frozen')
 pack=load(job,i);allowed={r['pair_id']:r for r in pack['pairs']};done={r['pair_id'] for r in rows(p/'review_results.jsonl')}
 output=[]
 for d in decisions:
  pid=d['pair_id'];source=allowed[pid]
  if pid in done:raise ValueError('pair already reviewed')
  if any(d['checks'].get(c) not in ['PASS','FAIL','UNCERTAIN'] for c in CHECKS):raise ValueError('all seven manual checks required')
  if any(type(d[k]) is not int and d[k] is not None for k in ['left','right']) or any(d[k] not in [0,1,None] for k in ['left','right']):raise ValueError('invalid independent labels')
  checks=dict(d['checks'],duplicate='DEFERRED_DETERMINISTIC')
  status='FAIL' if 'FAIL' in checks.values() else 'UNCERTAIN' if 'UNCERTAIN' in checks.values() or d['left'] is None or d['right'] is None else 'PASS'
  output.append(dict(pair_id=pid,kind=source['expected_kind'],status=status,checks=checks,independent_left_label=d['left'],independent_right_label=d['right'],reason=d['reason'],review_model='gpt-6.1-sol',reasoning_effort='xhigh',underlying_snapshot=None,round=1,reviewer=owner,reviewed_at=now(),source_id=pack['source_id'],job=job,blind_input='original source, case, condition, original label, actual edit, expected class only; no author reasoning'))
  done.add(pid)
 with (p/'review_results.jsonl').open('a',encoding='utf-8') as f:
  for r in output:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 result=dict(job=job,reviewed=len(done),total=sum(r['pairs'] for r in index(job)),counts=dict(collections.Counter(r['status'] for r in rows(p/'review_results.jsonl'))),updated_at=now())
 save(p/'review_progress.json',result);return result
def finish(job,owner):
 p=jobpath(job);claim(job,owner,'reviewer');rs=rows(p/'review_results.jsonl');expected={r['pair_id'] for entry in index(job) for r in json.loads(pathlib.Path(entry['path']).read_text())['pairs']}
 if len(rs)!=len(expected) or {r['pair_id'] for r in rs}!=expected:raise ValueError('unfinished review')
 result=dict(job=job,reviewer=owner,model='gpt-6.1-sol',reasoning_effort='xhigh',round=1,total=len(rs),counts=dict(collections.Counter(r['status'] for r in rs)),completed_at=now())
 save(p/'review_completed.json',result);return result
def main():
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['index','source','structure','cases','pairs','persist','finish']);a.add_argument('--job',required=True);a.add_argument('--owner');a.add_argument('--index',type=int,default=0);a.add_argument('--start',type=int,default=0);a.add_argument('--end',type=int);args=a.parse_args()
 if args.mode=='index':r=index(args.job)
 elif args.mode=='persist':r=persist(args.job,args.owner,args.index,json.load(sys.stdin))
 elif args.mode=='finish':r=finish(args.job,args.owner)
 else:
  pack=load(args.job,args.index)
  with (jobpath(args.job)/'review_read_operations.jsonl').open('a') as f:f.write(json.dumps(dict(index=args.index,part=args.mode,start=args.start,end=args.end,time=now()))+'\n')
  if args.mode=='source':
   s='\n'.join(doc+'\n'+text for doc,text in pack['full_source_documents'].items());r=s[args.start:args.end]
  elif args.mode=='structure':r=pack['original_guidelines']
  elif args.mode=='cases':r=pack['cases']
  else:
   r=[]
   for pair in pack['pairs'][args.start:args.end]:
    n=pack['nodes'][pair['group_id']]
    r.append(dict(pair_id=pair['pair_id'],expected_kind=pair['expected_kind'],is_artificial_counterfactual=pair['is_artificial_counterfactual'],**n,actual_modification=pack['contrasts'][pair['contrast']]))
 print(r if isinstance(r,str) else json.dumps(r,ensure_ascii=False))
if __name__=='__main__':main()
