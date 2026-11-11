"""Main-window-only final merge; all nine one-round reviews must finish first."""
from runtime_paths import ROOT, DATA, RAW, REPO, SCRIPTS, SCHEMA, source_document_path
import argparse,collections,fcntl,json,pathlib,subprocess,sys
from parallel_generation import ROOT,BASE,rows,save,now,status
def write_once(path,data):
 content=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data)
 if path.exists():
  if path.read_text(encoding='utf-8')!=content:raise ValueError('existing merged artifact differs: '+str(path))
 else:path.write_text(content,encoding='utf-8')
def merge(finalize=False):
 plan=json.loads((BASE/'plan.json').read_text());generated=[];reviews=[]
 for job in plan['jobs']:
  p=BASE/job
  if not (p/'author_completed.json').exists() or not (p/'review_completed.json').exists():raise RuntimeError('unfinished job: '+job)
  gg=rows(p/'generated_candidates.jsonl');rr=rows(p/'review_results.jsonl')
  if len(rr)!=len(gg) or {r['pair_id'] for r in rr}!={r['pair_id'] for r in gg}:raise ValueError('review IDs mismatch: '+job)
  if any(r.get('review_model')!='gpt-6.1-sol' or r.get('reasoning_effort')!='xhigh' or r.get('round')!=1 for r in rr):raise ValueError('review protocol mismatch')
  generated+=gg;reviews+=rr
 if len({r['pair_id'] for r in generated})!=len(generated):raise ValueError('duplicate new pair IDs')
 history=rows(ROOT/'historical_pairs.jsonl');historical_review=rows(ROOT/'review_results_historical.jsonl')
 if len(history)!=len(historical_review):raise ValueError('historical review coverage mismatch')
 if {r['pair_id'] for r in history}&{r['pair_id'] for r in generated}:raise ValueError('historical/new pair ID collision')
 write_once(ROOT/'generated_candidates.jsonl',generated)
 write_once(ROOT/'review_results.jsonl',historical_review+reviews)
 write_once(ROOT/'sol_author_edits.jsonl',[r for job in plan['jobs'] for r in rows(BASE/job/'author_edits.jsonl')])
 save(ROOT/'reports/generated_merge_summary.json',dict(new_candidates=len(generated),reviewed=len(reviews),review_status=dict(collections.Counter(r['status'] for r in reviews)),all_candidates=len(history)+len(generated),all_reviewed=len(historical_review)+len(reviews),jobs=len(plan['jobs']),completed_at=now(),review_rounds=1))
 if finalize and not (ROOT/'reports/completion_all.json').exists():
  with (ROOT/'logs/final_dataset_validation.log').open('w') as log:
   subprocess.run([sys.executable,str(SCRIPTS/'validate.py'),'--stage','all','--finalize'],stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
 if finalize and (ROOT/'reports/completion_all.json').exists():
  from final_dataset_report import main as write_report
  write_report()
 return dict(new_candidates=len(generated),new_reviews=len(reviews),finalized=(ROOT/'reports/completion_all.json').exists())
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--finalize',action='store_true');args=a.parse_args()
 with (BASE/'merge.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  print(json.dumps(merge(args.finalize)))
