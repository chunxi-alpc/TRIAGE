
from pathlib import Path
import json,time,argparse
import numpy as np
B=Path('/home/gcx/aster_next_20261109');O=B/'diagnosis';R=B/'condition_only_results'
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()] if Path(p).exists() else []
def run(wait):
 if wait:
  while not (R/'dev_short.done.json').exists():
   if (R/'failed.json').exists():raise RuntimeError('condition-only worker failed')
   time.sleep(5)
 if not (R/'dev_short.done.json').exists():raise RuntimeError('short scoring not complete')
 rr={r['id']:r for r in rows(R/'dev_short.jsonl') if r.get('status')=='ok'}
 short=rows(O/'condition_only_v1.dev_short.jsonl');assert len(short)==218 and all(r['id'] in rr for r in short)
 old=np.load('/home/gcx/aster_cancel_story_20261109/meeting/real_completion/coordinates.npz');ix={v:i for i,v in enumerate(old['ids'].tolist())}
 raw={v:float(old['ell'][ix[v]]) for v in ix if not v.startswith('path|')}
 for p in Path('/home/gcx/aster_cr_repair_v2_run_20261108/qwen/paths').glob('features_dev_*.jsonl'):
  for r in rows(p):
   if r.get('status')=='ok':raw['path|'+r['id']]=float(r['ell'])
 summary={}
 for tag in ['dev_natural','dev_focus']:
  mm=rows(O/('condition_only_v1.'+tag+'.jsonl'))
  subset={r['id']:r for r in short}
  def find(r):
   if r['id'] in rr:return rr[r['id']]
   for k,v in subset.items():
    if r['id'] in v.get('aliases',[]):return rr[k]
   raise KeyError(r['id'])
  score=np.array([find(r)['ell'] for r in mm]);label=np.array([r['label'] for r in mm]);co=(score>0)==label
  rawco=np.array([int(raw[r['id']]>0)==r['label'] for r in mm])
  summary[tag]=dict(n=len(mm),raw_correct=int(rawco.sum()),candidate_correct=int(co.sum()),
   fixed=[r['id'] for j,r in enumerate(mm) if co[j] and not rawco[j]],
   broken=[r['id'] for j,r in enumerate(mm) if not co[j] and rawco[j]])
 result=dict(method='condition_only_v1',scope='quick development diagnostic only; not a path result',
  summary=summary,candidate_count_this_family=1,b=0.,old_or_official_test_used=False)
 (O/'condition_only_quick_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(result))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');run(p.parse_args().wait)

