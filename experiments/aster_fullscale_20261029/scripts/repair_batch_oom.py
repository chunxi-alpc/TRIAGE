"""Archive transport-OOM attempts and schedule only their missing valid completions.
Never retry model refusals, output-format errors or context-overflow outcomes.
"""
import argparse,json,time,shutil
from pathlib import Path
p=argparse.ArgumentParser(__doc__);p.add_argument('--run',required=True);a=p.parse_args();root=Path(a.run);phase=root/'serial/confirm';cache=phase/'batch_cache';archive=root/'repairs'/('oom_'+str(time.time_ns()));archive.mkdir(parents=True)
def oom(value):return 'out of memory' in str(value).lower()
def load(path):
 try:return json.loads(path.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return None
def preserve(path):
 target=archive/path.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(path),str(target))
reroute=set();retry=set();counts={'failed_generated_cells':0,'failed_draws':0,'failed_score_aggregates':0,'retained_completed_rows':0,'archived_retry_rows':0}
for path in cache.glob('*/routing.json'):
 r=load(path)
 if r and oom(r.get('failure')):reroute.add(path.parent.name);preserve(path)
for path in list(cache.glob('*/generated/*.json')):
 r=load(path)
 if not r:continue
 case=path.parent.parent.name;method=path.stem
 if (r.get('technical_failure') and oom(r.get('failure_reason'))) or (case in reroute and method in {'learned','direct','mean','random','reverse'} and not r.get('technical_failure')):
  retry.add((case,method));preserve(path);counts['failed_generated_cells']+=1
for path in list(cache.glob('*/judge/*/*.json')):
 r=load(path)
 if r and r.get('technical_failure') and oom(r.get('response')):preserve(path);counts['failed_draws']+=1
for path in list(cache.glob('*/scores/*.json')):
 r=load(path)
 if not r:continue
 metrics=r['score'][2]
 if oom(metrics.get('official_pathway_judge',{}).get('reason')) or oom(metrics.get('official_accuracy_judge',{}).get('reason')):
  retry.add((path.parent.parent.name,path.stem));preserve(path);counts['failed_score_aggregates']+=1
for path in list(phase.glob('worker_*/*.jsonl')):
 lines=path.read_text().splitlines(keepends=True);kept=[];changed=False
 for line in lines:
  try:r=json.loads(line)
  except json.JSONDecodeError:changed=True;continue
  key=(r['case_id'],r['method']);dependent=r['case_id'] in reroute and r['method'] in {'learned','direct','mean','random','reverse'} and not r.get('technical_failure')
  bad=(r.get('technical_failure') and oom(r.get('failure_reason'))) or oom(r.get('official_pathway_judge',{}).get('reason')) or oom(r.get('official_accuracy_judge',{}).get('reason'))
  if key in retry or dependent or bad:retry.add(key);counts['archived_retry_rows']+=1;changed=True
  else:kept.append(line);counts['retained_completed_rows']+=1
 if changed:preserve(path);path.write_text(''.join(kept))
for case,method in retry:
 if case not in reroute:continue
 path=cache/case/'scores'/(method+'.json')
 if path.exists():preserve(path)
 folder=cache/case/'judge'/method
 if folder.exists():
  for path in list(folder.glob('*.json')):preserve(path)
summary={'status':'oom_attempts_archived_for_retry','archive':str(archive),'rerouted_cases':len(reroute),'retry_cells':len(retry),'counts':counts,'non_oom_failures_retained':True}
(archive/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(root/'oom_repair_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
