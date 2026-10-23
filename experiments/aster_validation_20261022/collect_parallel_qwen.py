"""Read-only ownership/progress snapshot; no inference or task control."""
from pathlib import Path
import json,datetime,shutil
r=Path('/home/gcx/aster_validation_20261022/private/run500/parallel_qwen');out=Path(__file__).resolve().parent/'results/run500/qwen/parallel';out.mkdir(parents=True,exist_ok=True);plan=json.loads((r/'plan.json').read_text());report={'time':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'protected_pid':plan['protected_running_pid'],'workers':{},'tasks':[]}
for g in [0,1]:
 p=r/f'gpu{g}.status';report['workers'][str(g)]=p.read_text().strip() if p.exists() else 'not_started'
for t in plan['tasks']:
 p=Path(t['output']);m=p.with_suffix('.manifest.json');done=0
 if p.exists():
  with p.open() as f:
   for line in f:
    try:json.loads(line);done+=1
    except json.JSONDecodeError:pass
 complete=False
 if m.exists():
  try:complete=json.loads(m.read_text()).get('complete',False)
  except json.JSONDecodeError:pass
 report['tasks'].append({'name':t['name'],'owner':t['owner'],'rows':done,'expected_rows':t['rows'],'complete':complete})
shutil.copyfile(r/'plan.json',out/'plan.json');(out/'STATUS.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
