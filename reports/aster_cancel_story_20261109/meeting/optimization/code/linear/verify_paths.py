
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
from pathlib import Path
import sys,json,numpy as np
o=Path(__file__).resolve().parent
sys.path.insert(0,'/home/gcx/TRIAGE-worktrees/aster-cr-20261108')
from experiments.aster_cr.paths import execute
sel=json.loads((o/'path_calibrated_selection.json').read_text())
allr={}
for tier,val in sel['tiers'].items():
 for kind,rr in val.items():
  for r in rr:allr[r['config']['name']]=r
p=dict(np.load(o/'dev_path_features.npz'))
meta=json.loads((o/'dev_path_metadata.json').read_text());rs=meta['records']
cases=[json.loads(s) for s in Path('/home/gcx/aster_cr_repair_v2_run_20261108/data/path_cases.jsonl').read_text().splitlines() if s.strip() and json.loads(s)['split']=='dev']
group={}
for i,r in enumerate(rs):group.setdefault(r['case_id'],{})[r['gid']]=i
result={};allrows=[]
for name,r in allr.items():
 c=r['config'];model=np.load(c['model_file'])
 z=(p[c['family']]-model['mu'])/model['scale'];ss=p['ell']+(z@model['w'])*c['gain']+c['new_b']
 rr=[dict(method=name,case_id=ca['case_id'],source_id=ca['source_id'],**execute(ca,{gid:float(ss[i]) for gid,i in group[ca['case_id']].items()})) for ca in cases]
 result[name]=dict(interval_correct=r['metrics']['dev']['path']['correct'],execute_correct=sum(x['correct'] for x in rr))
 allrows.extend(rr)
with (o/'path_calibrated_dev_results.jsonl').open('w') as f:
 for r in allrows:f.write(json.dumps(r)+'\n')
with (o/'path_calibrated_verification.json').open('w') as f:json.dump(result,f,indent=2)
assert all(r['interval_correct']==r['execute_correct'] for r in result.values()),result
print(json.dumps(result))

