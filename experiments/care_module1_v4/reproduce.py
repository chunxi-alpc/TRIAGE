"""Rebuild exact inputs from authorized local source; recompute cached metrics.
Use --run only on the server after reporting GPU resources to the user.
"""
import argparse,hashlib,json,pathlib,shutil,subprocess,sys,tempfile
from build_dataset import build
def main(a):
 root=pathlib.Path('.');freeze=json.loads((root/'FREEZE.json').read_text())['sha256'];checks={}
 with tempfile.TemporaryDirectory(prefix='care_v4_rebuild_') as tmp:
  out=pathlib.Path(tmp)/'data'
  if a.source:
   build(a.source,out,root/'prerequisites')
   rules=[json.loads(s) for s in (out/'private/rules.jsonl').read_text().splitlines()]
   (out/'private/rules_inputs.jsonl').write_text(''.join(json.dumps({k:r[k] for k in ['rule_id','text']})+'\n' for r in rules))
   for name,h in freeze.items():
    if name.startswith('data/'):
     p=out/name[len('data/'):];checks[name]=hashlib.sha256(p.read_bytes()).hexdigest()==h
   assert all(checks.values()),checks
   if not (root/'data/private').exists():shutil.copytree(out/'private',root/'data/private')
  else:shutil.copytree(root/'data',out)
  if not (out/'model').exists():shutil.copytree(root/'data/model',out/'model')
  subprocess.run([sys.executable,'evaluate.py','--data',str(out)],check=True,stdout=subprocess.DEVNULL)
  checks['cached_metrics_exact']=json.loads((out/'metrics.json').read_text())==json.loads((root/'data/metrics.json').read_text())
  assert checks['cached_metrics_exact']
  if a.run:
   assert a.source and a.model,'--run needs --source and --model'
   # Restore the private input files required by the frozen runner, never overwrite results.
   if not (root/'data/private').exists():shutil.copytree(out/'private',root/'data/private')
   fresh=root/'replication_data';assert not fresh.exists()
   shutil.copytree(out,fresh,ignore=shutil.ignore_patterns('model','metrics.json','errors.jsonl','comparison.csv'))
   subprocess.run([sys.executable,'run_experiment.py','--data',str(fresh),'--model',a.model,'--batch','4'],check=True)
   subprocess.run([sys.executable,'evaluate.py','--data',str(fresh)],check=True)
 (root/'reproduction_check.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=pathlib.Path);p.add_argument('--model');p.add_argument('--run',action='store_true');main(p.parse_args())
