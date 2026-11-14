import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--action',required=True)
p.add_argument('--seed',type=int,default=11)
p.add_argument('--split',default='dev')
a=p.parse_args()
root=Path('/home/gcx/aster-e5-gaps-20261112')
out=root/'reports/aster_e5_gaps_20261112/native_supervised_e1/lofit'
selection=json.loads((out/'selection.json').read_text())
assert selection['frozen']
cmd=[sys.executable,str(root/'experiments/aster_e5_gaps_20261112/native_supervised_baselines.py'),'--kind','lofit','--action',a.action,'--lr',str(selection['lr']),'--seed',str(a.seed),'--split',a.split,'--device','cuda:0']
raise SystemExit(subprocess.call(cmd,cwd=root))
