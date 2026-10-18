"""Record immutable pre-run code/config/data fingerprints, without model calls."""
from __future__ import annotations
import argparse,datetime,hashlib,json,pathlib,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[3]
EXP=ROOT/'experiments/care_module1_v5'

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,default=EXP/'data/RUN_FREEZE.json');a=p.parse_args()
 paths=[ROOT/'src/module1/care_v5.py',ROOT/'src/module1/compat_v5.py',ROOT/'src/module1/__init__.py',EXP/'config.json',
 EXP/'scripts/run_experiment.py',EXP/'data/labels_public.jsonl',EXP/'data/pairs_public.jsonl',EXP/'data/rules_v5.jsonl',
 EXP/'data/v4_qwen_direct_public.jsonl',EXP/'exploratory/data/FREEZE.json',EXP/'exploratory/data/inputs_public.jsonl',
 EXP/'exploratory/data/labels_public.jsonl',EXP/'exploratory/data/private/rules.jsonl']
 result={'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'git_base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
 'scope':'No scored model predictions read before freeze; unscored age-only interface check used for format compatibility.',
 'models':json.loads((EXP/'provenance/official_interface_sources.json').read_text()),'files':{}}
 for path in paths:
  b=path.read_bytes();result['files'][str(path.relative_to(ROOT))]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
 if a.output.exists():raise FileExistsError(f'Run freeze already exists: {a.output}')
 a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(str(a.output))
if __name__=='__main__':main()
