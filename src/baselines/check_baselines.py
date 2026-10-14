#!/usr/bin/env python3
"""Check original baseline imports only; never construct weights or datasets."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser()
parser.add_argument('--name',choices=['ProbeRAG','SEKA','JUICE','TrialGPT','all'],default='all')
args=parser.parse_args()
config=json.loads((ROOT/'configs/baselines.json').read_text())
status_path=ROOT/'results/baselines_status.json'
status=json.loads(status_path.read_text()) if status_path.exists() else {}
codes={
 'TrialGPT': "import runpy; m=runpy.run_path('/home/gcx/TRIAGE/src/module1/trialgpt_matching.py'); f,h=m['official_prompt_functions'](); s=m['smoke_input'](f); ss,method=m['sentences_from_patient'](s['patient']); f['get_matching_prompt'](s['trial'],'inclusion','\\\\n'.join(f'{i}. {x}' for i,x in enumerate(ss))); print('official TrialGPT prompts and existing sample read ok')",
 'ProbeRAG': 'import modules, dataset_wrapper; from scripts.evaluate_qa import QAEvaluator; print("original evaluation import ok")',
 'SEKA': 'from src.model import SEKALLM, AdaptiveSEKALLM; import benchmarks.eval_fact_gen; print("original core and benchmark imports ok")',
 'JUICE': 'import main_intervene; from dataset import FactualDataset; from pathlib import Path; d=FactualDataset(Path("dataset/factual/world_capital_gemma/head_size_4/unfiltered_test.json")); x=d[0]; assert "Clean Prompt" in x and "Answer" in x; print("original entrypoint import and existing sample read ok")'
}
for name in config:
 if args.name!='all' and name!=args.name:continue
 item=dict(config[name]); repo=Path(item['path'])
 env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=''
 env['PYTHONPATH']=os.pathsep.join([str(repo),str(repo/'src')])
 try:
  run=subprocess.run([sys.executable,'-c',codes[name]],cwd=repo,env=env,capture_output=True,text=True,timeout=120)
  (ROOT/'results'/f'baseline_{name}.log').write_text(run.stdout+run.stderr)
  item['import_test']='passed' if run.returncode==0 else 'failed'
  item['error_summary']='' if run.returncode==0 else (run.stderr.strip().splitlines()[-1] if run.stderr.strip() else f'exit {run.returncode}')
 except subprocess.TimeoutExpired:
  item['import_test']='timeout';item['error_summary']='导入检查超过120秒'
 status[name]=item
status_path.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({name:{'deployed':x['deployed'],'import_test':x.get('import_test'),'full_method_ready':x['full_method_ready']} for name,x in status.items()},ensure_ascii=False))
