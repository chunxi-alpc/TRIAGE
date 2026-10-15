#!/usr/bin/env python3
"""CPU replay of every delivered decision target and offline metric."""
import gzip,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from transformers import AutoTokenizer
from src.acre_pilot.span_mapping import build_prompt,map_annotations,select_targets
from src.acre_pilot.evaluator import evaluate_cases
BASE=ROOT/'results/acre_pilot/round2';OUT=BASE/'revision3'
items={x['case_id']:x for x in json.loads((BASE/'development_manifest.json').read_text())['samples']}
with gzip.open(OUT/'case_results.jsonl.gz','rt') as f:cases=[json.loads(x) for x in f]
tokenizer=AutoTokenizer.from_pretrained('/home/gcx/TRIAGE/models/modelscope/Qwen3-8B',local_files_only=True)
for c in cases:
 item=items[c['case_id']];prompt,_,mapping=build_prompt(tokenizer,item)
 assert hashlib.sha256(prompt.encode()).hexdigest()==c['base_prompt_sha256']
 mapped=map_annotations(item,c['annotations'],mapping)
 for method in ['B1','B2','B3']:
  targets,_=select_targets(mapped,mapping,method)
  if c['out_of_single_rule_scope']:targets=[]
  for beta in [0.,.25,.5,1.,2.]:
   v=c['variants'][method+'@'+str(beta)]
   assert targets==v['attention']['targets'],(c['case_id'],method,beta)
 targets,_=select_targets(map_annotations(item,c['oracle_annotations'],mapping),mapping,'B3')
 for beta in [0.,.25,.5,1.,2.]:assert targets==c['variants']['B6@'+str(beta)]['attention']['targets']
metrics,ann,details=evaluate_cases(cases,json.loads((BASE/'development_references.json').read_text()))
assert metrics==json.loads((OUT/'metrics.json').read_text())['all_grid_metrics']
check={'all_final_targets_recomputed':True,'all_final_metrics_recomputed':True,'case_count':len(cases),'device':'CPU','normal_references_not_used_for_target_replay':True,'reference_B6_replay_separate':True}
(OUT/'recompute_check.json').write_text(json.dumps(check,indent=2)+'\n')
print(json.dumps(check))
