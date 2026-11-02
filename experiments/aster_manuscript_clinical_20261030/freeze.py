"""Freeze temperature and acceptance using disjoint original clinical cases."""
import argparse,json,subprocess
from pathlib import Path
from collections import Counter
from src.aster.calibration import fit_temperature
from src.aster.manuscript_core import plan_gains
from .data import read,write,REV,MODEL
from .engine import decisions_from
from .analyze import load,gold
def classification_records(run):
 rows=[]
 for shard in [0,1]:
  root=run/('classification'+str(shard));status=json.loads((root/'status.json').read_text());assert status['status']=='complete'
  rows+=read(root/'records.jsonl')
 return rows
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=Path(a.run);records=classification_records(run)
 f=json.loads((run/'analysis/freeze_pre_gain.json').read_text());cal=[];dev=[]
 for row in records:
  if row['status']!='complete':continue
  case=row['case'];truth={('G'+r[1:-1]):r for r in case['gold_rules']}
  if case['phase']=='calibration':
   for gid,label in truth.items():
    d=row['raw_decisions'][gid]
    if d is not None:cal.append({'logits':d['logits'],'label':label})
  if case['phase']=='development':dev.append(row)
 original_count=len(cal)
 for r in load(run):
  if r['row']['split']=='calibration':
   for state,d in r['classification'].items():
    if 'logits' in d:cal.append({'logits':d['logits'],'label':gold(r,state)})
 f['temperature']=fit_temperature(cal,split='calibration');choices=[]
 for prob in [0.,.7,.9]:
  for margin in [0.,.1]:
   f['thresholds']={'probability':prob,'margin':margin};accepted=[];root=[];failures=0
   for r in dev:
    c=r['case'];decision=decisions_from(r['raw_decisions'],f);plan=plan_gains(c['guideline'],decision,g_plus=1,g_minus=1)
    truth={('G'+rid[1:-1]):rid for rid in c['gold_rules']}
    root.append(plan['groups']['G0']['enabled'])
    for gid,state in plan['groups'].items():
     if state['enabled']:accepted.append(state['prediction']==truth.get(gid))
   acc=sum(accepted)/len(accepted) if accepted else 0.
   choices.append({'probability':prob,'margin':margin,'accepted_groups':len(accepted),'accepted_accuracy':acc,'root_coverage':sum(root)/len(root),'development_cases':len(dev)})
 chosen=max(choices,key=lambda x:(x['accepted_accuracy']>=.9,x['accepted_groups'] if x['accepted_accuracy']>=.9 else x['accepted_accuracy'],x['accepted_groups'],-x['probability'],-x['margin']))
 f.update(thresholds=chosen,threshold_candidates=choices,model_revision=REV,model_path=str(MODEL),max_new_tokens=512,
  clinical_calibration={'original_visited_group_labels':original_count,'independent_three_state_labels':len(cal)-original_count,'original_calibration_cases':sum(r['case']['phase']=='calibration' for r in records)},
  scope='clinical original-vignette evaluation; age-only rule-verified direction training; no physician gold claim',
  gain_selection='development final leaf-id accuracy, tie smaller gain',
  evaluation_metrics=['leaf_id_accuracy','path_rule_F1','exact_path','invalid_generation','technical_failure','official_Qwen_judged_PCC_secondary'],
  methods=['baseline','zero','spectral','mean','random','reverse','score_bias','enhance_only','suppress_only','direct_map','text_append','text_clip','native_official'],
  query_observation_offsets=[0,1,2,4,8,16],sampling='greedy',repeats=1,bootstrap_repeats=1000,
  code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
 write(run/'clinical_freeze_pre_gain.json',f);write(run/'clinical_calibration_audit.json',{'temperature':f['temperature'],'labels':f['clinical_calibration'],'chosen_thresholds':chosen,'all_thresholds':choices})
 print(json.dumps({'calibration_labels':len(cal),'temperature':f['temperature']['temperature'],'thresholds':chosen,'head_count':len(f['selected_heads'])}))
if __name__=='__main__':main()

