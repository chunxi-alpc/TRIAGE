#!/usr/bin/env python3
"""Audit frozen single-factor edits before generating any inference manifests."""
import sys,json,hashlib,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.acre_pilot.pair_verification import verify_pair
from src.acre_pilot.span_mapping import options_for,build_prompt
from src.acre_pilot.constrained_annotation import payload_for
from transformers import AutoTokenizer
OUT=ROOT/'results/acre_pilot/round2'
cfg=json.loads((ROOT/'configs/datasets.json').read_text())['MEGA-CDP'];D=Path(cfg['path'])
def load(name):return [json.loads(s) for s in (D/'train_val'/(name+'.jsonl')).read_text().splitlines()]
paths={r['guideline_id']:r for r in load('case_paths')};cases={(r['guideline_id'],r['path_id']):r for r in load('vignette_cases')};trees={r['guideline_id']:r for r in load('decision_trees')}
reviews=json.loads((OUT/'pair_reviews.json').read_text());historical={3,5,9,14,18,20,22};verified=[];failures=[];samples=[];references={};oracles={};used=set();source_hashes={}
model=json.loads((ROOT/'configs/models.json').read_text())['models']['qwen3_8b']['path'];tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
class Backend:model_path=model
rules_by_guide={}
for review in reviews:
 gid=review['guideline_id'];pid=review['source_path_id']
 try:
  if gid in historical:raise ValueError('historical_guideline_in_development')
  if (gid,review['node_index']) in used:raise ValueError('duplicate_tree_condition_pair')
  source={'patient':cases[(gid,pid)]['text'],'path':paths[gid]['paths'][pid-1]};v=verify_pair(review,source,paths[gid]['paths'])
  # Check case_paths against the independent original tree representation.
  tree=json.loads(trees[gid]['text'])['decision_tree']
  for route in [v['original_path'],v['counterfactual_path']]:
   cursor=tree
   for element in route[:-1]:
    if cursor['index']!=element['index'] or cursor['description']!=element['condition']:raise ValueError('path_tree_condition_mismatch')
    cursor=cursor[element['decision_outcome']]
   if cursor['index']!=route[-1]['index'] or cursor['conclusion']!=route[-1]['conclusion']:raise ValueError('path_tree_action_mismatch')
  if gid not in rules_by_guide:
   rules=[]
   for route in paths[gid]['paths']:
    conditions=[{'condition_id':f'mega_{gid}_node_{n["index"]}_{n["decision_outcome"]}','text':n['condition'],'negated':n['decision_outcome']=='no'} for n in route[:-1]]
    text='Apply this recommendation only when ALL requirements hold: '+' AND '.join('The answer must be '+('NO' if c['negated'] else 'YES')+': '+c['text'] for c in conditions)+'. Recommendation: '+route[-1]['conclusion']
    rules.append({'rule_id':f'mega_{gid}_leaf_{route[-1]["index"]}','text':text,'action':route[-1]['conclusion'],'conditions':conditions,'source_node_index':route[-1]['index'],'exceptions':[]})
   rules_by_guide[gid]=rules
  rules=rules_by_guide[gid];pair_id=review['pair_id'];target_id=f'mega_{gid}_leaf_{next(p[-1]["index"] for p in [v["original_path"],v["counterfactual_path"]] if p[-2]["decision_outcome"]=="yes")}'
  local=[];localrefs={};localoracles={}
  for member,patient,route,evidence in [('original',v['original_patient'],v['original_path'],v['evidence']['original']),('counterfactual',v['counterfactual_patient'],v['counterfactual_path'],v['evidence']['counterfactual'])]:
   cid=pair_id+'_'+member;gold=f'mega_{gid}_leaf_{route[-1]["index"]}';other=next(p for p in [v['original_path'],v['counterfactual_path']] if p[-1]['index']!=route[-1]['index']);comp=f'mega_{gid}_leaf_{other[-1]["index"]}'
   item={'decision_protocol':'single_token_label.v2','case_id':cid,'pair_id':pair_id,'guideline_id':gid,'guideline':{'rules':rules},'patient':patient,'question':'For this patient, which next action is recommended by the complete guideline?','input_source':{'dataset':'MEGA-CDP','split':'train_val','guideline_id':gid,'filename':trees[gid]['filename'],'original_path_id':pid,'counterfactual':member=='counterfactual','single_factor_edit':True}}
   options=options_for(item)
   if any(len(tokenizer.encode(o["label"],add_special_tokens=False))!=1 for o in options):raise ValueError("option_label_not_single_token")
   prompt,enc,mapping=build_prompt(tokenizer,item)
   payload,key=payload_for(item,Backend(),{'annotation_evidence_slots':6})
   if len(tokenizer.encode(prompt,add_special_tokens=False))>16000 or len(tokenizer.encode(json.dumps(payload,ensure_ascii=False),add_special_tokens=False))>15500:raise ValueError('full_guideline_input_budget_exceeded')
   local.append(item);states={r['rule_id']:('applicable' if r['rule_id']==gold else 'inapplicable') for r in rules}
   ref={'case_id':cid,'pair_id':pair_id,'guideline_id':gid,'gold_rule_id':gold,'gold_action':route[-1]['conclusion'],'competitor_action':other[-1]['conclusion'],'target_rule_id':target_id,'target_action':next(r['action'] for r in rules if r['rule_id']==target_id),'target_applicability':'applicable' if gold==target_id else 'inapplicable','rule_states':states,'evidence':evidence['text'],'evidence_span':evidence,'information_sufficient':True,'incompatible_actions_verified':True,'source_pair_audit':pair_id,'source_route':route}
   localrefs[cid]=ref
   # Oracle relevance includes ALL supplied guideline rules, not gold-selected
   # rule-only prompting. A rule is refuted at its first divergent signed node.
   known={n['index']:n for n in route[:-1]};oracle=[]
   for r,rule_path in zip(rules,paths[gid]['paths']):
    isgold=r['rule_id']==gold;ev=evidence if r['rule_id'] in (gold,comp) else None
    if not isgold and ev is None:
     divergence=next(n for n in rule_path[:-1] if n['index'] in known and n['decision_outcome']!=known[n['index']]['decision_outcome'])
     proof=next(p for p in v['ancestor_checks'] if p['node_index']==divergence['index'])
     if 'numeric_check' in proof:
      ev=evidence
     else:
      quoted=proof['unchanged_exact_evidence'][0]['text'];start=patient.index(quoted);ev={'text':quoted,'start':start,'end':start+len(quoted)}
    supporting=[{'text':evidence['text'],'start':evidence['start'],'end':evidence['end']}]
    if isgold:
     # Full verified ancestor evidence, plus deciding changed-condition quote.
     for proof in v['ancestor_checks']:
      if 'unchanged_exact_evidence' in proof:
       q=proof['unchanged_exact_evidence'][0]['text'];a=patient.index(q)
       if not any(e['text']==q for e in supporting):supporting.append({'text':q,'start':a,'end':a+len(q)})
    oracle.append({'rule_id':r['rule_id'],'relevant':True,'applicability':states[r['rule_id']],'action_spans':[r['action']],'supporting_evidence':supporting if isgold else [],'refuting_evidence':[{'text':ev['text'],'start':ev['start'],'end':ev['end']}] if not isgold else [],'exception_spans':[]})
   localoracles[cid]={'verified':True,'rules':oracle,'scope':'All guide rules retained and relevant; signed-state truth and original evidence from separately audited complete route; no oracle rule-only prompt'}
  samples.extend(local);references.update(localrefs);oracles.update(localoracles);used.add((gid,review['node_index']));verified.append({'review':review,'verification':v})
 except (ValueError,StopIteration,KeyError) as error:failures.append({'pair_id':review['pair_id'],'reason':str(error)})
status='ready' if len(verified)>=40 and not failures else 'blocked'
audit={'schema_version':'acre_pilot.data_audit.v2','status':status,'required_development_pairs':40,'verified_development_pairs':len(verified),'development_cases':len(samples),'failed_pairs':failures,'guideline_ids':sorted({s['guideline_id'] for s in samples}),'historical_guideline_ids':sorted(historical),'guide_overlap_with_history':sorted({s['guideline_id'] for s in samples}&historical),'independent_validation_run':False,'independent_validation_guideline_ids':[],'validation_isolation':'Validation not selected; future validation must exclude development and historical guide IDs and duplicate-source families.','source':cfg,'source_jsonl_sha256':{name:hashlib.sha256((D/'train_val'/(name+'.jsonl')).read_bytes()).hexdigest() for name in ['decision_trees','case_paths','vignette_cases']},'frozen_pair_review_sha256':hashlib.sha256((OUT/'pair_reviews.json').read_bytes()).hexdigest(),'pairs':verified,'condition_types':dict(collections.Counter('numeric_threshold' if r.get('numeric_spec') else 'explicit_boolean_feasibility_or_availability' for r in reviews)),'automatic_checks':['unique source substring','one reviewed causal factor','all other characters unchanged','unchanged reviewed original ancestor evidence or invariant numeric truth','original state matches path','one uniquely matched counterfactual path','independent decision-tree paths/actions exact match','fixed original guideline complete action options','input/token budget','one pair per distinct guideline decision node'],'review_performer':'Code-assisted source-text review by the coding agent; no independent human or clinical expert adjudication.','source_text_review_checks':['original signed path supported by actual patient context','clinical consistency','current-episode action incompatibility; shared ancillary care alone not counted as equivalence'],'review_limitations':['Manual semantic reviews are deterministic source-text adjudications, not clinical expert certification. Boolean templates cover explicit statements; they are not a general clinical NLI verifier.','37/40 pairs concern documented availability/feasibility and 3 numerical thresholds; generalization to subtle clinical exceptions is not established.'],'reference_usage':{'construction_and_offline_evaluation':True,'annotation':False,'normal_decision_prompt':False,'normal_attention_targets':False,'oracle_B6_only':True},'selection_independent_of_predictions':True,'old_exclusions_preserved':True}
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'data_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
if status=='ready':
 for name,value in [('development_manifest.json',{'samples':samples}),('development_references.json',references),('development_oracle.json',oracles)]:
  (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
if status=='ready':
 audit['manifest_sha256']=hashlib.sha256((OUT/'development_manifest.json').read_bytes()).hexdigest()
 audit['references_sha256']=hashlib.sha256((OUT/'development_references.json').read_bytes()).hexdigest()
 audit['oracle_sha256']=hashlib.sha256((OUT/'development_oracle.json').read_bytes()).hexdigest()
 (OUT/'data_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':status,'pairs':len(verified),'cases':len(samples),'failures':failures}))
