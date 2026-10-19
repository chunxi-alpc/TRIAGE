#!/usr/bin/env python3
"""Offline-reference corruption diagnostics, never a normal inference path."""
import argparse,copy,hashlib,json,os,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.run_acre_pilot import logits,write
from src.module1.backend import LocalBackend
from src.acre_pilot.span_mapping import build_prompt,map_annotations,select_targets,options_for
from src.acre_pilot.attention_bias import AttentionBias
BASE=ROOT/'results/acre_pilot/round2';OUT=BASE/'revision3';SEED=20261015

def plans():
 items=json.loads((BASE/'development_manifest.json').read_text())['samples'];refs=json.loads((BASE/'development_references.json').read_text());oracle=json.loads((BASE/'development_oracle.json').read_text())
 order=sorted(x['case_id'] for x in items);random.Random(SEED).shuffle(order);jobs=[]
 for item in items:
  cid=item['case_id'];rid=refs[cid]['target_rule_id']
  predicted=json.loads((OUT/'development'/('case_'+cid+'.json')).read_text())['annotations']
  ann=copy.deepcopy(predicted)
  changed=[]
  for r in ann:
   old=r['applicability'];r['applicability']=refs[cid]['rule_states'][r['rule_id']]
   if old!=r['applicability']:changed.append({'rule_id':r['rule_id'],'old':old,'new':r['applicability']})
  jobs.append({'name':'reference_state_only','item':item,'rules':ann,'change':{'case_id':cid,'field':'applicability','changes':changed,'predicted_relevance_and_evidence_unchanged':True}})
  for rate in (.1,.2,.3):
   if cid not in order[:round(len(order)*rate)]:continue
   ann=copy.deepcopy(oracle[cid]['rules']);r=next(r for r in ann if r['rule_id']==rid);old=r['applicability'];r['applicability']='inapplicable' if old=='applicable' else 'applicable'
   jobs.append({'name':'flip_'+str(rate),'item':item,'rules':ann,'change':{'case_id':cid,'rule_id':rid,'field':'applicability','old':old,'new':r['applicability'],'evidence_unchanged':True}})
  if cid in order[:20]:
   ann=copy.deepcopy(oracle[cid]['rules']);r=next(r for r in ann if r['rule_id']==rid);field='supporting_evidence' if r['applicability']=='applicable' else 'refuting_evidence';text=item['patient'];bad=refs[cid]['evidence'];start=text.find(bad)
   if start<0 or text.count(bad)!=1:raise ValueError('Ambiguous audited decisive evidence')
   length=len(bad);candidates=[]
   for a in range(max(0,len(text)-length+1)):
    b=a+length
    if b<=start or a>=start+len(bad):candidates.append((a,b))
   if not candidates:
    jobs.append({'name':'span_noise_25pct','item':item,'rules':ann,'change':{'case_id':cid,'rule_id':rid,'field':field,'status':'no_disjoint_equal_length_window','changed':False}});continue
   rng=random.Random(SEED+sum(cid.encode()));a,b=rng.choice(candidates);old=r[field];r[field]=[{'text':text[a:b],'start':a,'end':b}]
   jobs.append({'name':'span_noise_25pct','item':item,'rules':ann,'change':{'case_id':cid,'rule_id':rid,'field':field,'old':old,'new':r[field],'changed':True,'same_character_length_as_decisive_evidence':True,'overlaps_changed_condition':False}})
 return jobs

def main():
 p=argparse.ArgumentParser();p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(SEED)
 backend=LocalBackend();backend._load_qwen();out=OUT/'noise';out.mkdir(parents=True,exist_ok=True)
 for i,job in enumerate(plans()):
  if i%a.workers!=a.worker:continue
  item=job['item'];cid=item['case_id'];dest=out/(job['name']+'_'+cid+'.json')
  if dest.exists():continue
  prompt,encoded,mapping=build_prompt(backend.tokenizer,item);encoded=encoded.to(backend.model.device);mapped=map_annotations(item,job['rules'],mapping)
  if any(r['mapping_errors'] for r in mapped):raise ValueError('Corrupted span failed source mapping')
  targets,errors=select_targets(mapped,mapping,'B3')
  if job['name']=='reference_state_only':
   applicable=[r for r in mapped if r['relevant'] and r['applicability']=='applicable']
   actions={x['text'] for r in applicable for x in r['mapped']['action_spans'] if any(x['text']==o['action'] for o in options_for(item)[:-1])}
   if len(applicable)>1 and (len(actions)!=1 or any(not r['mapped']['action_spans'] for r in applicable)):targets=[]
  clean=json.loads((OUT/'development'/('case_'+cid+'_oracle.json')).read_text());clean_targets=clean['variants']['B6@2.0']['attention']['targets']
  with AttentionBias(backend.model,targets,2.,mapping['prompt_tokens']-1) as bias:value=logits(backend,encoded)
  options=options_for(item);ids=[backend.tokenizer.encode(r['label'],add_special_tokens=False)[0] for r in options];chosen=value[ids];lp=value.log_softmax(-1)[ids];index=int(chosen.argmax())
  row={'case_id':cid,'name':job['name'],'seed':SEED,'beta':2.,'reference_diagnostic_only':True,'change':job['change'],'annotation_rules':job['rules'],'base_prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'variant':{'prediction':options[index]['label'],'action':options[index]['action'],'option_logits':chosen.tolist(),'option_log_probabilities':lp.tolist(),'option_normalized_probabilities':chosen.softmax(0).tolist(),'attention':bias.stats},'target_set_changed':set(targets)!=set(clean_targets),'added_tokens_vs_clean_oracle':len(set(targets)-set(clean_targets)),'removed_tokens':len(set(clean_targets)-set(targets)),'errors':errors}
  if row['base_prompt_sha256']!=clean['base_prompt_sha256']:raise ValueError('Decision prompt differs')
  write(dest,row)
 write(out/('done_'+str(a.worker)+'.json'),{'complete':True,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3})
if __name__=='__main__':main()
