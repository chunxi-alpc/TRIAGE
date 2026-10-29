"""prepare -> smoke -> fit -> select -> freeze -> evaluate -> report.
All controlled pairs are synthetic logical tasks, never clinical gold.
"""
import argparse,json,re,time,traceback,random,os
from pathlib import Path
from collections import Counter,defaultdict
import torch
from fullscale_data import write,prepare
from fullscale_engine import Engine
from paired_cluster_stats import summarize
META={}
def metadata():
 if not META:
  for split in ('train_val','test'):
   for line in (Path('/home/gcx/TRIAGE/data/MEGA-CDP')/split/'vignette_cases.jsonl').open():
    r=json.loads(line);META[f'mega:{split}:{r["guideline_id"]}:{r["path_id"]}']={k:r[k] for k in ('language','filename')}
 return META
METHODS=['baseline','learned','direct','mean','random','reverse','oracle','zero']
def rows(root,dataset,splits):
 for line in (Path(root)/(dataset+'.jsonl')).open():
  r=json.loads(line)
  if r['split'] in splits:
   if dataset=='mega':r.update(metadata()[r['case_id']])
   yield r

def fit(e,args,out,limit=None):
 layers=[e.model.config.num_hidden_layers//3,e.model.config.num_hidden_layers//2,e.model.config.num_hidden_layers*2//3]
 sums={};cov={};seen=set();n=0;fail=Counter();admit=Counter()
 with (out/'pair_log.jsonl').open('x') as log:
  for c in rows(args.run,'mega',{'fit'}):
   if c['invalid_source']:continue
   for rule in c['rules']:
    key=(c['group'],rule['id'])
    if key in seen:continue
    seen.add(key);admit['candidate_conditions']+=1
    cond=rule['condition']
    if not cond.strip():admit['empty_condition']+=1;continue
    prompts=[e.prompt('Rule: If condition C is true, choose A; if condition C is false, choose B.\nCondition C: '+cond+'\nPatient fact: condition C is '+state+'.\nReturn A or B only. Answer:') for state in ('true','false')]
    try:
     ids=[e.encode(p)[0]['input_ids'].shape[1] for p in prompts]
     if ids[0]!=ids[1]:admit['token_length_unaligned']+=1;continue
     a,b=[e.query(p,layers) for p in prompts]
     for l in layers:
      d=(a[l]-b[l]).double();sums[l]=sums.get(l,torch.zeros_like(d))+d
      outer=torch.einsum('hi,hj->hij',d,d);cov[l]=cov.get(l,torch.zeros_like(outer))+outer
     n+=1;status='accepted'
    except Exception as ex:status=type(ex).__name__;fail[status]+=1
    log.write(json.dumps({'group':c['group'],'node':rule['id'],'status':status,'synthetic':True})+'\n');log.flush()
    if limit and n>=limit:break
   if limit and n>=limit:break
 (out/'pair_counts.json').write_text(json.dumps({'candidate_conditions':len(seen),'accepted_pairs':n,'admission':dict(admit),'technical_failures':dict(fail),'clinical_gold_pairs':0},indent=2))
 if not n:raise RuntimeError('no_aligned_logical_pairs')
 candidates=[]
 for l in layers:
  vals,vec=torch.linalg.eigh(cov[l]/n)
  for h in range(len(vals)):candidates.append((float(vals[h,-1]),l,h,vec[h,:,-1].float(),(sums[l][h]/n).float()))
 _,l,h,d,mean=max(candidates,key=lambda x:x[0]);d=d/d.norm();mean=mean/mean.norm().clamp_min(1e-8)
 generator=torch.Generator().manual_seed(20261029);rd=torch.randn(d.shape,generator=generator);rd/=rd.norm()
 # Reverse is a negative intervention strength, because a rank-one projector is sign-invariant.
 artifact={'selection':{'layer':l,'head':h,'strength':.5},'directions':{'learned':d,'mean':mean,'random':rd,'reverse':d},'training_pairs':n,'training_split':'mega:fit','method':'leading eigenvector of paired final-query second moment; raw paired mean control'}
 torch.save(artifact,out/'directions.pt');return artifact

def score(c,raw):
 try:
  if c['dataset']=='med':
   m=re.fullmatch(r'\s*([A-Z])[.\s]*',raw);pred=m.group(1) if m else None
   return pred==c['gold'],pred is None,{'mcqa_prediction':pred}
  m=re.search(r'\[[\s\d,]+\]',raw);pred=json.loads(m.group()) if m else None
  if not isinstance(pred,list) or not pred or not all(type(x)==int for x in pred):return False,True,{}
  # Source-node exact path / terminal accuracy; not official DTW semantic PCC.
  gold=c['gold'];prefix=0
  for a,b in zip(pred,gold):
   if a!=b:break
   prefix+=1
  return pred==gold,False,{'predicted_path':pred,'terminal_correct':pred[-1]==gold[-1],'path_prefix_fraction':prefix/len(gold)}
 except Exception:return False,True,{}

def evaluate(e,args,out,artifact,dataset,splits,limit=None,methods=METHODS,tag='evaluation'):
 selected=artifact['selection'];directions=artifact['directions'];counts=Counter();path=out/(tag+'.jsonl')
 judge=None
 if dataset=='mega':
  from official_judge import Judge
  judge=Judge(e)
 history_paths=list(out.parent.glob('worker_*/*.jsonl')) if getattr(args,'world',None) else ([path] if path.exists() else [])
 existing=[]
 for history in history_paths:
  for line in history.open():
   try:existing.append(json.loads(line))
   except json.JSONDecodeError:continue
 done={(r['case_id'],r['method']) for r in existing}
 with path.open('a') as log:
  for case_index,c in enumerate(rows(args.run,dataset,set(splits))):
   if args.stage=='full' and ((dataset=='mega' and tag in ('mega_confirm','mega_official_test')) or (tag=='med_heldout' and hasattr(args,'world'))) and case_index%getattr(args,'world',2)!=args.worker:continue
   if all((c['case_id'],m) in done for m in methods):continue
   if limit and counts['cases']>=limit:break
   counts['cases']+=1;t=time.time();routing={};raw_route='';route_failure=None;prep_failure=None
   try:
    if c['invalid_source']:raise ValueError('invalid_source:'+c['invalid_source'])
    e.active=None
    try:routing,raw_route,route_failure=e.applicability(c)
    except Exception as ex:routing={r['id']:0 for r in c['rules']};route_failure=str(ex)
    text,masks=e.task(c)
   except Exception as ex:prep_failure=str(ex)
   score_cache={}
   metric_keys={'official_pcc','official_accuracy','official_failure','official_pathway_judge','official_accuracy_judge','official_judge_failed','parsed_native_output','mcqa_prediction','predicted_path','terminal_correct','path_prefix_fraction'}
   for previous in existing:
    if previous['case_id']==c['case_id'] and previous.get('raw_output') and not previous['technical_failure']:
     score_cache[previous['raw_output']]=(previous['correct'],previous['invalid_output'],{k:v for k,v in previous.items() if k in metric_keys})
   for method in methods:
    if (c['case_id'],method) in done:continue
    r={'case_id':c['case_id'],'guideline_id':c['group'],'dataset':dataset,'split':c['split'],'method':method,'correct':False,'technical_failure':False,'invalid_output':False,'intervention_applied':False,'route_failure':route_failure,'raw_routing':raw_route,'raw_output':''}
    try:
     if prep_failure:raise ValueError(prep_failure)
     route=c['gold_route'] if method=='oracle' else routing
     selection=dict(selected)
     if method=='reverse':selection['strength']=-selection['strength']
     e.activate(selection,method,masks,route,directions)
     raw=e.generate(text,2048 if dataset=='mega' else 8)
     e.active=None
     r['raw_output']=raw
     if raw not in score_cache:score_cache[raw]=judge.score(c,raw) if judge else score(c,raw)
     r['correct'],r['invalid_output'],metrics=score_cache[raw];r.update(metrics)
     r['intervention_applied']=method not in ('baseline','zero') and any(route.values())
    except Exception as ex:
     r['technical_failure']=True;r['failure_reason']=str(ex);torch.cuda.empty_cache()
    finally:e.active=None
    r['elapsed_case_seconds']=time.time()-t;log.write(json.dumps(r,ensure_ascii=False)+'\n');log.flush()
   if counts['cases']%10==0:write(out/(tag+'_progress.json'),{'cases':counts['cases'],'last_case_id':c['case_id'],'seconds':time.time()-args.started})
 return path

def continuous_stats(rs,base):
 result={}
 for metric,worst in [('official_accuracy',0.),('official_pcc',1.)]:
  if not any(metric in r for r in rs):continue
  groups=defaultdict(list)
  for r in rs:
   if r['case_id'] in base:groups[r['guideline_id']].append((r.get(metric,worst),base[r['case_id']].get(metric,worst)))
  means={g:(sum(x[0] for x in v)/len(v),sum(x[1] for x in v)/len(v)) for g,v in groups.items()}
  gs=sorted(means);rng=random.Random(20261029);deltas=[means[g][0]-means[g][1] for g in gs]
  boot=sorted(sum(rng.choice(deltas) for _ in gs)/len(gs) for _ in range(1000))
  result[metric]={'method_macro':sum(x[0] for x in means.values())/len(gs),'baseline_macro':sum(x[1] for x in means.values())/len(gs),'paired_macro_delta':sum(deltas)/len(gs),'paired_guideline_bootstrap_95ci':[boot[24],boot[974]]}
 return result

def report(out):
 groups=defaultdict(list)
 for p in out.rglob('*.jsonl'):
  if p.name=='pair_log.jsonl':continue
  for line in p.open():
   r=json.loads(line)
   if 'method' in r:groups[(p.stem,r['method'])].append(r)
 summary={}
 for (tag,method),rs in groups.items():
  base={r['case_id']:r for r in groups.get((tag,'baseline'),[])}
  paired=[{'case_id':r['case_id'],'guideline_id':r['guideline_id'],'baseline_correct':base[r['case_id']]['correct'],'method_correct':r['correct'],'technical_failure':r['technical_failure'],'invalid_output':r['invalid_output'],'intervention_applied':r['intervention_applied']} for r in rs if r['case_id'] in base]
  if paired:
   stats=summarize(paired,bootstrap_draws=1000);stats['official_pcc']=sum(r.get('official_pcc',1.) for r in rs)/len(rs) if any('official_pcc' in r for r in rs) else None;stats['official_accuracy']=sum(r.get('official_accuracy',0.) for r in rs)/len(rs) if any('official_accuracy' in r for r in rs) else None;stats['route_failures']=sum(bool(r.get('route_failure')) for r in rs);stats['judge_failures']=sum(bool(r.get('official_judge_failed')) for r in rs);stats['continuous_guideline_statistics']=continuous_stats(rs,base);stats['failure_reasons']=dict(Counter(r.get('failure_reason') for r in rs if r['technical_failure']));summary[tag+':'+method]=stats
 write(out/'summary.json',summary)
 lines=['# 实际运行报告','', 'MEGA 使用官方五次 Qwen 评审、PCC 与结论评分，推理传输改为本地 Transformers；MedGUIDE 为原始 MCQA。所有失败保留在分母。','', '|数据/方法|病例|指南|全对率/MCQA|官方结论分|PCC|失败|无效输出|','|---|---:|---:|---:|---:|---:|---:|---:|']
 for key,s in summary.items():lines.append(f'|{key}|{s["n_cases"]}|{s["n_guideline_clusters"]}|{s["method_accuracy_micro"]:.4f}|{s.get("official_accuracy")}|{s.get("official_pcc")}|{s["failure_rows"]}|{s["invalid_output_rows"]}|')
 (out/'REPORT.zh.md').write_text('\n'.join(lines)+'\n');return summary

def select_strength(e,args,out,artifact):
 scores=Counter();denominator=0;seen=set();sel=dict(artifact['selection']);d=artifact['directions']['learned']
 with (out/'selection_log.jsonl').open('x') as f:
  for c in rows(args.run,'mega',{'select'}):
   if c['invalid_source']:continue
   for r in c['rules']:
    key=(c['group'],r['id'])
    if key in seen:continue
    seen.add(key)
    for state,target,sign in [('true','A',1),('false','B',-1)]:
     user='Rule: If condition C is true, choose A; if condition C is false, choose B.\nCondition C: '+r['condition']+'\nPatient fact: condition C is '+state+'.\nReturn A or B only. Answer:'
     text=e.prompt(user);inp,offset=e.encode(text,True);end=text.index('\nPatient fact:');mask=torch.tensor([b<=end and b>a for a,b in offset]);signs=mask.float()*sign
     for strength in (.1,.5,1.):
      try:
       e.active=dict(layer=sel['layer'],head=sel['head'],strength=strength,signs=signs,method='learned',direction=d)
       with torch.inference_mode():logits=e.model(**inp,use_cache=False).logits[0,-1].float()
       ids=[e.tok.encode(x,add_special_tokens=False)[0] for x in ('A','B')];pred='AB'[int(logits[ids].argmax())];correct=pred==target
      except Exception:correct=False
      finally:e.active=None
      scores[strength]+=int(correct)
     denominator+=1
    f.write(json.dumps({'group':c['group'],'node':r['id'],'scores':dict(scores),'denominator':denominator})+'\n');f.flush()
 sel['strength']=max(scores,key=lambda v:(scores[v],-v));write(out/'selection.json',{'scores':dict(scores),'denominator':denominator,'selection':sel,'task':'synthetic logical applicability heldout source families; no official test tuning'})
 return sel

def main():
 p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=['prepare','smoke','full','report']);p.add_argument('--run',required=True);p.add_argument('--output');p.add_argument('--model',default='/home/gcx/TRIAGE/models/modelscope/Qwen3-8B');p.add_argument('--mega',default='/home/gcx/TRIAGE/data/MEGA-CDP');p.add_argument('--med',default='/home/gcx/TRIAGE/data/MedGUIDE/data/train-00000-of-00001.parquet');p.add_argument('--paths',default='/home/gcx/aster_medguide_20261022/private/all_paths.json');p.add_argument('--artifact');p.add_argument('--worker',type=int,default=0);args=p.parse_args();args.started=time.time()
 if args.stage=='prepare':prepare(args);return
 out=Path(args.output)
 if args.stage=='report':report(out);return
 out.mkdir(parents=True,exist_ok=False);write(out/'status.json',{'status':'running','stage':args.stage,'worker':args.worker,'model':args.model})
 try:
  e=Engine(args.model)
  if args.stage=='smoke':
   artifact=fit(e,args,out,limit=4)
   evaluate(e,args,out,artifact,'mega',['select'],limit=1,tag='mega_smoke')
   evaluate(e,args,out,artifact,'med',['select'],limit=1,tag='med_smoke')
   rs=[json.loads(x) for x in (out/'mega_smoke.jsonl').open()]
   bases={r['case_id']:r['raw_output'] for r in rs if r['method']=='baseline'}
   assert all(r['raw_output']==bases[r['case_id']] for r in rs if r['method']=='zero'),'zero_identity_failure'
   if all(r['technical_failure'] for r in rs):raise RuntimeError('all_smoke_cases_failed')
  elif args.worker==0:
   artifact=torch.load(args.artifact,weights_only=False,map_location='cpu') if args.artifact else fit(e,args,out)
   # Select strength on the complete source-heldout development selection split.
   artifact['selection']=select_strength(e,args,out,artifact)
   torch.save(artifact,out/'frozen.pt');write(out/'frozen.json',{'selection':artifact['selection'],'training_pairs':artifact['training_pairs'],'test_used_for_selection':False})
   evaluate(e,args,out,artifact,'mega',['confirm'],tag='mega_confirm')
   evaluate(e,args,out,artifact,'mega',['official_test'],tag='mega_official_test')
  else:
   frozen=Path(args.run)/'full_gpu0'/'frozen.pt'
   while not frozen.exists():
    status=Path(args.run)/'full_gpu0'/'status.json'
    if status.exists() and json.loads(status.read_text()).get('status')=='failed':raise RuntimeError('training_worker_failed')
    time.sleep(20)
   artifact=torch.load(frozen,weights_only=False,map_location='cpu')
   evaluate(e,args,out,artifact,'med',['heldout'],tag='med_heldout')
   evaluate(e,args,out,artifact,'mega',['confirm'],tag='mega_confirm')
   evaluate(e,args,out,artifact,'mega',['official_test'],tag='mega_official_test')
  report(out);write(out/'status.json',{'status':'complete','stage':args.stage,'seconds':time.time()-args.started,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30})
  if args.stage=='full':
   statuses=[Path(args.run)/('full_gpu'+str(i))/'status.json' for i in (0,1)]
   if all(p.exists() and json.loads(p.read_text()).get('status')=='complete' for p in statuses):report(Path(args.run))
 except Exception as ex:
  write(out/'status.json',{'status':'failed','reason':str(ex),'traceback':traceback.format_exc()});raise
if __name__=='__main__':main()
