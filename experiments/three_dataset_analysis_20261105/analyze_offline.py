"""Offline reanalysis: no model loading, annotation, or GPU inference."""
import json,random,collections,sys,importlib.util
import numpy as np
from common import *
from fusion import fit,evaluate

def deletion():
 reports=[];paired=[]
 for backbone in ('base_final','task_only_final'):
  for split in ('dev','test'):
   p=REPO/f'outputs/S8/physician_{split}/{backbone}/cases.jsonl';rr=rows(p)
   full={x['case_hash']:x for x in rr if x['policy']=='FULL'}
   groups=collections.defaultdict(list)
   for r in rr:
    if r['policy']=='FULL':continue
    groups[(r['policy'],r.get('candidate'))].append(r)
   reports.append(dict(backbone=backbone,split=split,policy='FULL',candidate=None,n=len(full),accuracy=rate([r['correct'] for r in full.values()]),source=str(p)))
   for (policy,candidate),rs in groups.items():
    by=collections.defaultdict(list)
    for r in rs:by[r['case_hash']].append(r)
    data=[]
    for cid,variants in by.items():
     f=full[cid];acc=np.mean([v['correct'] for v in variants]);s=np.mean([v['task_score'] for v in variants])
     data.append(dict(case_id=cid,split=split,backbone=backbone,policy=policy,candidate=candidate,full_correct=int(f['correct']),masked_correct_mean=float(acc),A=float(f['correct']-acc),gold_logp_change=float(s-f['task_score']),correct_to_wrong=float(f['correct']*(1-acc)),wrong_to_correct=float((not f['correct'])*acc),control_repeats=len(variants),source_path=str(p)))
    paired.extend(data)
    reports.append(dict(backbone=backbone,split=split,policy=policy,candidate=candidate,n=len(data),output_rows=len(rs),accuracy=mean_ci([r['masked_correct_mean'] for r in data]),full_paired_accuracy=rate([r['full_correct'] for r in data]),A=mean_ci([r['A'] for r in data]),gold_logp_change=mean_ci([r['gold_logp_change'] for r in data]),correct_to_wrong=mean_ci([r['correct_to_wrong'] for r in data]),correct_to_wrong_given_full_correct=bootstrap([r['correct_to_wrong'] for r in data],[r['full_correct'] for r in data]),source=str(p)))
 write_rows(ROOT/'results/medpair_deletion_pairs.jsonl',paired);write_json(ROOT/'results/medpair_deletion.json',reports)
 return reports

def pair_metrics(records,groups=None):
 n=len(records);a=np.array([r['pred_a']==r['gold_a'] for r in records]);b=np.array([r['pred_b']==r['gold_b'] for r in records])
 unchanged=np.array([r['pred_a']==r['pred_b'] for r in records]);persist=np.array([r['pred_a']==r['gold_a'] and r['pred_b']==r['gold_a'] for r in records])
 d=np.array([r['delta'] for r in records]);toward=d>0;failure=~(a&b)
 return dict(n=n,accuracy_a=rate(a,groups),accuracy_b=rate(b,groups),accuracy_b_minus_a=mean_ci(b.astype(float)-a,groups),P_b=rate(failure,groups),
  single_error=bootstrap((~a).astype(int)+(~b).astype(int),np.full(n,2),groups),unchanged=rate(unchanged,groups),
  correct_then_persist_old=rate(persist,groups),correct_then_persist_old_conditional=bootstrap(persist,a,groups),
  delta_positive=rate(toward,groups),delta_mean=mean_ci(d,groups),delta_median=float(np.median(d)),
  toward_but_pair_failure=rate(toward&failure,groups),toward_but_changed_case_wrong=rate(toward&~b,groups),not_toward=rate(~toward,groups))

def guide():
 source=HOME/'qwen800_methods_20261023/private';cases=rows(source/'cases.jsonl');case_by={r['id']:r for r in cases}
 trees=sorted({r['disease'] for r in cases});random.Random(42).shuffle(trees)
 split_by={r['id']:'dev' if r['disease'] in trees[:11] else 'test' for r in cases}
 assert len(trees)==53 and sum(s=='dev' for s in split_by.values())==149
 write_json(ROOT/'data/medguide_split.json',dict(seed=42,dev_trees=trees[:11],test_trees=trees[11:],case_split=split_by))
 path=HOME/'TRIAGE-worktrees/qwen800-methods/experiments/aster_validation_20261022/mechanism_20261023/run.py'
 spec=importlib.util.spec_from_file_location('old_guide_prompts',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 data={};supp=collections.defaultdict(list);inputs=[]
 for part in (0,1):
  p=source/f'predictions{part}.jsonl'
  for r in rows(p):
   split=split_by[r['id']]
   labels,l,pred=score_row(r)
   if r['method'] not in ('branch_visible','branch_hidden'):
    supp[(r['method'],r['kind'],split,r['variant'],r.get('oracle',False))].append((pred==r['gold'],r['prediction']==r['gold']))
    continue
   c=case_by[r['id']];assert labels==[chr(65+j) for j in range(len(c['branches']))]
   assert r['gold']==chr(65+c['gold_branch'][r['variant']])
   mode=r['method'].removeprefix('branch_');prompt=mod.branch_prompt(c,r['variant'],mode)
   nr=dict(id=r['id'],tree=c['disease'],protocol=c['pair_protocol'],split=split,variant=r['variant'],view=mode,
    labels=labels,gold=r['gold'],prediction=pred,log_probabilities=l.tolist(),raw_logits=r['option_logits'],generated_prediction=r['prediction'],source_path=str(p),source_case_path=str(source/'cases.jsonl'),prompt=prompt)
   data[(r['id'],r['variant'],mode)]=nr;inputs.append(nr)
 write_rows(ROOT/'data/medguide_scores_and_inputs.jsonl',inputs)
 stats=[];swaps=[];swap_details=[];fusion_data=[];threshold=[]
 for c in cases:
  cid=c['id'];base=data[cid,'A','hidden']
  for v in c['variant_rules']:
   f=data[cid,v,'visible'];m=data[cid,v,'hidden'];assert f['labels']==m['labels'] and f['gold']==m['gold']
   fusion_data.append(dict(id=cid,tree=c['disease'],protocol=c['pair_protocol'],variant=v,split=split_by[cid],labels=f['labels'],gold_idx=f['labels'].index(f['gold']),f=f['log_probabilities'],m=m['log_probabilities']))
   if v in ('B','C','D'):
    assert base['prompt']==m['prompt'] and base['gold']==m['gold']
    for mode in ('visible','hidden'):
     a=data[cid,'A',mode];b=data[cid,v,mode];ac=a['prediction']==a['gold'];bc=b['prediction']==b['gold']
     swap_details.append(dict(id=cid,tree=c['disease'],protocol=c['pair_protocol'],split=split_by[cid],variant=v,view=mode,
      changed=a['prediction']!=b['prediction'],correct_to_wrong=ac and not bc,wrong_to_correct=not ac and bc,
      correct_a=ac,correct_variant=bc,gold_logp_change=b['log_probabilities'][b['labels'].index(b['gold'])]-a['log_probabilities'][a['labels'].index(a['gold'])],
      hidden_identical_input=base['prompt']==m['prompt'],hidden_identical_scores=np.allclose(base['log_probabilities'],m['log_probabilities'],rtol=0,atol=1e-6)))
   if v=='CS':
    assert split_by[cid]=='test' and m['gold']!=base['gold']
    ya=base['labels'].index(base['gold']);yb=m['labels'].index(m['gold'])
    threshold.append(dict(id=cid,tree=c['disease'],split='test',gold_a=base['gold'],gold_b=m['gold'],pred_a=base['prediction'],pred_b=m['prediction'],
      delta=(m['log_probabilities'][yb]-m['log_probabilities'][ya])-(base['log_probabilities'][yb]-base['log_probabilities'][ya]),
      new_gold_logp_change=m['log_probabilities'][yb]-base['log_probabilities'][yb],
      old_gold_logp_change=m['log_probabilities'][ya]-base['log_probabilities'][ya],
      own_gold_logp_change=m['log_probabilities'][yb]-base['log_probabilities'][ya]))
 assert len(threshold)==84
 for split in ('dev','test','all'):
  for protocol in ('all','PR23_strict','extended_offpath'):
   for v in ('A','B','C','D','CS'):
    for mode in ('visible','hidden'):
     rr=[r for r in inputs if (split=='all' or r['split']==split) and r['variant']==v and r['view']==mode and (protocol=='all' or r['protocol']==protocol)]
     if not rr:continue
     groups=[r['tree'] for r in rr];stats.append(dict(split=split,protocol=protocol,variant=v,view=mode,n=len(rr),accuracy=rate([r['prediction']==r['gold'] for r in rr],groups),generated_accuracy=rate([r['generated_prediction']==r['gold'] for r in rr],groups)))
   for v in ('B','C','D'):
    for mode in ('visible','hidden'):
     rr=[r for r in swap_details if (split=='all' or r['split']==split) and r['variant']==v and r['view']==mode and (protocol=='all' or r['protocol']==protocol)]
     if not rr:continue
     groups=[r['tree'] for r in rr];swaps.append(dict(split=split,protocol=protocol,variant=v,view=mode,n=len(rr),
      changed=rate([r['changed'] for r in rr],groups),correct_to_wrong=rate([r['correct_to_wrong'] for r in rr],groups),
      correct_to_wrong_given_correct=bootstrap([r['correct_to_wrong'] for r in rr],[r['correct_a'] for r in rr],groups),
      wrong_to_correct=rate([r['wrong_to_correct'] for r in rr],groups),
      accuracy_change=mean_ci([int(r['correct_variant'])-int(r['correct_a']) for r in rr],groups),
      gold_logp_change=mean_ci([r['gold_logp_change'] for r in rr],groups)))
 calibration=fit([r for r in fusion_data if r['split']=='dev']);write_json(ROOT/'results/medguide_fusion_parameters.json',calibration)
 fusion=[];fused=[]
 for v in ('A','B','C','D','CS'):
  dd=[r for r in fusion_data if r['split']=='test' and r['variant']==v]
  for kind,param in dict(Full=None,Masked=None,**calibration['parameters']).items():
   result,rr=evaluate(dd,kind,param);fusion.append(dict(dataset='MedGUIDE',variant=v,method=kind,parameter=param,**result));fused.extend(rr)
 threshold_result=pair_metrics(threshold,[r['tree'] for r in threshold])
 for key in ('new_gold_logp_change','old_gold_logp_change','own_gold_logp_change'):threshold_result[key]=mean_ci([r[key] for r in threshold],[r['tree'] for r in threshold])
 within_tree=pair_metrics(threshold)
 for key in ('new_gold_logp_change','old_gold_logp_change','own_gold_logp_change'):within_tree[key]=mean_ci([r[key] for r in threshold])
 within_tree.update(bootstrap_unit='case pair within the single source guideline tree',interpretation='conditional on the observed tree; not across-guideline uncertainty')
 write_json(ROOT/'results/medguide_threshold_within_tree_supplement.json',within_tree)
 write_json(ROOT/'results/medguide_threshold.json',threshold_result);write_rows(ROOT/'results/medguide_threshold_pairs.jsonl',threshold)
 write_rows(ROOT/'results/medguide_swap_pairs.jsonl',swap_details);write_json(ROOT/'results/medguide_swaps.json',swaps)
 write_json(ROOT/'results/medguide_accuracy.json',stats);write_json(ROOT/'results/medguide_fusion.json',fusion)
 write_rows(ROOT/'results/medguide_fusion_predictions.jsonl',fused)
 write_json(ROOT/'results/medguide_historical_supplement.json',[dict(method=k[0],answer_space=k[1],split=k[2],variant=k[3],oracle=k[4],n=len(v),score_argmax_accuracy=float(np.mean([x[0] for x in v])),generated_accuracy=float(np.mean([x[1] for x in v]))) for k,v in supp.items()])
 write_json(ROOT/'results/medguide_pairing_checks.json',dict(cases=800,strict=304,extended=496,dev=149,test=651,threshold_pairs=84,hidden_identical_input=all(r['hidden_identical_input'] for r in swap_details),hidden_identical_scores=all(r['hidden_identical_scores'] for r in swap_details),baseline_count='A only; identical B/C/D hidden inputs not independent baseline observations',CS_tree_count=len(set(r['tree'] for r in threshold))))

def einst():
 root=HOME/'causal-evidence-routing-private/reports';gold={}
 m1summary=[]
 for split in ('train','test'):
  rr=rows(root/f'm1/behavior/{split}_behavior.jsonl')
  gold.update({r['pair_id']:(r['target_a'],r['target_b']) for r in rr})
  for score in ('score_sum','score_mean'):
   records=[]
   for r in rr:
    a,b=r['target_a'],r['target_b'];s=r[score]
    # Both saved margins are score(target_a)-score(target_b).
    pa=a if s['margin_a']>=0 else b;pb=a if s['margin_b']>=0 else b
    records.append(dict(id=r['pair_id'],gold_a=a,gold_b=b,pred_a=pa,pred_b=pb,delta=s['margin_a']-s['margin_b']))
   m1summary.append(dict(split=split,score=score,**pair_metrics(records)))
 write_json(ROOT/'results/medeinst_m1_supplement.json',m1summary)
 results=[];allpairs=[]
 for p in sorted((root/'m3a/scoring').glob('*/*.jsonl')):
  for score in ('logprob_sum','logprob_mean'):
   records=[]
   for i,r in enumerate(rows(p),1):
    ya,yb=gold[r['pair_id']];ca=r['conditions']['a']['candidates'];cb=r['conditions']['b']['candidates']
    aa={x['diagnosis']:x[score] for x in ca};bb={x['diagnosis']:x[score] for x in cb}
    assert set(aa)==set(bb) and len(aa)==46 and ya!=yb
    labels=sorted(aa);pa=max(labels,key=aa.get);pb=max(labels,key=bb.get)
    records.append(dict(id=r['pair_id'],model=r['model_slug'],split=r['split'],score=score,gold_a=ya,gold_b=yb,pred_a=pa,pred_b=pb,delta=(bb[yb]-bb[ya])-(aa[yb]-aa[ya]),new_gold_score_change=bb[yb]-aa[yb],old_gold_score_change=bb[ya]-aa[ya],source_path=str(p),source_line=i,
     labels=labels,scores_a=[aa[l] for l in labels],scores_b=[bb[l] for l in labels],log_probabilities_a=lp([aa[l] for l in labels]).tolist(),log_probabilities_b=lp([bb[l] for l in labels]).tolist()))
   assert len(records)==(512 if p.stem=='train_discovery' else 256)
   results.append(dict(model=p.parent.name,split=p.stem,score=score,**pair_metrics(records)));allpairs.extend(records)
 write_rows(ROOT/'results/medeinst_independent_predictions.jsonl',allpairs);write_json(ROOT/'results/medeinst_m3a.json',results)
 import pandas as pd
 work=[]
 for exp,rel in [('E1','e1_behavior'),('E5','e5_causal_utilization/behavior')]:
  p=REPO/f'legacy/work3/frozen-e5-stop-20260903/analysis/{rel}/pair_results.parquet'
  frame=pd.read_parquet(p)
  for split,d in frame.groupby('split'):
   records=[]
   for r in d.to_dict('records'):
    records.append(dict(id=r['case_id'],gold_a=r['control_gt'],gold_b=r['trap_gt'],pred_a=r['control_prediction'],pred_b=r['trap_prediction'],delta=(r['trap_gt_mean_logprob']-r['trap_old_mean_logprob'])-(r['control_alt_mean_logprob']-r['control_gt_mean_logprob'])))
   work.append(dict(experiment=exp,split=split,prediction='historical free generation',delta_score='historical token mean',source=str(p),**pair_metrics(records)))
 write_json(ROOT/'results/medeinst_work3_supplement.json',work)

def supplementary():
 pcd=[]
 for name in ('base','task_only','full_v1'):
  p=REPO/f'outputs/S6/pcd_dev/{name}/cases.jsonl';by=collections.defaultdict(list)
  for r in rows(p):by[r['pcd_lambda']].append(r)
  for lam,rr in by.items():
   pcd.append(dict(backbone=name,pcd_lambda=lam,n=len(rr),source=str(p),**{key:float(np.mean([r[key] for r in rr])) for key in ('mcq_accuracy_no_priority','mcq_accuracy_gold_priority','mcq_accuracy_swapped_priority','task_score_no_priority','task_score_gold_priority','task_score_swapped_priority')}))
 write_json(ROOT/'results/medpair_pcd_supplement.json',pcd)
 # Preserve the historical CEBU report independently; do not pool overlapping sources.
 import shutil
 src=HOME/'CEBU_SERVER_RUN_PACKAGE/outputs/reports/balanced_pilot_gpt_5_4_mini_50case_20260515/report.json'
 shutil.copyfile(src,ROOT/'results/cebu_historical_report.json')
 write_json(ROOT/'results/historical_source_notes.json',dict(CEBU='Historical 50-case balanced report copied separately; overlaps the 65-case early source. No pooling.',PCD='Existing development-only priority decoding; no retraining',MedGUIDE='Autoregressive CAD and internal/oracle methods reported separately from branch-score fusion',MedEinst='M1 two-diagnosis scores and Work3 generated predictions are supplementary; primary is M3A independent 46-candidate argmax.'))

def main():
 deletion();guide();einst();supplementary()
 write_json(ROOT/'results/offline_status.json',dict(status='complete',automatic_preexperiment=True,new_annotations=0,gpu_inference=False))
if __name__=='__main__':main()
