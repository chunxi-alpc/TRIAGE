"""Paired descriptive analysis; no invented observations or absent-arm denominators."""
import argparse,csv,gzip,hashlib,json,math,random
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from run import noisy_assignment

def option_prediction(row):
 return row.get('score_prediction') or row.get('prediction')

def wilson(k,n):
 if not n:return [None,None]
 z=1.959963984540054;p=k/n;d=1+z*z/n
 c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
 return [c-h,c+h]

def paired(a,b,seed=20261022,reps=2000):
 """Pair by case ID, resample disease clusters (and report ordinary paired CI)."""
 aa={r['id']:r for r in a};bb={r['id']:r for r in b}
 if set(aa)!=set(bb):raise ValueError('Unmatched paired rows')
 ids=sorted(aa);delta=np.array([float(bb[i]['correct'])-float(aa[i]['correct']) for i in ids]);n=len(ids)
 if not n:return {'n':0}
 win=sum(not aa[i]['correct'] and bb[i]['correct'] for i in ids);loss=sum(aa[i]['correct'] and not bb[i]['correct'] for i in ids)
 rng=np.random.default_rng(seed);means=np.array([delta[rng.integers(n,size=n)].mean() for _ in range(reps)])
 clusters=defaultdict(list)
 for j,i in enumerate(ids):clusters[aa[i]['disease']].append(delta[j])
 cs=list(clusters.values());sums=np.array([sum(c) for c in cs]);sizes=np.array([len(c) for c in cs]);nc=len(cs)
 if nc>1:
  ix=rng.integers(nc,size=(reps,nc));cluster_ci=np.quantile(sums[ix].sum(1)/sizes[ix].sum(1),[.025,.975]).tolist()
 else:cluster_ci=None
 # Paired continuous margin difference, same pairs and disease resampling.
 margins={}
 for metric in ('margin_prior','margin_sibling'):
  diff=np.array([bb[i][metric]-aa[i][metric] for i in ids]);sm=np.array([sum(diff[j] for j,i in enumerate(ids) if aa[i]['disease']==c) for c in clusters])
  margins[metric]={'mean_delta':float(diff.mean()),'disease_bootstrap_ci':np.quantile(sm[ix].sum(1)/sizes[ix].sum(1),[.025,.975]).tolist() if nc>1 else None}
 return {'n':n,'corrected':win,'harmed':loss,'accuracy_delta':float(delta.mean()),'mcnemar_exact_p':float(binomtest(win,win+loss,.5).pvalue) if win+loss else 1.,'paired_bootstrap_ci':np.quantile(means,[.025,.975]).tolist(),'disease_bootstrap_ci':cluster_ci,'disease_clusters':nc,'margin_deltas':margins}

def metrics(rows):
 n=len(rows);k=sum(r['correct'] for r in rows)
 return {'n':n,'correct':k,'accuracy':k/n if n else None,'wilson95':wilson(k,n),'score_accuracy':sum(option_prediction(r)==r['gold_answer'] for r in rows)/n if n else None,'parse_failure_rate':sum(r['prediction'] is None for r in rows)/n if n else None,'original_action_wilson95':wilson(sum(r['prediction']==r['original_answer'] for r in rows),n),'misuse_wilson95':wilson(sum(r['prediction']==r['sibling_action_label'] for r in rows),n),'original_action_selection_rate':sum(r['prediction']==r['original_answer'] for r in rows)/n if n else None,'sibling_misuse_rate':sum(r['prediction']==r['sibling_action_label'] for r in rows)/n if n else None,'margin_prior':float(np.mean([r['margin_prior'] for r in rows])) if n else None,'margin_sibling':float(np.mean([r['margin_sibling'] for r in rows])) if n else None}

def corrupt_ids(ids,rate,seed):
 return set(sorted(ids,key=lambda i:hashlib.sha256(f'{seed}:{i}'.encode()).digest())[:round(len(ids)*rate)])

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--dataset-manifest',required=True);a=p.parse_args()
 inp=Path(a.input);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 manifest=json.loads(inp.with_suffix('.manifest.json').read_text());rows=[json.loads(l) for l in inp.read_text().splitlines()]
 data=json.loads(Path(a.dataset_manifest).read_text());dev=set(data['splits']['dev']['ids']);phase=manifest['phase']
 groups={r['id'] for r in rows};expected=manifest['groups']*manifest.get('rows_per_case',50 if phase=='diagnostic' else 52)
 keys=[(r['id'],r['variant'],r['method'],r.get('gain'),r.get('assigned_wrong')) for r in rows]
 if len(set(keys))!=len(keys) or len(rows)!=expected or len(groups)!=manifest['groups']:raise ValueError('Incomplete or duplicate run')
 result={'phase':phase,'complete':True,'n_rows':len(rows),'n_cases':len(groups),'input_sha256':manifest['input_sha256'],'metrics':{},'comparisons':{},'notes':['Main accuracy uses parsed greedy generation on all conditions; score_accuracy separately uses first-position option argmax. Diagnostics tune gain on option scores.','K/U denotes observed original-action preference/non-selection, not proven parameter memory or knowledge absence; it is model-specific, defined by original closed-book parsed greedy answer in main; diagnostic strata use option scores. Invalid closed-book outputs are counted in U and explicitly flagged by parse_failure_rate.','A/C correctness, B original-action selection, C edited sibling misuse are separate metrics.','Source path labels define constructed counterfactuals; not independent clinical ground truth.']}
 # Add an explicitly heuristic objective-condition comparison without changing model inputs.
 labels={}
 dataset_file=Path(a.dataset_manifest).parent/'all.jsonl'
 if dataset_file.exists():
  import re
  for line in dataset_file.read_text().splitlines():
   r=json.loads(line)
   if r['variant']!='A_original':continue
   text=r['condition_text']
   if r['condition_type']=='judgment_candidate':kind='judgment_candidate'
   elif re.search(r'\d|\b(?:stage|days?|weeks?|months?|years?|HER2|EGFR|BRAF|ALK|KRAS|BRCA|positive|negative)\b',text,re.I):kind='objective_candidate'
   else:kind='other_unreviewed'
   labels[r['id']]=kind
  annotation_file=Path(a.dataset_manifest).parent/'condition_labels.json'
  if annotation_file.exists():
   annotation=json.loads(annotation_file.read_text());labels={k:v['label'] for k,v in annotation['labels'].items()};result['condition_annotation']=annotation['annotation']
  for r in rows:r['condition_type']=labels.get(r['id'],r.get('condition_type'))
  result['condition_label_counts']=dict(Counter(labels[i] for i in groups if i in labels))
  result['condition_label_status']='Exploratory AI semantic labels when annotation file exists; otherwise keyword candidates. Not expert clinical ground truth.'
 # Private audit includes raw completions for 20 source cases. Public report contains counts only.
 audit=[{k:r.get(k) for k in ('id','variant','method','completion','parsed_prediction','prediction','score_prediction','unconstrained_first_token')} for r in rows if 'completion' in r]
 (inp.parent/(inp.stem+'_parse_audit.json')).write_text(json.dumps(audit,indent=2))
 result['parse_audit']={'rows':len(audit),'unparsed':sum(r['parsed_prediction'] is None for r in audit),'generation_vs_option_disagreement':sum(r['parsed_prediction']!=option_prediction(r) for r in audit),'human_review':'pending: private raw audit available; no human review claimed'}
 if phase=='diagnostic':
  buckets=defaultdict(list)
  for r in rows:
   if r['method'] in ('app','signed','negative'):buckets[(r['variant'],r['method'],r['gain'])].append(r)
  curve=[]
  for (variant,method,g),rs in sorted(buckets.items()):
   mass=[np.mean([v['mass'] for v in r['attention']],axis=0) for r in rs]
   curve.append({'variant':variant,'method':method,'gain':g,**metrics(rs),'target_attention_mass':np.mean(mass,axis=0).tolist(),'max_zero_logit_error':max((r.get('zero_max_logit_error') or 0) for r in rs)})
  candidates=[c for c in curve if c['variant'].startswith('B_') and c['method']=='signed']
  selected=max(candidates,key=lambda c:(c['accuracy'],-c['gain']))
  result.update(selected_gain=selected['gain'],gain_selection='20 development cases, signed B option accuracy; tie chooses smaller gain',curve=curve)
  layers=defaultdict(list)
  for r in rows:
   if not r.get('attention'):continue
   for v in r['attention']:
    for target,mass in enumerate(v['mass']):
     count=r['rule_token_counts'][target]
     layers[(r['variant'],r['method'],r['gain'],v['layer'],target)].append((mass,mass/(count/r['input_tokens'])))
  with (out/'attention_layers.csv').open('w') as f:
   w=csv.writer(f);w.writerow(['variant','method','gain','layer','target','mean_mass','mean_mass_relative_uniform'])
   for key,values in sorted(layers.items()):w.writerow([*key,*np.mean(values,axis=0)])
  with (out/'gain_curve.csv').open('w') as f:
   fields=['variant','method','gain','n','accuracy','margin_prior','margin_sibling','target_attention_mass','max_zero_logit_error'];w=csv.DictWriter(f,fields);w.writeheader();w.writerows({k:c[k] for k in fields} for c in curve)
  (out/'gain.json').write_text(json.dumps({'selected_gain':selected['gain'],'source_sha256':hashlib.sha256(inp.read_bytes()).hexdigest(),'selection':result['gain_selection']},indent=2))
  # Assertions concern wiring and exact g=0 identity, not accuracy improvement.
  result['zero_bias_max_error']=max(c['max_zero_logit_error'] for c in curve)
  result['attention_response']={}
  for variant in ('B_applicable_rule_changed','C_sibling_rule_changed'):
   for method,target in (('app',0),('negative',1)):
    cs=[c for c in curve if c['variant']==variant and c['method']==method]
    low=next(c for c in cs if c['gain']==0);high=next(c for c in cs if c['gain']==16)
    result['attention_response'][variant+'/'+method]=high['target_attention_mass'][target]-low['target_attention_mass'][target]
  if result['zero_bias_max_error']>1e-5:raise ValueError('g=0 does not reproduce native option logits')
 else:
  result['development_cases']=len(groups&dev);result['heldout_cases']=len(groups-dev)
  for split in ('all','heldout'):
   selected=[r for r in rows if split=='all' or r['id'] not in dev]
   buckets=defaultdict(list)
   for r in selected:
    if r.get('experiment')=='judgment':continue
    if r.get('assigned_wrong',False):continue
    for stratum in ('all',r['stratum']):buckets[(r['variant'],r['method'],stratum)].append(r)
   for (variant,method,stratum),rs in buckets.items():
    key='/'.join((split,variant,method,stratum));result['metrics'][key]=metrics(rs)
    base=buckets.get((variant,'baseline',stratum))
    if base and method not in ('closed_book','baseline'):result['comparisons'][key]=paired(base,rs)
   # Shared, exact-rate corrupted rule assignments; potential outcomes were evaluated once per case.
   potentials={(r['id'],r['variant'],r['method'],r['assigned_wrong']):r for r in selected if 'assigned_wrong' in r}
   ids=sorted({r['id'] for r in selected});methods=sorted({k[2] for k in potentials})
   for seed in (11,22,33):
    for rate in (0,.1,.2,.3):
     corrupt=corrupt_ids(ids,rate,seed)
     for variant in ('A_original','B_applicable_rule_changed','C_sibling_rule_changed'):
      for method in methods:
       rs=[potentials[i,variant,method,i in corrupt] for i in ids]
       for stratum in ('all','K','U'):
        subset=[r for r in rs if stratum=='all' or r['stratum']==stratum]
        result.setdefault('noise',{})['/'.join(map(str,(split,seed,rate,variant,method,stratum)))]=metrics(subset)
   # Compare noisy signed injection against annotation and pruning on the same assignments.
   for seed in (11,22,33):
    for rate in (0,.1,.2,.3):
     if rate==0 and seed!=11:continue
     corrupt=corrupt_ids(ids,rate,seed)
     for variant,stratum in (('B_applicable_rule_changed','K'),('C_sibling_rule_changed','all')):
      for reference in ('annotate','prune'):
       aa=[potentials[i,variant,reference,i in corrupt] for i in ids]
       bb=[potentials[i,variant,'signed',i in corrupt] for i in ids]
       if stratum=='K':
        aa=[r for r in aa if r['stratum']=='K'];bb=[r for r in bb if r['stratum']=='K']
       if aa:result['comparisons']['/'.join(map(str,('noise',split,seed,rate,variant,stratum,reference,'to_signed')))]=paired(aa,bb)
   judge=defaultdict(dict)
   for r in selected:
    if r.get('experiment')=='judgment':judge[r['id']][r['method']]=r
   for kind in ('all','judgment_candidate','objective_candidate','other_unreviewed'):
    for stratum in ('all','K','U'):
     js=[v for v in judge.values() if (kind=='all' or next(iter(v.values()))['condition_type']==kind) and (stratum=='all' or next(iter(v.values()))['stratum']==stratum)]
     for exposure in ('aligned','conflicting'):
      flips=sum(v['judgment_hidden']['prediction']!=v['judgment_'+exposure]['prediction'] for v in js)
      invalid=sum(any(v[m]['prediction'] is None for m in ('judgment_hidden','judgment_'+exposure)) for v in js)
      valid=[v for v in js if all(v[m]['prediction'] is not None for m in ('judgment_hidden','judgment_'+exposure))]
      valid_flips=sum(v['judgment_hidden']['prediction']!=v['judgment_'+exposure]['prediction'] for v in valid)
      result.setdefault('judgment',{})[split+'/'+kind+'/'+stratum+'/'+exposure]={'n':len(js),'invalid_pairs':invalid,'valid_pairs':len(valid),'flips_valid':valid_flips,'flip_rate_valid':valid_flips/len(valid) if valid else None,'wilson95_valid':wilson(valid_flips,len(valid)),'label_status':'AI research labels/candidates; no expert ground-truth claim','action_label':'original_action' if exposure=='aligned' else 'replacement_action','interpretation':'aligned/conflicting with the observed closed-book choice only for K; not evidence of parameter memory'}
  # Holm adjustment over all reported baseline accuracy comparisons; exploratory families are explicit.
  ordered=sorted(result['comparisons'],key=lambda k:result['comparisons'][k]['mcnemar_exact_p']);last=0
  for j,k in enumerate(ordered):
   last=max(last,min(1.,result['comparisons'][k]['mcnemar_exact_p']*(len(ordered)-j)));result['comparisons'][k]['holm_p']=last
 # Export safe per-case numeric outcomes, no patient/guideline/completion text.
 with gzip.open(out/'scores.jsonl.gz','wt') as f:
  for r in rows:f.write(json.dumps({k:v for k,v in r.items() if k not in ('completion','attention','unconstrained_first_token')})+'\n')
 (out/'summary.json').write_text(json.dumps(result,indent=2))
 text=['ASTER validation: '+phase,'Cases: '+str(len(groups)),'Complete: true','Main accuracy: parsed greedy output; first-position option scores reported separately. Raw completions private.','SEKA spectral positive control is separate; direct bias is not SEKA.']
 if phase=='diagnostic':text+=['Selected gain: '+str(result['selected_gain']),'Zero gain error: '+str(result['zero_bias_max_error']),'Attention response: '+json.dumps(result['attention_response'])]
 else:
  text+=['Heldout cases: '+str(result['heldout_cases'])]
  for k,v in result['metrics'].items():
   if k.startswith('heldout/B_') and k.endswith('/K'):text.append(k+': '+str(v['correct'])+'/'+str(v['n']))
 (out/'REPORT.txt').write_text('\n'.join(text)+'\n')
 print(json.dumps({'phase':phase,'cases':len(groups),'selected_gain':result.get('selected_gain'),'complete':True}))
if __name__=='__main__':main()
