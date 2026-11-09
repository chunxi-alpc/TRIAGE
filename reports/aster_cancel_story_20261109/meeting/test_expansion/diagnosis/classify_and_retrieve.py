
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
import json,csv,re,math
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
OUT=Path('/home/gcx/aster_next_20261109/diagnosis')
ROOT=Path('/home/gcx/aster_cr_repair_v2_run_20261108')
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvwrite(p,rr):
 keys=list(dict.fromkeys(k for r in rr for k in r))
 with p.open('w',newline='',encoding='utf-8') as h:
  w=csv.DictWriter(h,fieldnames=keys);w.writeheader()
  for r in rr:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def sentences(text):
 # Keep exact substrings; do not normalize numbers, units, negation or time.
 parts=[s.strip() for s in re.split(r'(?<=[。！？!?；;])|\n+|(?<=[.])\s+(?=[A-Z])',text) if s.strip()]
 return parts or [text]
STOP={'the','a','an','is','are','was','were','be','been','does','do','did','has','have','had','of','to','in','at','for','with','and','or','there','it','its','that','this','such','as','by','on','from','than','patient','individual','person','whether','currently'}
def terms(text):
 text=text.lower().replace('\\u202f',' ').replace('\u202f',' ')
 en=re.findall(r'[a-z]+(?:-[a-z]+)*|\d+(?:\.\d+)?',text)
 content=[s for s in en if s not in STOP]
 tokens=content+['bi:'+content[i]+'_'+content[i+1] for i in range(len(content)-1)]
 for run in re.findall(r'[\u4e00-\u9fff]+',text):
  tokens.extend('zh:'+run[i:i+2] for i in range(max(1,len(run)-1)))
 return tokens
def retrieve(condition,patient,topk=2):
 ss=sentences(patient)
 v=TfidfVectorizer(analyzer=terms,norm='l2',sublinear_tf=True)
 mat=v.fit_transform(ss+[condition]);score=np.asarray((mat[:-1]@mat[-1].T).toarray()).reshape(-1)
 order=sorted(range(len(ss)),key=lambda j:(-score[j],j))[:topk]
 return [dict(rank=i+1,sentence_index=j,quote=ss[j],matching_score=float(score[j])) for i,j in enumerate(order)]
def support_prefix(condition,selected):
 return 'Current condition (verbatim):\n'+condition+'\n\nCandidate patient evidence (verbatim; verify variable, time point, unit and negation before judging):\n'+'\n'.join('['+str(s['rank'])+'] '+s['quote'] for s in selected)+'\n\nUse the complete guideline and original patient text below to make the final judgment. The excerpts are retrieval candidates, not answers.\n'
d=json.loads((OUT/'diagnosis.json').read_text());ex=json.loads((OUT/'error_examples.json').read_text())
rawfirst=[r for r in ex if r['first_error_for']['raw']]
partition=Counter()
for e in rawfirst:
 if e['raw_wrong_with_correct_AP_endpoint']:partition['same_evidence_AP_prediction_change_observed']+=1
 elif abs(e['raw_score'])<=3:partition['near_threshold_without_observed_AP_repair']+=1
 else:partition['confident_error_without_observed_AP_repair']+=1
# One dominant family, supported by literal variable/value matches only.
pma=[];retrieval_checks=[]
for e in ex:
 selected=retrieve(e['condition'],e['patient'])
 if e['first_error_for']['raw']:
  row=dict(case_id=e['case_id'],source_id=e['source_id'],gid=e['gid'],condition=e['condition'],
   raw_score=e['raw_score'],gold_label=e['gold_label'],evidence_candidates=selected)
  if e['source_id']=='train_val:2072' and e['gid']=='G2':
   normalized=e['patient'].replace('\\u202f',' ').replace('\u202f',' ')
   matches=list(re.finditer(r'postmenstrual age\s*(?:was|of|is)?\s*(\d+)\s*\+\s*(\d+)\s*weeks',normalized,flags=re.I))
   if len(matches)!=1:raise RuntimeError('ambiguous explicit PMA match '+e['case_id'])
   q=matches[0];week=int(q.group(1));day=int(q.group(2))
   # This is a literal text comparison against the condition's numeric threshold;
   # it does not infer any diagnosis or rewrite the original label.
   row.update(explicit_PMA_weeks=week,explicit_PMA_days=day,condition_threshold_weeks=31,
    literal_numeric_comparison=week+day/7>=31,
    top1_contains_PMA=bool(re.search(r'postmenstrual age',selected[0]['quote'],re.I)),
    top2_contains_PMA=any(re.search(r'postmenstrual age',s['quote'],re.I) for s in selected))
   pma.append(row)
  retrieval_checks.append(row)
# Specific inspected examples; descriptions refer directly to quoted text.
example_specs=[
 ('mega:train_val:2072:1','G2','variable_and_time_alignment',
  'at a postmenstrual age of 35+0 weeks.',
  'The condition asks for PMA at the initial examination; the patient explicitly gives 35+0 weeks. Birth gestational age29+4 refers to another variable/time. This supports trying explicit variable/time alignment, without proving the model internal cause.'),
 ('mega:train_val:379:5','G4','near_threshold_with_explicit_procedural_statement',
  '随后在除颤后未进行节律或脉搏重新评估即立即不间断继续心肺复苏。',
  'The patient directly states immediate uninterrupted CPR after shock. The raw wrong score is -2.886 and the old linear correction approaches but does not cross zero.'),
 ('mega:train_val:918:7','G10','explicit_fact_missed_in_current_score',
  'however, the patient did meet one major criterion and also fulfilled at least two minor criteria',
  'The condition requires at least one major criterion; this exact clause directly states one. All four cached scores remain negative.'),
 ('mega:train_val:2612:13','G25','numeric_variable_alignment_with_AP_sensitivity',
  'while pravastatin was planned at 80 mg/day.',
  'The condition asks whether the planned pravastatin dose exceeds40mg/day; the literal dose is80. An A/P variant also changes the prediction, so both evidence alignment and interference are observable here.'),
 ('mega:train_val:1475:10','G0','reference_text_mismatch_requires_review',
  'the finding was not consistent with a disease-causing gene for a monogenic disorder',
  'The reference label is yes for a causative gene, while this clause rejects a disease-causing monogenic finding. Flag for source/reference review; do not count it as proven evidence-reading failure.'),
 ('mega:train_val:1322:15','G28','temporal_status_requires_review',
  '临床上曾一度怀疑存在深静脉血栓形成或肺栓塞的可能，但随后已通过客观检测将DVT/PE诊断排除。',
  'The condition asks whether DVT/PE is suspected; the patient states prior suspicion followed by exclusion. The intended time point needs review.'),
 ('mega:train_val:2731:17','G37','explicit_text_contradiction_requires_review',
  'There were no features of an active clinical phenotype, with no ongoing stone disease, nephrocalcinosis, or renal impairment documented, and it was also not the case that there was no ongoing clinical disease.',
  'The same sentence asserts no ongoing manifestations and then negates no ongoing disease. Flag the input/reference wording before attributing the error to the model.'),
 ('mega:train_val:918:15','G19','editor_introduced_near_threshold_error',
  'this testing did not show an absence of pathogenic variants across known genes.',
  'The original score -0.832 is correct against the reference; all three editors flip it. The centered learned correction, not a raw-score-only margin shift, is the material contributor.')
]
specific=[]
for cid,gid,tag,quote,note in example_specs:
 e=next(e for e in ex if e['case_id']==cid and e['gid']==gid)
 assert quote in e['patient'],(cid,quote)
 specific.append(dict(case_id=cid,source_id=e['source_id'],gid=gid,category=tag,condition=e['condition'],
  quote=quote,note=note,gold_label=e['gold_label'],
  scores={n:e[n] for n in ['raw_score','old_linear_score','balanced_score','strict_score']},
  observed_AP_pairs=e['observed_AP_pairs'],AP_prediction_changing_pairs=e['raw_AP_prediction_changing_pairs']))
fitviews={v['view_id']:v for v in rows(ROOT/'data/views.jsonl') if v['split']=='fit'}
fit_patient=fitviews['new-388-G15:full']['prompt']
quote='血清总IgE并未低于 1000 U/mL（仍为 1280 U/mL），因此不符合'
assert quote in fit_patient
specific.append(dict(view_id='new-388-G15:full',split='fit',category='reference_text_mismatch_requires_review',
 condition='患者是否符合真菌致敏且血清总IgE<1000 U/ml的严重（难治性）哮喘表现',reference_label=1,
 quote=quote,note='The saved label is yes, but1280 is not below1000 and the text explicitly says the condition is not met. This is a direct arithmetic/reference conflict requiring review. No label changed here.'))
# Pure-CPU retrieval prototype on the available fit/dev natural nodes.
allnodes=[n for n in rows(ROOT/'data/original_nodes.jsonl') if n['split'] in ('fit','dev')]
prefixrows=[]
for n in allnodes:
 guide=n['guideline'];group=next(g for g in guide['groups'] if g['group_id']==n['gid'])
 condition=re.sub(r'（判断为[是否]）$','',group['rules'][0]['condition'])
 selected=retrieve(condition,n['patient'])
 assert all(s['quote'] in n['patient'] for s in selected)
 prefixrows.append(dict(group_id=n['group_id'],case_id=n['case_id'],source_id=n['source_id'],split=n['split'],gid=n['gid'],
  condition=condition,evidence=selected,support_prefix=support_prefix(condition,selected)))
dump(OUT/'supported_error_categories.json',dict(protocol='Only fit/dev; fixed models and labels unchanged',
 raw_first_error_partition=dict(partition),
 dominant_family=dict(source_id='train_val:2072',gid='G2',error_paths=len(pma),share_of_raw_error_paths=len(pma)/36,
  question='PMA at initial examination >=31+0weeks',explicit_PMA_min_weeks=min(r['explicit_PMA_weeks']+r['explicit_PMA_days']/7 for r in pma),
  explicit_PMA_max_weeks=max(r['explicit_PMA_weeks']+r['explicit_PMA_days']/7 for r in pma),
  all_literal_values_meet_threshold=all(r['literal_numeric_comparison'] for r in pma)),
 specific_examples=specific,
 smallest_supported_next_step=dict(name='verbatim current-condition evidence prefix',
  operation='Before the unchanged full input, repeat the current condition and prepend the two patient sentences selected by fixed lexical matching. Preserve variable names, time points, units and negation verbatim. Do not infer or insert the answer.',
  fit_dev_CPU_check=dict(natural_nodes=len(prefixrows),fit=sum(r['split']=='fit' for r in prefixrows),dev=sum(r['split']=='dev' for r in prefixrows),
   all_quotes_exact_substrings=True,PMA_error_top1_retrieval=sum(r['top1_contains_PMA'] for r in pma),
   PMA_error_top2_retrieval=sum(r['top2_contains_PMA'] for r in pma),PMA_error_total=len(pma),
   classification_or_path_gain_measured=False,backbone_forwards=0),
  caution='This verifies evidence retrieval only. It does not establish answer/path improvement. Evaluate a fixed candidate on fit/dev before any separate future confirmation; do not adjust the already frozen expanded test comparison.'),
 categories_overlap_note='The raw partition is mutually exclusive and totals36 paths. Inspected examples/semantic flags are evidence notes, may overlap, and are not automatic causal labels.'))
dump(OUT/'retrieval_cpu_check.json',dict(protocol='fit/dev only, no model inference',backbone_forwards=0,
 raw_first_error_retrieval=retrieval_checks,
 proposed_prefix_examples=prefixrows[:4],
 totals=dict(natural_fit_dev_nodes=len(prefixrows),all_quotes_verbatim=True,PMA_top1=sum(r['top1_contains_PMA'] for r in pma),
 PMA_top2=sum(r['top2_contains_PMA'] for r in pma),PMA_total=len(pma))))
csvwrite(OUT/'pma_numeric_evidence.csv',pma)
csvwrite(OUT/'supported_examples.csv',specific)
with (OUT/'candidate_evidence_prefixes.fit_dev.jsonl').open('w',encoding='utf-8') as f:
 for r in prefixrows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
print(json.dumps(dict(partition=dict(partition),PMA_cases=len(pma),PMA_top1=sum(r['top1_contains_PMA'] for r in pma),PMA_top2=sum(r['top2_contains_PMA'] for r in pma),prefixes=len(prefixrows),examples=len(specific)),ensure_ascii=True))

