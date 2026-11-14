"""CPU-only census of frozen pair structure, provenance, duplicates and source format."""
from pathlib import Path
import json,collections,re,difflib,csv,sys,os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4'
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
SRC=Path('/home/gcx/aster-cr-retrain-paired-v2-20261111')
PARENT=SRC/'reports/aster_cr_multisource_20261112'
BAL=PARENT/'balanced_10006'
OUT=Path(__file__).resolve().parents[1]
PRIVATE=Path('/home/gcx/aster-cr-medeinst-alignment-audit-private-20261112')
def rows(p):return [json.loads(x) for x in Path(p).open() if x.strip()]
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvwrite(p,rs):
 with Path(p).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def norm(s):return ' '.join(s.split())
def stats(a):
 a=np.asarray(a,float)
 return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),q05=float(np.quantile(a,.05)),q95=float(np.quantile(a,.95)),min=float(a.min()),max=float(a.max()))
med=rows(BAL/'data/medeinst_train_selected.jsonl')
fit=rows(BAL/'data/fit_pairs.jsonl');fm={r['pair_id']:r for r in fit}
raw=rows(PARENT/'raw/medeinst_train.jsonl');test=rows(PARENT/'raw/medeinst_test.jsonl')
rg=collections.defaultdict(list);tg=collections.defaultdict(list)
for r in raw:rg[r['case_id']].append(r)
for r in test:tg[r['case_id']].append(r)
inputs={r['view_id']:r for r in rows(BAL/'data/extra_inputs.jsonl')}
test_text={norm(r['narrative']) for r in test}
pairkeys=collections.defaultdict(list);endkeys=collections.defaultdict(set)
for p in med:
 pairkeys[(norm(p['original_input']),norm(p['edited_input']))].append(p['pair_id'])
 for k in ['original_input','edited_input']:endkeys[norm(p[k])].add(p['pair_id'])
program=[]
for p in med:
 f=fm[p['pair_id']];rr=rg[p['case_id']];by={r['case_type']:r for r in rr}
 a,b=inputs[f['left']],inputs[f['right']]
 cand=sorted([p['label_before'],p['label_after']])
 def expected(t):
  return 'Clinical vignette:\n'+t+'\n\nDifferential diagnosis task. Choose the better-supported diagnosis from the two candidates.\n'+'是: '+cand[0]+'\n否: '+cand[1]+'\nReturn only 是 or 否.\nAnswer:'
 aa=[x.strip() for x in p['original_input'].splitlines() if x.strip()];bb=[x.strip() for x in p['edited_input'].splitlines() if x.strip()]
 changes=[(t,i,j,k,l) for t,i,j,k,l in difflib.SequenceMatcher(None,aa,bb,autojunk=False).get_opcodes() if t!='equal']
 metadata_same=all(by['control'][k]==by['trap'][k] for k in ['age','sex'])
 program.append(dict(pair_id=p['pair_id'],case_id=p['case_id'],source_id=p['source_id'],
  raw_pair_count=len(rr),raw_narratives_match=by['control']['narrative']==p['original_input'] and by['trap']['narrative']==p['edited_input'],
  raw_labels_match=by['control']['ground_truth']==p['label_before'] and by['trap']['ground_truth']==p['label_after'],
  diagnosis_before=p['label_before'],diagnosis_after=p['label_after'],left_binary_label=f['left_label'],right_binary_label=f['right_label'],
  labels_flip=f['left_label']!=f['right_label'],candidate_space_same=a.get('diagnosis_candidates')==b.get('diagnosis_candidates')==cand,
  prompt_exact=expected(p['original_input'])==a['prompt'] and expected(p['edited_input'])==b['prompt'],
  label_mapping_correct=f['left_label']==int(p['label_before']==cand[0]) and f['right_label']==int(p['label_after']==cand[0]),
  target_same_within_pair=True,target_megaf_equivalent=False,age_sex_metadata_same=metadata_same,
  changed_text_blocks=len(changes),removed_nonempty_lines=sum(j-i for _,i,j,k,l in changes),added_nonempty_lines=sum(l-k for _,i,j,k,l in changes),
  prior_qwen_clinical_factor_count=p.get('clinical_factors_changed'),
  prior_qwen_count_is_not_this_audit=True,clinical_fact_count_status='requires_semantic_review',
  normalized_text_changed=norm(p['original_input'])!=norm(p['edited_input']),
  exact_pair_duplicate_other_count=len(pairkeys[(norm(p['original_input']),norm(p['edited_input']))])-1,
  exact_endpoint_duplicate_other_pair_count=len((endkeys[norm(p['original_input'])]|endkeys[norm(p['edited_input'])])-{p['pair_id']}),
  case_id_overlap_test=p['case_id'] in tg,exact_endpoint_overlap_test=norm(p['original_input']) in test_text or norm(p['edited_input']) in test_text,
  ddxplus_original_id_mapping='unavailable',original_label_rationale='released_label_only; patient-specific DDXPlus probabilities and generation evidence unavailable'))
csvwrite(OUT/'program_audit.csv',program)
print('census_done',flush=True)
full=rows('/home/gcx/aster_cr_pair_dataset_20261111/full_pool_train.jsonl')
mf=[p for p in full if p['kind']=='F']
mp=Path('/home/gcx/TRIAGE/data/MEGA-CDP/train_val')
v={(int(r['guideline_id']),int(r['path_id'])):r for r in rows(mp/'vignette_cases.jsonl')}
trees={int(r['guideline_id']):json.loads(r['text']) for r in rows(mp/'decision_trees.jsonl')}
def walk(x):
 if isinstance(x,dict):
  if 'index' in x and 'description' in x:yield str(x['index']),x['description']
  for z in x.values():yield from walk(z)
 elif isinstance(x,list):
  for z in x:yield from walk(z)
tn={g:dict(walk(t)) for g,t in trees.items()}
mega=[]
for p in mf:
 a,b=p['original'],p['modified'];key=(int(p['guideline_id']),int(p['case_id'].split(':')[-1]))
 mega.append(dict(pair_id=p['pair_id'],case_id=p['case_id'],source_id=p['source_id'],
 original_case_found=key in v,raw_patient_matches=norm(p.get('raw_patient',a['patient']))==norm(v.get(key,{}).get('text','')),
 original_condition_matches_tree=norm(a['condition'])==norm(tn.get(key[0],{}).get(str(p['raw_node_index']),'')),
 guideline_exists=Path(p['source_document']).exists(),condition_same=a['condition']==b['condition'],guide_same=a['guide']==b['guide'],
 definitions_same=a['definitions']==b['definitions'],patient_changed=a['patient']!=b['patient'],label_flip=a['label']!=b['label'],
 patient_only_edit_components=all(e['component']=='patient' for e in p['edits']),
 explicit_program_rule=bool(p.get('program_rule')),source_original_label=a['label'],source_modified_label=b['label'],
 origin=p['origin'],language=p['language'],char_before=len(a['prompt']),char_after=len(b['prompt'])))
csvwrite(OUT/'mega_f_program_audit.csv',mega)
print('mega_done',flush=True)
# Pair-level similarity candidates: both narratives in order. This is a retrieval flag, not same-patient proof.
valid_test=[]
for cid,rr in sorted(tg.items()):
 d={r['case_type']:r for r in rr}
 if len(rr)==2 and set(d)=={'control','trap'}:valid_test.append((cid,d['control']['narrative'],d['trap']['narrative']))
traintexts=[norm(p['original_input'])+' PAIR_SIDE_BOUNDARY '+norm(p['edited_input']) for p in med]
testtexts=[norm(a)+' PAIR_SIDE_BOUNDARY '+norm(b) for _,a,b in valid_test]
vec=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=2,max_features=100000,dtype=np.float32)
x=vec.fit_transform(traintexts+testtexts);xtr=x[:len(med)];xte=x[len(med):]
nn=NearestNeighbors(n_neighbors=2,metric='cosine',algorithm='brute',n_jobs=4).fit(xtr)
d,i=nn.kneighbors(xtr)
nt=NearestNeighbors(n_neighbors=1,metric='cosine',algorithm='brute',n_jobs=4).fit(xte)
dt,it=nt.kneighbors(xtr)
near=[]
for j,p in enumerate(med):
 q=next(k for k in range(2) if i[j,k]!=j)
 near.append(dict(pair_id=p['pair_id'],nearest_train_pair=med[int(i[j,q])]['pair_id'],train_cosine=float(1-d[j,q]),
  nearest_test_case=valid_test[int(it[j,0])][0],test_cosine=float(1-dt[j,0])))
csvwrite(OUT/'near_duplicate_candidates.csv',near)
counts={k:sum(bool(p[k]) for p in program) for k in ['raw_narratives_match','raw_labels_match','labels_flip','candidate_space_same','prompt_exact','label_mapping_correct','age_sex_metadata_same','normalized_text_changed','case_id_overlap_test','exact_endpoint_overlap_test']}
megac={k:sum(bool(p[k]) for p in mega) for k in ['original_case_found','raw_patient_matches','original_condition_matches_tree','guideline_exists','condition_same','guide_same','definitions_same','patient_changed','label_flip','patient_only_edit_components','explicit_program_rule']}
train_sources={p['source_id'] for p in fit if p['dataset_id']=='MEGA-CDP'}
train_cases={p['case_id'] for p in fit if p['dataset_id']=='MEGA-CDP'}
evalmeta=[]
base=SRC/'reports/aster_cr_retrain_paired_v2_20261111/data'
for filename in ['historical_views.jsonl','new60_views.jsonl','path_interference_views.jsonl','path_features.jsonl']:
 for r in rows(base/filename):
  if r.get('split') in ['dev','test','new60']:evalmeta.append(r)
es={p['source_id'] for p in evalmeta if p.get('source_id')};ec={p['case_id'] for p in evalmeta if p.get('case_id')}
summary=dict(census_size=len(med),unique_med_cases=len({p['case_id'] for p in med}),frozen_count=len(fit),
 class_source_counts=dict(collections.Counter(p['kind']+'|'+p['dataset_id'] for p in fit)),checks=counts,
 original_train_pairs=len(rg),original_train_rows=len(raw),original_test_pairs=len(tg),original_test_rows=len(test),
 raw_train_test_case_overlap=len(set(rg)&set(tg)),selected_diagnosis_transition_count=len({(p['label_before'],p['label_after']) for p in med}),
 prior_qwen_factor_counts=dict(collections.Counter(str(p.get('clinical_factors_changed')) for p in med)),
 text_block_counts=dict(collections.Counter(str(p['changed_text_blocks']) for p in program)),
 exact_duplicate_pair_groups=sum(len(v)>1 for v in pairkeys.values()),
 exact_duplicate_endpoint_groups=sum(len(v)>1 for v in endkeys.values()),
 selected_pairs_with_repeated_endpoint=sum(p['exact_endpoint_duplicate_other_pair_count']>0 for p in program),
 near_duplicate_method='char 3-5 TFIDF, min_df=2, max_features=100000; concatenated normalized original and edited; nearest other pair. Similarity signals synthetic template similarity, not proven same patient/leakage.',
 near_duplicate_thresholds={str(t):dict(train_pairs_with_neighbor=sum(p['train_cosine']>=t for p in near),train_pairs_with_test_neighbor=sum(p['test_cosine']>=t for p in near)) for t in [.95,.98]},
 near_train_cosine=stats([p['train_cosine'] for p in near]),near_test_cosine=stats([p['test_cosine'] for p in near]),
 mega_F=dict(n=len(mf),cases=len({p['case_id'] for p in mf}),sources=len({p['source_id'] for p in mf}),checks=megac,language=dict(collections.Counter(p['language'] for p in mf)),origin=dict(collections.Counter(p['origin'] for p in mf))),
 leakage_scope=dict(mega_eval_metadata_rows=len(evalmeta),mega_train_eval_source_overlap=sorted(train_sources&es),mega_train_eval_case_overlap=sorted(train_cases&ec),ddxplus_underlying_patient_overlap='not verifiable without mapping',base_model_pretraining_contamination='not investigated; not inferable from record IDs'))
write(OUT/'program_summary.json',summary)
print(json.dumps({'status':'complete','checks':counts,'near_duplicates':summary['near_duplicate_thresholds'],'mega_F':summary['mega_F']},ensure_ascii=False))
