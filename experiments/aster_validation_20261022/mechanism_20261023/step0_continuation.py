"""CPU continuation: source overlap, structural sensitivity, and existing CAD outcomes.

No model calls, no clinical judgments, and no changes to frozen results.
"""
import argparse
import collections
import hashlib
import json
import re
from pathlib import Path
from step0_reanalysis import cluster_ci, read_jsonl, index_method

PAPER = 'https://arxiv.org/html/2505.11613v1'
# Disease nomenclature verified against the original paper, Appendix D/Table 2.
CANCERS = dict(aml='Acute myeloid leukemia', bla='Bladder cancer', bon='Bone cancer',
 bre='Breast cancer', cer='Cervical cancer', cml='Chronic myeloid leukemia',
 cns='Central nervous system cancer', col='Colon cancer',
 eso='Esophageal and esophagastric junction cancer', gas='Gastric cancer',
 hcc='Hepatocellular carcinoma', nscl='Non-small cell lung cancer',
 pancreatic='Pancreatic cancer', prostate='Prostate cancer',
 sclc='Small cell lung cancer', thyroid='Thyroid carcinoma', vulvar='Vulvar cancer')

def sha(path):
 return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def cancer(tree):
 match=re.fullmatch(r'([a-z]+)_?\d+', tree)
 if not match or match[1] not in CANCERS:
  raise ValueError('unmapped tree: '+tree)
 return match[1]

def pair(ids, scores, left, right, disease):
 if not ids:return {'n':0,'delta':None,'tree_ci_descriptive':None}
 differences=[int(scores[i,*right]['correct'])-int(scores[i,*left]['correct']) for i in ids]
 grouped=collections.defaultdict(list)
 for i,d in zip(ids,differences):grouped[cancer(disease[i])].append(d)
 loo={g:{'excluded_n':len(v),'remaining_n':len(ids)-len(v),
  'delta':(sum(differences)-sum(v))/(len(ids)-len(v)) if len(v)<len(ids) else None}
  for g,v in sorted(grouped.items())}
 return dict(n=len(ids),left_correct=sum(scores[i,*left]['correct'] for i in ids),
  right_correct=sum(scores[i,*right]['correct'] for i in ids),
  wins=differences.count(1),losses=differences.count(-1),delta=sum(differences)/len(ids),
  tree_ci_descriptive=cluster_ci(differences,[disease[i] for i in ids]),
  cancer_ci_descriptive=cluster_ci(differences,[cancer(disease[i]) for i in ids]),
  publisher_cluster_count=1,publisher_ci=None,
  leave_one_cancer_out=loo)

def overlap(old,new,field):
 old_set={r[field] for r in old};new_set={r[field] for r in new}
 return dict(old_unique=len(old_set),new_unique=len(new_set),
  common_unique=len(old_set&new_set),new_cases_with_shared_value=sum(r[field] in old_set for r in new),
  common_values=sorted(old_set&new_set))

def clinical_inventory(directory):
 output={}
 for reviewer in (1,2):
  p=directory/f'reviewer{reviewer}.jsonl';rows=read_jsonl(p)
  output[f'reviewer{reviewer}']={'items':len(rows),'sha256':sha(p),
   'items_with_any_rating':sum(any(v is not None for v in r['ratings'].values()) for r in rows),
   'items_with_all_rating_fields':sum(all(v is not None for v in r['ratings'].values()) for r in rows)}
 return {'files':output,'expert_identity_or_qualification_verified':False,
  'clinical_confound_excluded':False,'note':'Filled fields do not establish an authentic expert review.'}

def analyze(structure,scores,cohort800):
 s=index_method(scores);mapping=structure['sources']['case_mapping'];ids=sorted(r['id'] for r in mapping)
 if len(ids)!=200 or len(set(ids))!=200:raise ValueError('expected frozen 200 unique cases')
 if len(cohort800)!=800 or len({r['id'] for r in cohort800})!=800:raise ValueError('expected frozen 800 unique cases')
 disease={r['id']:r['disease'] for r in mapping}
 groups={'all':ids,'single_action_sibling':sorted(r['id'] for r in structure['structure_rows'] if r['sibling_actions_after']==1),
  'multi_action_sibling':sorted(r['id'] for r in structure['structure_rows'] if r['sibling_actions_after']>1)}
 comparisons={'branch_A_minus_D':(('D','branch_visible'),('A','branch_visible')),
  'D_hidden_minus_visible':(('D','visible_prune'),('D','hidden_prune')),
  'A_hidden_minus_visible':(('A','visible_prune'),('A','hidden_prune')),
  'C_hidden_minus_visible':(('C','visible_prune'),('C','hidden_prune'))}
 sensitivity={}
 for group,chosen in groups.items():
  for st in ('all','K','U'):
   chosen_st=[i for i in chosen if st=='all' or s[i,'A','branch_visible']['stratum']==st]
   sensitivity[group+'/'+st]={name:pair(chosen_st,s,*arms,disease) for name,arms in comparisons.items()}
 old=[dict(r,cancer=cancer(r['disease']),publisher='NCCN') for r in mapping]
 new=[{'id':r['id'],'disease':r['disease'],'cancer':cancer(r['disease']),'publisher':'NCCN'} for r in cohort800]
 trees=[]
 for tree,n in sorted(collections.Counter(disease.values()).items()):
  rows=[r for r in mapping if r['disease']==tree]
  hashes={r['tree_path_set_sha256'] for r in rows}
  if len(hashes)!=1:raise ValueError('tree fingerprints disagree')
  trees.append(dict(tree=tree,cases200=n,cancer_code=cancer(tree),cancer_name=CANCERS[cancer(tree)],
   shared_publisher='NCCN',evidence=PAPER,disease_mapping_evidence=PAPER,
   exact_document=None,version=None,page=None,independence_verified=False,
   tree_path_set_sha256=next(iter(hashes))))
 cad=structure['cad']['cases'];cad_groups={}
 for outcome in sorted({r['outcome'] for r in cad}):
  rows=[r for r in cad if r['outcome']==outcome]
  if not rows:continue
  cad_groups[outcome]={'n':len(rows),'first_token_changed':sum(r['first_token_changed'] for r in rows),
   'old_first_token_gold':sum(r['old_first_token_is_gold'] for r in rows),
   'first_step_gold_certainly_masked':sum(r['first_step_gold_certainly_masked_from_option_lower_bound'] for r in rows),
   'old_length_stop':sum(r['old_stop_reason']=='length' for r in rows),
   'filtered_length_stop':sum(r['filtered_stop_reason']=='length' for r in rows),
   'old_parse_failure':sum(r['old_parse_failure'] for r in rows),
   'filtered_parse_failure':sum(r['filtered_parse_failure'] for r in rows)}
 return {'analysis':'retrospective descriptive sensitivity; original tests unchanged','new_model_calls':0,
  'source_mapping':trees,'overlap200_800':{f:overlap(old,new,f) for f in ('id','disease','cancer','publisher')},
  'cancer_counts200':dict(sorted(collections.Counter(r['cancer'] for r in old).items())),
  'sensitivity':sensitivity,'cad_existing_evidence':cad_groups,
  'cad_outcome_labels':dict(collections.Counter(r['outcome'] for r in cad)),
  'cad_full_trajectories_available':False,'clinical_reasonableness_adjudicated':False,
  'limitations':['Cancer categories are verified nomenclature, not independent guideline documents.',
   'All records share NCCN publisher; one-cluster interval is undefined, not zero.',
   'Publisher grouping is a shared-origin sensitivity, not proof of perfect statistical dependence.',
   'No document/version/page mapping; no independent-source confirmation claim.',
   'Structure strata do not establish clinical plausibility or effect modification.',
   'Existing free text cannot reliably recover original token IDs, masks or per-step probabilities.']}

def main():
 p=argparse.ArgumentParser();p.add_argument('--cohort800',type=Path,required=True);p.add_argument('--clinical',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 root=Path(__file__).parent/'results';sp=root/'step0_review/private_structure_summary.json';fp=root/'priority200/qwen/full_parser_v2/scores.jsonl'
 result=analyze(json.loads(sp.read_text()),read_jsonl(fp),read_jsonl(a.cohort800));result['clinical_inventory']=clinical_inventory(a.clinical)
 result['input_sha256']={str(x):sha(x) for x in (sp,fp,a.cohort800,Path(__file__))}
 a.output.mkdir(parents=True,exist_ok=False);(a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'new_model_calls':0,'overlap':result['overlap200_800'],'cad':result['cad_existing_evidence']}))
if __name__=='__main__':main()
