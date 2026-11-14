"""Aggregate completed blinded AI review; no patient narratives are exported."""
from pathlib import Path
import json,csv,collections,math,datetime
import numpy as np
from scipy.stats import hypergeom
OUT=Path(__file__).resolve().parents[1]
PRIVATE=Path('/home/gcx/aster-cr-medeinst-alignment-audit-private-20261112')
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).open() if s.strip()]
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvrows(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig')))
def csvwrite(p,rs):
 with Path(p).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,list(rs[0]));w.writeheader();w.writerows(rs)
manifest=read(OUT/'sampling_manifest.json');samples=manifest['samples'];sm={r['review_id']:r for r in samples}
review=[];reviewer_counts={}
for who in [1,2,3]:
 finals=[]
 for phase in ['pilot','main']:
  f=PRIVATE/f'reviewer_{who}_{phase}_final.jsonl'
  rr=rows(f)
  expected=20 if phase=='pilot' else 180
  if len(rr)!=expected:raise ValueError(f'Reviewer {who} {phase}: {len(rr)}/{expected}; unfinished, not published')
  blind=rows(PRIVATE/(f'reviewer_{who}_pilot_blind_results_v2.jsonl' if phase=='pilot' else f'reviewer_{who}_main_blind_results.jsonl'))
  bm={r['review_id']:r for r in blind}
  for r in rr:
   assert r['review_id'] in sm and sm[r['review_id']]['reviewer']==who
   assert r['final_category'] in ['S1','S2','S3','S4']
   assert r['pair_id']==sm[r['review_id']]['pair_id']
   for k in ['left_choice','right_choice','provisional_category','fact_count','reason']:
    assert r[k]==bm[r['review_id']][k],(r['review_id'],k,'blind field altered after reveal')
   r['reviewer']=who;r['review_phase']=phase;r['review_model_requested']='gpt-6-astra';r['reasoning_requested']='xhigh'
   finals.append(r)
 review+=finals
 reviewer_counts[str(who)]=dict(n=len(finals),classes=dict(collections.Counter(r['final_category'] for r in finals)))
assert len(review)==600 and len({r['review_id'] for r in review})==600
byid={r['pair_id']:r for r in review}
program=csvrows(OUT/'program_audit.csv')
out=[]
for p in program:
 r=byid.get(p['pair_id'])
 extra=dict(review_id=r['review_id'] if r else '',semantic_reviewed=bool(r),category=r['final_category'] if r else 'S4',
 classification_basis='blinded_AI_review_then_label_reveal' if r else 'administrative_unknown_not_semantically_reviewed',
 reason=r['reason']+' '+r['final_reason'] if r else '未进入600对分层语义样本；S4仅表示尚未审查，不是已发现医学歧义，也不是临床质量判定。',
 confidence=r['confidence'] if r else 0,clinical_fact_count=r['fact_count'] if r else '',
 changed_facts=json.dumps(r['changed_facts'],ensure_ascii=False) if r else '',
 issues=json.dumps(r['issues'],ensure_ascii=False) if r else '["NOT_SEMANTICALLY_REVIEWED","DDXPLUS_LABEL_RATIONALE_UNAVAILABLE"]',
 left_blind_choice=r['left_choice'] if r else '',right_blind_choice=r['right_choice'] if r else '',
 change_sufficient=r['change_sufficient'] if r else '',label_check=json.dumps(r.get('label_check'),ensure_ascii=False) if r else '',
 needs_expert=r.get('needs_expert') if r else '',reviewer=r['reviewer'] if r else '',unrelated_change=r.get('unrelated_change') if r else '')
 out.append(dict(p,**extra))
csvwrite(OUT/'pair_audit.csv',out)
(OUT/'semantic_review.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in sorted(review,key=lambda r:r['review_id'])))
strata=manifest['strata'];N=4368;H=sum(s['sample_size']<s['population'] for s in strata)
rsby=collections.defaultdict(list)
for r in review:rsby[sm[r['review_id']]['stratum_id']].append(r)
est={};stratum_out=[]
for cat in ['S1','S2','S3','S4']:
 phat=var=0.;loK=hiK=0
 for s in strata:
  Nh=s['population'];nh=s['sample_size'];rr=rsby[s['stratum_id']]
  assert len(rr)==nh
  x=sum(r['final_category']==cat for r in rr);p=x/nh;W=Nh/N;phat+=W*p
  sh2=p*(1-p)*nh/(nh-1) if nh>1 else 0.
  var+=W*W*(1-nh/Nh)*sh2/nh
  if nh==Nh:L=U=x
  else:
   kval=np.arange(x,Nh-nh+x+1)
   a=.05/(2*H)
   ok=(hypergeom.sf(x-1,Nh,kval,nh)>=a)&(hypergeom.cdf(x,Nh,kval,nh)>=a)
   L=int(kval[ok].min());U=int(kval[ok].max())
  loK+=L;hiK+=U
  stratum_out.append(dict(category=cat,stratum_id=s['stratum_id'],population=Nh,n=nh,count=x,design_weight=Nh/nh))
 se=math.sqrt(var)
 if 0<phat<1 and se>0:
  l=math.log(phat/(1-phat));t=1.95996398454*se/(phat*(1-phat))
  approx=[1/(1+math.exp(-(l-t))),1/(1+math.exp(-(l+t)))]
 else:approx=None
 exact=[loK/N,hiK/N]
 if cat=='S1':
  exact=[0.,0.];approx=[0.,0.]
 est[cat]=dict(sample_count=sum(r['final_category']==cat for r in review),sample_fraction=sum(r['final_category']==cat for r in review)/600,
 population_estimate=phat,estimated_count=phat*N,design_standard_error=se,approximate_95ci_logit=approx,
 conservative_95ci_hypergeometric_bonferroni=exact,interval_note='S1 zero established by full task/prompt census' if cat=='S1' else 'approximate design-based logit CI primary; conservative finite-population simultaneous stratum bounds also supplied')
csvwrite(OUT/'stratum_estimates.csv',stratum_out)
aux=[dict(review_id=r['review_id'],pair_id=r['pair_id'],category=r['final_category'],confidence=r['confidence'],reason=r['reason']) for r in review if r['final_category'] in ['S1','S2']]
write(OUT/'audited_auxiliary_candidates.json',dict(n=len(aux),strict_S1=sum(r['category']=='S1' for r in aux),auxiliary_S2=sum(r['category']=='S2' for r in aux),
 status='candidates for separately marked auxiliary experiment only; no training/config change',pairs=aux))
scores={r['pair_id']:r for r in csvrows(OUT/'pair_score_diagnostics.csv')}
for r in aux:
 r['base_direction_matches_released_labels']=scores[r['pair_id']]['original_direction_correct']=='True'
 r['base_both_labels_correct']=scores[r['pair_id']]['original_both_correct']=='True'
write(OUT/'audited_auxiliary_candidates.json',dict(n=len(aux),strict_S1=sum(r['category']=='S1' for r in aux),auxiliary_S2=sum(r['category']=='S2' for r in aux),
 status='candidates for separately marked auxiliary experiment only; no training/config change',pairs=aux))
prog=read(OUT/'program_summary.json');diag=read(OUT/'source_diagnostics.json');overlap=read(OUT/'template_overlap_summary.json')
def weight(r):s=sm[r['review_id']];return s['N_h']/s['n_h']/N
multi=[r for r in review if isinstance(r['fact_count'],(int,float)) and r['fact_count']>=2]
zero=[r for r in review if r['fact_count']==0]
summary=dict(audit_date='2026-11-12',completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 decision='从核心训练中移除 MedEinst F，改用 MEGA 内部 F',
 review_scope=dict(program_census=4368,semantic_sample=600,pilot_included=60,not_semantically_reviewed=3768,clinical_expert_reviews=0,
 model_requested='Astra Extra High',backend_requested='gpt-6-astra',reasoning_requested='xhigh',pro_entitlement='not exposed',exact_backend_snapshot='not exposed',
 independent_contexts=3,reviewers_per_pair=1,inter_rater_agreement='not estimable: disjoint assignments, not duplicate ratings'),
 strict_S1_census=dict(n=4368,count=0,reason='all actual prompts are fixed two-candidate differential diagnosis, not criterion truth-value tasks'),
 semantic_sample_counts={k:sum(r['final_category']==k for r in review) for k in ['S1','S2','S3','S4']},semantic_population_estimates=est,
 administrative_all_pair_counts={k:sum(r['category']==k for r in out) for k in ['S1','S2','S3','S4']},
 administrative_count_warning='Unreviewed 3768 are administrative S4 only; never interpret these counts as the population medical quality prevalence.',
 reviewers=reviewer_counts,
 issue_statistics=dict(sample_multifact=len(multi),estimated_multifact_fraction=sum(weight(r) for r in multi),
 sample_no_new_fact=len(zero),estimated_no_new_fact_fraction=sum(weight(r) for r in zero),
 sample_needs_expert=sum(bool(r['needs_expert']) for r in review),
 sample_unrelated_change_true=sum(r.get('unrelated_change') is True for r in review)),
 source_distribution=prog['class_source_counts'],program=prog,template_overlap=overlap,source_diagnostics=diag,
 sampling=dict(seed=20261112,strata=148,n=600,design='stratified SRS without replacement; case unique; at least 2 per stratum unless census',
 interval='finite-population stratified variance with logit transformation; additionally exact hypergeometric Bonferroni bounds across noncensus strata',
 ci_scope='sampling uncertainty conditional on these AI classifications; excludes clinical-review error and shared model bias'),
 limitations=['No complete semantic census of 4368 pairs.','No clinical expert adjudication.','No patient-level DDXPlus mapping or per-record generation label evidence.','Model source probes reuse representations fitted on all original training data.','Existing evaluation is reused and MEGA test F only has 7 pairs.'],
 artifacts=dict(pair_audit='pair_audit.csv',manifest='sampling_manifest.json',report='REPORT.zh.md',examples='examples.md',experiment_plan='experiment_plan.zh.md'))
write(OUT/'summary.json',summary)
manifest['completed_semantic_reviews']=600;manifest['reviewer_counts']=reviewer_counts;manifest['rubric_version']='v2 frozen after blinded pilot, before label reveal';write(OUT/'sampling_manifest.json',manifest)
print(json.dumps({'sample_counts':summary['semantic_sample_counts'],'estimates':est,'aux_n':len(aux),'administrative_counts':summary['administrative_all_pair_counts']},ensure_ascii=False))
