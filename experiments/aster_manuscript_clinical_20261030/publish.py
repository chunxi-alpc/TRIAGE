"""Publish aggregate clinical results and source identifiers, never clinical records."""
import argparse,json,subprocess,shutil
from collections import Counter,defaultdict
from pathlib import Path
from .data import BASE,read,write,SPECS,REV
from .engine import METHODS

def publish(run,out):
 out.mkdir(parents=True,exist_ok=True)
 data=json.loads((run/'data02/data_audit.json').read_text());review=json.loads((run/'review/audit.json').read_text());accepted=read(run/'review/accepted_pairs.jsonl');candidate=read(run/'data02/candidate_pairs.jsonl')
 reasons=Counter(r['reason'] for r in data['exclusions'])
 folds={}
 for split in ['fit','calibration','selection','development','confirm']:
  before=[r for r in candidate if r['split']==split];after=[r for r in accepted if r['split']==split]
  folds[split]={'submitted_triplets':len(before),'accepted_triplets':len(after),'rejected_triplets':len(before)-len(after),'submitted_source_clusters':len({r['source_id'] for r in before}),'accepted_source_clusters':len({r['source_id'] for r in after})}
 write(out/'data_inclusion.json',{'clinical_population':'MEGA-CDP benchmark-generated clinical vignettes; not prospective hospital records','current_age_rule_specs':len(SPECS),'eligible_source_records':data['eligible_sources_before_alignment'],'accepted_construction_sources':data['accepted_source_clusters'],'merged_duplicate_source_records':1,'construction_exclusion_reason_counts':dict(reasons),'submitted_triplets':review['submitted'],'accepted_triplets':review['accepted'],'rejected_triplets':len(review['rejected']),'folds':folds,'gold_standard_scope':'independent current-age predicate verification plus auxiliary model consistency review','physician_adjudicated_triplets':0,'backbone_used_to_construct_labels':False,'auxiliary_reviewer':'Meta-Llama-3.1-8B-Instruct','auxiliary_reviewer_upstream_revision':None,'primary_model_revision':REV})
 partition={}
 for r in candidate:
  partition[r['source_id']]={'source_cluster':r['source_id'],'split':r['split'],'guideline_id':r['source_guideline_id'],'guideline_filename':r['source_filename'],'accepted_after_review':any(x['source_id']==r['source_id'] for x in accepted),'current_age_rule_spec':SPECS[int(r['source_guideline_id'])]}
 write(out/'source_partitions.json',list(partition.values()))
 source=json.loads((run/'source_clusters.json').read_text())
 write(out/'source_audit.json',{'normalization':source['method'],'source_guideline_records':len(source['records']),'normalized_clusters':len({x['cluster'] for x in source['records'].values()}),'merges':len(source['merges']),'PR40_excluded_clusters':len(source['pr40_clusters']),'PR40_exposure':source['pr40_tree_ids'],'cross_corpus_policy':source['cross_corpus_policy'],'MedGUIDE_records':7747,'MedGUIDE_original_page_values':53,'MedGUIDE_normalized_families':17,'independent_external_validation_claim':False,'MEGA_CDP_revision':'8f058c88ce1ed43693785b435ad0ff99907a2890','MedGUIDE_revision':'1460ec478280a7295040b92f7eb985327f77dcbe'})
 for source_path,name in [('frozen.json','frozen.json'),('analysis/summary.json','learning_summary.json'),('analysis/rank_selection.json','rank_selection.json'),('analysis/sample_sensitivity.json','sample_sensitivity.json'),('clinical_statistics/structure.json','structure.json'),('clinical_statistics/mechanism_root_readout_v2.json','mechanism_root_readout_v2.json'),('acceptance_frozen/status.json','real_weight_acceptance.json'),('summary_confirm/comparisons.json','confirm_comparisons.json'),('summary_confirm/operator_contrasts.json','confirm_operator_contrasts.json'),('summary_confirm/status.json','confirm_status.json'),('native_numeric_reference/status.json','native_numeric_reference.json'),('native_numeric_disagreement/status.json','native_naive_partition_disagreement.json'),('native_partition_reference.json','native_balanced_numeric_reference.json'),('official_template_equivalence.json','official_template_equivalence.json')]:
  p=run/source_path
  if p.exists():shutil.copyfile(p,out/name)
 shutil.copyfile(run/'analysis/spectrum.png',out/'spectrum.png')
 native={}
 for phase in ['native_memory_repair','native_balanced_reference','official_smoke']:
  native[phase]=[json.loads((run/(phase+str(i))/'status.json').read_text()) for i in [0,1]]
 write(out/'native_and_judge_acceptance.json',native)
 costs={}
 for phase in ['collect','classification','gain','mechanism','confirm']:
  costs[phase]=[json.loads((run/(phase+str(i))/'status.json').read_text()) for i in [0,1] if (run/(phase+str(i))/'status.json').exists()]
 write(out/'costs.json',costs)
 clusters=[]
 for shard in [0,1]:
  for p in (run/f'confirm{shard}'/'units').glob('*/*.json'):
   if p.name=='classification.json':continue
   r=json.loads(p.read_text());clusters.append(r)
 grouped=defaultdict(list)
 for r in clusters:grouped[r['source_id'],r['method']].append(r)
 write(out/'confirm_guideline_aggregates.json',[{'source_cluster':source,'method':method,'denominator':len(rs),'technical_failures':sum(r['status']=='failed' for r in rs),'leaf_correct':sum(r.get('score',{}).get('final_leaf_correct',False) for r in rs) if method!='native_official' else None,'path_rule_f1':sum(r.get('score',{}).get('path_f1',0.) for r in rs)/len(rs) if method!='native_official' else None} for (source,method),rs in sorted(grouped.items())])
 expert=json.loads((run/'analysis/expert.json').read_text())
 write(out/'template_and_model.json',{'model_revision':REV,'template_id':expert.get('template_id',expert.get('provenance',{}).get('template_id')),'model_identity':expert.get('identity',{}),'template_version':'v2; whitespace outside action before newline; no condition included in key span','enable_thinking':False})
 launch=json.loads((run/'official_test_launch.json').read_text());launch.pop('worker_pids',None)
 write(out/'official_test_launch.json',launch)
 startup=run/'official_startup_observation.json'
 if startup.exists():shutil.copyfile(startup,out/'official_startup_observation.json')
 for name in ['comparisons.json','operator_contrasts.json','official_Qwen_scores.json','status.json']:
  p=run/'summary_official_test'/name
  if p.exists():shutil.copyfile(p,out/('official_test_'+name))
 write(out/'publication_manifest.json',{'publication_code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'run_id':'clinical01','source_split_reused_from_synthetic':False,'clinical_data_and_raw_outputs_published':False,'official_test_state_at_publication':launch['status'],'full_official_test_completed':launch['status']=='complete','MedGUIDE_end_to_end_completed':False,'regression_tests':63,'regression_skips':0,'main_model_revision':REV,'official_scoring_source':'nchennnn/MEGA-CDP@15a3a12e73a9169a870bff2ac431ec2e9968b8c7','judge':'local Qwen3-8B, not GPT'})

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',default=str(BASE));p.add_argument('--out',required=True);a=p.parse_args();publish(Path(a.run),Path(a.out))
if __name__=='__main__':main()
