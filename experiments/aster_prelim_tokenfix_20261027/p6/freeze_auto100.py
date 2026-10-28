"""Freeze 100 automatically constructed development pairs, no human annotation gate."""
import collections,hashlib,json,math,shutil,sys
from pathlib import Path
from datetime import datetime,timezone
D=Path(__file__).resolve().parent;B=D.parent
sys.path.insert(0,str(D))
from transformers import AutoTokenizer
from run_auto_pilot import encoded,MODEL,PROJECTION
import run_auto_pilot as runner
OLD=runner.OLD
S=OLD/'runtime/vendors/MEGA-CDP-ssh/mega_cdp/scripts/evaluate'
def fh(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 source=B/'private/p6_auto_exact_v2';rows=[json.loads(l) for l in (source/'pairs.jsonl').read_text().splitlines()]
 groups=collections.defaultdict(list)
 for r in rows:groups[r['guideline_id']].append(r)
 order=sorted(groups,key=lambda k:hashlib.sha256(('p6-auto100-guideline:'+k).encode()).hexdigest())
 for k in order:groups[k].sort(key=lambda r:hashlib.sha256(('p6-auto100-case:'+r['pair_id']).encode()).hexdigest())
 selected=[];depth=0
 while len(selected)<100:
  added=0
  for k in order:
   if depth<len(groups[k]):
    selected.append(groups[k][depth]);added+=1
    if len(selected)==100:break
  assert added
  depth+=1
 out=B/'private/p6_auto100';out.mkdir(exist_ok=False)
 shutil.copytree(source/'guidelines',out/'guidelines');(out/'decoy_guidelines').mkdir()
 ledger=[]
 for r in selected:
  shutil.copy2(source/r['decoy_guideline_file'],out/r['decoy_guideline_file'])
  a=(out/r['native_guideline_file']).read_text();z=(out/r['decoy_guideline_file']).read_text()
  assert hashlib.sha256(a.encode()).hexdigest()==r['native_guideline_sha256']
  edits=sorted(r['guide_edits'],key=lambda e:e['start']);actual=a
  assert edits[0]['end']<=edits[1]['start']
  for e in reversed(edits):
   assert a[e['start']:e['end']]==e['before']
   actual=actual[:e['start']]+e['after']+actual[e['end']:]
  assert actual==z and hashlib.sha256(z.encode()).hexdigest()==r['decoy_guideline_sha256']
  assert r['validation']['selected_terms_not_in_tree_conditions']
  assert r['source_split']=='train_val/development'
  ledger.append({'pair_id':r['pair_id'],'guideline_id':r['guideline_id'],'decision_nodes':r['decision_nodes'],'native_guideline_sha256':r['native_guideline_sha256'],'decoy_guideline_sha256':r['decoy_guideline_sha256'],'program_checks':'PASS_EXACT_EDIT_CONTRACT','clinical_semantics':'UNVERIFIED'})
 (out/'pairs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
 results=D/'results/auto100';results.mkdir(parents=True,exist_ok=False)
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);lengths=[];runner.DATA=out
 for r in selected:
  for variant in ['native','decoy']:
   for method in ['Vanilla','TS-Note','Prompt-Repeat','Patient-First']:
    pack,meta=encoded(tok,r,variant,method,[.5,.5])
    lengths.append({'pair_id':r['pair_id'],'variant':variant,'method':method,'input_tokens':len(pack['input_ids']),'branch_token_counts':[len(x) for x in meta['branch_tokens']]})
 maximum=max(lengths,key=lambda x:x['input_tokens'])
 budget=2048;context=65536 if maximum['input_tokens']+budget<=65536 else 131072
 assert maximum['input_tokens']+budget<=context,'Need a separate long-context implementation; do not drop cases'
 rope={'rope_type':'yarn','factor':context/32768,'original_max_position_embeddings':32768}
 sources=['run_auto_pilot.py','generation_gate.py','score_auto_pilot.py','score_cache.py','scan_exact_treatments_v2.py','build_exact_pairs_v2.py','summarize_auto_pilot.py']
 source_hashes={p:fh(D/p) for p in sources}
 official={p:fh(S/p) for p in ['parse_prediction_output_singleturn.py','pathway_consistency_eval_batch_singleturn.py','pairwise_score_parser.py','pathway_metrics.py','summarize_eval_metrics.py']}
 scorer={'version':'auto-pilot-pathcost-rubric-v1','model':MODEL,'independent_model':False,'official_commit':'15a3a12e73a9169a870bff2ac431ec2e9968b8c7','official_files_sha256':official,'implementation_sha256':source_hashes['score_auto_pilot.py'],'client_sha256':fh(OLD/'mega_local_qwen_client.py'),'pathway_samples_per_unique_score':1,'pathway_max_new_tokens':4096,'rubric_max_new_tokens':1024,'temperature':.7,'top_p':.8,'top_k':20,'thinking':False,'seed_policy':'hash of unique score tuple','empty_or_parse_failure_pathcost':1.,'semantic_validation':'AUTOMATIC_UNVERIFIED'}
 freeze={'version':'p6_auto100_v1_1_scorer_guards','created_at':datetime.now(timezone.utc).isoformat(),'case_count':100,'input_count':200,'guideline_count':len({r['guideline_id'] for r in selected}),'complex_pair_count':sum(r['decision_nodes']>2 for r in selected),'source_candidates':len(rows),'sampling':'Round-robin over SHA256-ranked development guidelines and SHA256-ranked cases, no method outcome or length selection','smoke_ids':[r['pair_id'] for r in selected[:20]],'longest_pair_id':maximum['pair_id'],'maximum_input_tokens':maximum['input_tokens'],'pairs_sha256':fh(out/'pairs.jsonl'),'source_hashes':source_hashes,'protocol_sha256':fh(D/'protocol.json'),'projection_sha256':fh(PROJECTION),'generation':{'max_new_tokens':budget,'max_context_tokens':context,'rope_scaling':rope,'dtype':'bfloat16','greedy':True,'thinking':False,'truncation':False},'scorer':scorer,'test_files_read':False,'human_annotation_required_for_p6':False,'clinical_semantic_validity':'UNVERIFIED','evidence_use':'Exploratory continuation decision only, not standalone clinical validity or significant method superiority'}
 assert freeze['guideline_count']>=20 and freeze['complex_pair_count']>=40
 dump(results/'freeze.json',freeze);dump(results/'program_validation.json',ledger);dump(results/'token_lengths.json',lengths)
 dump(results/'candidate_construction.json',json.loads((source/'audit.json').read_text()))
 print(json.dumps({k:freeze[k] for k in ['case_count','guideline_count','complex_pair_count','maximum_input_tokens','longest_pair_id','generation']}))
if __name__=='__main__':main()
