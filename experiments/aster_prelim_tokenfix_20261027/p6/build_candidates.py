"""Prepare P6 source dossiers on server; no model inference, no test access."""
import collections,csv,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
D=Path(__file__).resolve().parent
BASE=D.parent
OLD=Path('/home/gcx/TRIAGE-worktrees/aster-seka-execution-20261025/experiments/aster_seka_execution')
ROOT=Path('/home/gcx/TRIAGE/data/.hf_cache/datasets--nchennnn--MEGA-CDP/snapshots/8f058c88ce1ed43693785b435ad0ff99907a2890')
sys.path.insert(0,str(OLD))
from mega_trainval_reader import snapshot_digest,safe_guideline_path,value_digest
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
 with p.open() as f:
  for line in f:
   if line.strip():yield json.loads(line)
def key(r):return (r.get('case_path_id',r.get('guideline_id')),r.get('path_index',r.get('path_id')),r['filename'])
def walk(n,parent=None,edge=None):
 yield n,parent,edge
 for k,v in n.items():
  if isinstance(v,dict) and 'index' in v:yield from walk(v,n,k)
def main():
 assert json.loads((BASE/'results/p5/gate_status.json').read_text())['P5']=='PASS_PRUNE_GATE'
 lock=json.loads((OLD/'mega_dev16_input_lock.json').read_text())
 groups=[g for g in lock['groups'] if g['partition']=='development']
 allowed={fn:g['source_cluster'] for g in groups for fn in set(g['filenames'])}
 rows=collections.defaultdict(list)
 source=ROOT/'train_val/vignette_cases.jsonl'
 assert snapshot_digest(ROOT,source)==lock['source_file_sha256']
 for r in read(source):
  if r['filename'] in allowed:rows[allowed[r['filename']]].append(r)
 ordered=sorted(groups,key=lambda g:value_digest([20261027,'P6-guideline',g['source_cluster']]))
 eligibility=[];chosen=[]
 for g in ordered:
  n=len(rows[g['source_cluster']]);ok=n>=8
  eligibility.append({'guideline_id':g['source_cluster'],'available_cases':n,'structurally_eligible_for_eight':ok,'selection':'selected' if ok and len(chosen)<20 else 'reserve' if ok else 'fewer_than_eight_cases'})
  if ok and len(chosen)<20:chosen.append(g)
 assert len(chosen)==20
 selected=[]
 for g in chosen:
  selected+=sorted(rows[g['source_cluster']],key=lambda r:value_digest([20261027,'P6-case',list(key(r))]))[:8]
 filenames={r['filename'] for r in selected}
 tree_rows={r['filename']:r for r in read(ROOT/'train_val/decision_trees.jsonl') if r['filename'] in filenames}
 path_rows={r['filename']:r for r in read(ROOT/'train_val/case_paths.jsonl') if r['filename'] in filenames}
 refs={key(r):r for r in read(ROOT/'train_val/gt_paths.jsonl') if r['filename'] in filenames}
 private=BASE/'private/p6_candidates_v1';private.mkdir(exist_ok=False)
 (private/'guidelines').mkdir();(private/'dossiers').mkdir()
 source_hashes={name:snapshot_digest(ROOT,ROOT/'train_val'/name) for name in ['vignette_cases.jsonl','decision_trees.jsonl','case_paths.jsonl','gt_paths.jsonl']}
 docs={}
 for fn in sorted(filenames):
  p=safe_guideline_path(ROOT/'train_val/guideline_md',fn);h=snapshot_digest(ROOT,p);assert h==allowed[fn]
  docs[fn]=p.read_text()
  (private/'guidelines'/(fn+'.md')).write_text(docs[fn])
 cases=[];public=[];failures=[];reviews=[]
 for r in selected:
  gid=allowed[r['filename']];pid=value_digest(['P6',list(key(r))])[:20]
  try:
   tree=json.loads(tree_rows[r['filename']]['text'])
   path=path_rows[r['filename']]['paths'][int(key(r)[1])-1]
   nodes={int(n['index']):(n,p,e) for n,p,e in walk(tree['decision_tree'])}
   leaf=path[-1];n,parent,edge=nodes[int(leaf['index'])]
   assert leaf.get('conclusion')==n.get('conclusion')
   assert all(int(v['index']) in nodes for v in path)
   ref=refs[key(r)];reftext=ref['output_text'];reference=json.loads(reftext)
   assert isinstance(reference,list) and reference
   mapping=tree.get('node_guideline_mapping',{})
   relevant=[int(v['index']) for v in path]
   sibling_nodes=[]
   if parent:
    for branch,other in parent.items():
     if isinstance(other,dict) and 'index' in other and other['index']!=n['index']:
      sibling_nodes.append({'branch':branch,'node':other});relevant.append(int(other['index']))
   quotes=[]
   for index in dict.fromkeys(relevant):
    values=mapping.get(str(index),[])
    if isinstance(values,str):values=[values]
    if not isinstance(values,list):continue
    for quote in values:
     if not isinstance(quote,str):continue
     matches=[];start=0
     while (pos:=docs[r['filename']].find(quote,start))>=0:
      matches.append([pos,pos+len(quote)]);start=pos+len(quote) if quote else pos+1
      if not quote:break
     quotes.append({'node_id':index,'quote':quote,'exact_spans':matches})
   complex_n=sum('condition' in v for v in path)
   candidate={'pair_id':pid,'guideline_id':gid,'source_case_key':list(key(r)),'source_filename':r['filename'],'source_split':'train_val/development','language':r.get('language','en'),'patient_facts':r['text'],'question':'Follow the supplied full guideline and generate the complete decision path.','native_guideline_file':'guidelines/'+r['filename']+'.md','native_guideline_sha256':gid,'native_reference_path':reference,'source_decision_path':path,'proposed_target_parent':parent['index'] if parent else None,'target_condition':parent.get('description') if parent else None,'target_branch':edge,'target_action_node':n,'sibling_nodes':sibling_nodes,'mapping_quotes':quotes,'decision_nodes':complex_n,'multinode_candidate':complex_n>2,'decoy_patch':None,'decoy_reference_path':None,'review_status':'PENDING_HUMAN_REVIEW','candidate_only':True}
   cases.append(candidate)
   public.append({k:candidate[k] for k in ['pair_id','guideline_id','source_case_key','source_filename','source_split','language','native_guideline_sha256','decision_nodes','multinode_candidate','review_status']})
   review={k:'' for k in (D/'review_template.csv').read_text().strip().split(',')}
   review.update(pair_id=pid,guideline_id=gid,source_case_id=json.dumps(list(key(r))),source_split='train_val/development',source_revision=ROOT.name,multinode_or_cross_fragment=str(complex_n>2),complexity_reason=f'{complex_n} source decision nodes',verdict='PENDING');reviews.append(review)
   sections=[f'# P6 candidate {pid}', '', '**未人工核验；诱饵改写尚未完成；不得当作有效配对。**', '', '## Patient',r['text'],'','## Source decision path',json.dumps(path,ensure_ascii=False,indent=2),'','## Reference path',json.dumps(reference,ensure_ascii=False,indent=2),'','## Target and sibling candidates',json.dumps({'condition':candidate['target_condition'],'branch':edge,'action_node':n,'siblings':sibling_nodes},ensure_ascii=False,indent=2),'','## Source quotation mappings',json.dumps(quotes,ensure_ascii=False,indent=2),'',f'Full guideline: ../guidelines/{r["filename"]}.md']
   (private/'dossiers'/(pid+'.md')).write_text('\n'.join(sections)+'\n')
  except (KeyError,ValueError,TypeError,IndexError,AssertionError) as e:
   failures.append({'pair_id':pid,'source_case_key':list(key(r)),'reason':type(e).__name__+': '+str(e)})
 (private/'candidates.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in cases))
 with (private/'review.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=reviews[0] if reviews else (D/'review_template.csv').read_text().strip().split(','));w.writeheader();w.writerows(reviews)
 res=D/'results';res.mkdir(exist_ok=True)
 dump(res/'candidate_index.json',public);dump(res/'guideline_selection.json',eligibility);dump(res/'construction_failures.json',failures)
 audit={'status':'CANDIDATE_DOSSIERS_PREPARED_NOT_VALID_PAIRS','created_at':datetime.now(timezone.utc).isoformat(),'development_guidelines':len(groups),'development_cases':sum(map(len,rows.values())),'selection':'SHA256(seed=20261027, purpose, guideline/case); 20 development guidelines with >=8 source cases, eight cases each; no model score selection','candidate_count':len(cases),'guideline_count':len({c['guideline_id'] for c in cases}),'multinode_candidates':sum(c['multinode_candidate'] for c in cases),'source_alignment_failures':len(failures),'drafted_decoy_pairs':0,'human_verified_pairs':0,'model_inference_count':0,'test_files_read':False,'prior_group_manifest_sha256':sha(OLD/'mega_dev16_input_lock.json'),'source_files_sha256':source_hashes,'script_sha256':sha(Path(__file__)),'private_package':str(private),'case_file_sha256':sha(private/'candidates.jsonl'),'guideline_lengths_chars':{fn:len(txt) for fn,txt in docs.items()},'next':'Draft only action-span edits and corresponding references; human verification required before validity claim.'}
 dump(res/'candidate_audit.json',audit)
 print(json.dumps({k:v for k,v in audit.items() if k not in ['guideline_lengths_chars','source_files_sha256']},ensure_ascii=False))
if __name__=='__main__':main()

