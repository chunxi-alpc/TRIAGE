
"""Prepare official-source fidelity pairs without inference, fitting or new annotations."""
import sys,json,time,re,copy
from pathlib import Path
from collections import Counter
sys.path.insert(0,'/home/gcx/TRIAGE-worktrees/aster-cr-20261108')
from experiments.aster_cr.repair.build import action_guides,rule_variants,replace_condition,edit_values,edited_fact,ancestors_preserved
from experiments.aster_cr.repair.program_edits import background_from_measurements,numeric_edits
from experiments.aster_cr.repair.core import condition,evaluate,key,replace_spans
from experiments.aster_closed_form.data import canonical_guide,prompt
BASE=Path('/home/gcx/aster_next_20261109')
OUT=BASE/'fidelity/mega_official'
SOURCE=BASE/'inventory/official_all_path_cases.jsonl'
ANNOTATION=Path('/home/gcx/aster_cr_repair_20261108')
def rows(path):
 p=Path(path)
 if not p.exists():return
 with p.open() as f:
  for line in f:
   if line.strip():yield json.loads(line)
def write(handle,row):handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
def dump(path,row):Path(path).write_text(json.dumps(row,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def signature(q,patient,definitions):return q,patient,json.dumps(definitions,ensure_ascii=False,sort_keys=True)
def load_annotations():
 rr={};fr={};er={}
 for folder in [ANNOTATION/'pilot',ANNOTATION/'bulk']:
  for p in sorted(folder.glob('rules_*.jsonl')):rr.update({r['id']:r for r in rows(p)})
  for p in sorted(folder.glob('facts_*.jsonl')):fr.update({r['id']:r for r in rows(p)})
  for p in sorted(folder.glob('edits_*.jsonl')):er.update({r['id']:r for r in rows(p)})
 overrides=ANNOTATION/'pilot/review_overrides.json'
 if overrides.exists():
  changes=json.loads(overrides.read_text());rr.update(changes.get('rules',{}));fr.update(changes.get('facts',{}));er.update(changes.get('edits',{}))
 matched={};qcomplete=set()
 for r in rr.values():
  if r.get('state')=='complete' and r.get('condition'):qcomplete.add(r['condition'])
 for g in rows(ANNOTATION/'data/nodes.jsonl'):
  rule=rr.get(key(g),{});fact=fr.get(g['group_id'],{})
  if rule.get('state')!='complete' or fact.get('state')!='complete':continue
  matched[signature(condition(g),g['patient'],g.get('definitions',[]))]=(rule,fact,er.get(g['group_id'],{}))
 return matched,qcomplete
def load_cache():
 lookup={};files=[]
 for root in [BASE/'primary_results/mega_official_test',BASE/'results']:
  for file in sorted(root.glob('shard*/nodes.jsonl')):
   files.append(str(file))
   for r in rows(file):
    if r.get('input_variant','canonical')!='canonical' or '|' in r.get('id',''):continue
    if r.get('status')!='ok' or not r.get('feature') or not Path(r['feature']).exists():continue
    lookup.setdefault(r['id'],dict(r,reuse_source_file=str(file)))
 return lookup,files
def main():
 started=time.time();OUT.mkdir(parents=True,exist_ok=True);cache,cachefiles=load_cache();ann,qcomplete=load_annotations()
 handles={name:(OUT/(name+'.jsonl')).open('w',encoding='utf-8') for name in ['views_cases','pairs','nodes','groups','unavailable']}
 stats=Counter();kinds=Counter();full=Counter();sources=set();first_condition_keys=set();all_condition_keys=set();trees={};pair_sources={k:set() for k in ['A','P','C','F','F0']}
 canonical_prompt_checks=0;matched_ann_count=0;rule_reuse_count=0;views_to_forward=0;first_cached=0;all_gold_cached=0
 for c in rows(SOURCE):
  sources.add(c['source_id']);stats['paths']+=1
  tk=(c['guideline_id'],c.get('language','en'))
  if tk not in trees:
   tree=canonical_guide(c['guideline']);ct={n['id']:n['condition'] for n in tree['nodes']}
   proto=dict(c,gid=c['gold_path'][0]['gid'],language=c.get('language','en'),ancestors=[])
   ag=list(action_guides(proto))
   trees[tk]=(ct,ag)
  ct,ag=trees[tk];gold=c['gold_path'];full['gold_nodes']+=len(gold);full['A_pairs']+=len(gold)*len(ag)
  # Count complete-gold coverage and program P availability without choosing by model scores.
  ancestors=[]
  for gl in gold:
   og=c['case_id']+':'+gl['gid'];all_condition_keys.add((c['guideline_id'],gl['gid']))
   all_gold_cached+=og in cache
   gg=dict(c,gid=gl['gid'],ancestors=list(ancestors),language=c.get('language','en'))
   if background_from_measurements(gg):full['program_P_pairs']+=1
   ancestors.append(dict(gid=gl['gid'],label=gl['label'],condition=ct[gl['gid']]))
  chosen=gold[0];gid=chosen['gid'];first_condition_keys.add((c['guideline_id'],gid));cachekey=c['case_id']+':'+gid
  group_id='official-fidelity:'+c['case_id']+':'+gid
  g=dict(c,gid=gid,group_id=group_id,original_label=chosen['label'],original_selected=True,ancestors=[],language=c.get('language','en'))
  base=dict(group_id=group_id,source_id=c['source_id'],source_key=c.get('source_key'),split='test',dataset='MEGA-CDP',
    dataset_split='official_test',evaluation_group=c.get('evaluation_group','MEGA-CDP:official_test'),
    history_group=c.get('history_group'),current_source_role=c.get('current_source_role'),known_earlier_input_use=c.get('known_earlier_input_use'),
    parent_case_id=c['case_id'],target_gid=gid,selection='first released gold node, independent of model scores')
  left_prompt=prompt(c['guideline'],c.get('definitions',[]),c['patient'],gid)
  if left_prompt!=prompt(g['guideline'],g.get('definitions',[]),g['patient'],g['gid']):raise RuntimeError('Original endpoint prompt changed')
  canonical_prompt_checks+=1
  def add(name,kind,patient=None,guide=None,label=None,**extra):
   nonlocal views_to_forward,first_cached
   vid=group_id+':'+name;y=chosen['label'] if label is None else int(label)
   text=prompt(guide if guide is not None else c['guideline'],c.get('definitions',[]),c['patient'] if patient is None else patient,gid)
   v=dict(base,case_id=vid,id=vid,view_id=vid,kind=kind,variant=name,original=name=='full',
    task_type='binary',gid='G0',positive_leaf='L1',negative_leaf='L0',label=y,
    prompt_text=text,gold_path=[dict(gid='G0',label=y)],gold_leaf='L1' if y else 'L0',**extra)
   if name=='full' and cachekey in cache:
    previous=cache[cachekey];rec=dict(previous,id=vid+':G0',case_id=vid,gid='G0',source_id=c['source_id'],input_variant='canonical',
      original_node_id=cachekey,reused_from_file=previous['reuse_source_file'])
    write(handles['nodes'],rec);v.update(requires_forward=False,original_node_id=cachekey,cached_feature=previous['feature']);first_cached+=1
   else:v['requires_forward']=True;views_to_forward+=1
   write(handles['views_cases'],v);stats['views']+=1
   return v
  left=add('full','original');pair_number=0
  def pair(kind,right,subtype,**extra):
   nonlocal pair_number
   pp=dict(base,case_id=c['case_id'],pair_id=group_id+':'+kind+':'+str(pair_number),kind=kind,subtype=subtype,
    left=left['view_id'],right=right['view_id'],left_label=left['label'],right_label=right['label'],**extra)
   pair_number+=1;kinds[kind]+=1;pair_sources[kind].add(c['source_id']);write(handles['pairs'],pp)
  for subtype,guide,protected in ag:
   variant={'presence':'remove_action','neutral':'neutral_action','swap':'swap_action'}[subtype]
   right=add(variant,'A',guide=guide);pair('A',right,'action_'+subtype,protected_action_ids=protected)
  q=ct[gid];rule_reuse_count+=q in qcomplete
  existing=ann.get(signature(q,c['patient'],c.get('definitions',[])))
  ready=existing is not None
  if ready:
   rr,fr,er=existing;rule=rr['value'];fv=fr['value'];facts=fv['facts'];obs=fv['observations']
   ready=evaluate(rule['predicate'],facts) is not None and int(evaluate(rule['predicate'],facts))==chosen['label']
  else:rr=fr=er={};rule=fv=facts=obs={}
  bg=background_from_measurements(g)
  if bg:
   try:
    context=condition(g)
    if bg.get('unrelated_to_all_conditions') is not True or bg['field'] in context or bg['field'] in obs:raise ValueError('background belongs to current rule')
    if bg['old'] not in c['patient'] or bg['old']==bg['new']:raise ValueError('unchanged/unlocated background')
    patient=replace_spans(c['patient'],[dict(old=bg['old'],new=bg['new'])])
    pair('P',add('P_background','P',patient=patient),'background',changed_field=bg['field'],edit_locations=[dict(old=bg['old'],new=bg['new'])],conditions_checked=[q],relevance_reason=bg.get('reason'))
   except (ValueError,KeyError,TypeError) as exc:write(handles['unavailable'],dict(base,kind='P',reason=str(exc)))
  else:write(handles['unavailable'],dict(base,kind='P',reason='Original program-background rule found no supported unrelated measurement'))
  cfcounts=Counter()
  if ready:
   matched_ann_count+=1;annlocal=dict(rules={key(g):rr},facts={group_id:fr},edits={group_id:er},ancestor_facts={})
   eds=numeric_edits(g,rule,fv,annlocal)+[ed for ed in er.get('value',{}).get('edits',[]) if ed.get('legal_values')]
   for i,(subtype,newq,expr,edit) in enumerate(rule_variants(q,rule['predicate'],eds)):
    yy=evaluate(expr,facts)
    if yy is None or int(yy)==chosen['label']:continue
    pair('C',add('C_'+str(i),'C',guide=replace_condition(g,newq),label=yy),subtype,edit_locations=[edit],patient_unchanged=True);cfcounts['C']+=1
   for j,ed in enumerate(eds):
    for k,(value,replacements,grid) in enumerate(edit_values(ed,rule['predicate'])):
     if value==facts.get(ed['field']):continue
     try:
      patient,edits=edited_fact(g,ed,value,replacements,obs);trace=ancestors_preserved(g,ed,value,annlocal)
      yy=evaluate(rule['predicate'],dict(facts,**{ed['field']:value}))
      if yy is None or patient==c['patient']:continue
      kind='F' if int(yy)!=chosen['label'] else 'F0'
      pair(kind,add(f'F_{j}_{k}',kind,patient=patient,label=yy),'cross' if kind=='F' else 'same_side',edit_locations=edits,ancestor_checks=trace);cfcounts[kind]+=1
     except (ValueError,KeyError,TypeError,IndexError):continue
  for kind in ('C','F','F0'):
   if not cfcounts[kind]:write(handles['unavailable'],dict(base,kind=kind,reason='No exact-condition + exact-patient qualified existing rule/observations' if not ready else 'Existing annotation produced no supported original-rule variant'))
  write(handles['groups'],dict(base,gid=gid,label=chosen['label'],original_node_id=cachekey,original_cached=cachekey in cache,
    A_variants=len(ag),P_available=bg is not None,existing_qualified_annotation=bool(ready)))
 for handle in handles.values():handle.close()
 summary=dict(protocol='repair-v2 program action/background rules; fixed first gold node per official path; no model-score selection',
  source_manifest=str(SOURCE),selected=dict(paths=stats['paths'],sources=len(sources),independent_tree_nodes=len(first_condition_keys),
   original_endpoints=stats['paths'],cached_original_endpoints=first_cached,missing_original_cache=stats['paths']-first_cached,
   views=stats['views'],pairs=dict(kinds),pair_sources={k:len(v) for k,v in pair_sources.items()},new_forward_upper_bound=views_to_forward),
  all_gold_alternative=dict(gold_nodes=full['gold_nodes'],independent_tree_nodes=len(all_condition_keys),cached_original_endpoints=all_gold_cached,
   missing_original_cache=full['gold_nodes']-all_gold_cached,A_pairs=full['A_pairs'],program_P_pairs=full['program_P_pairs'],
   new_forward_upper_bound_A_P=full['gold_nodes']-all_gold_cached+full['A_pairs']+full['program_P_pairs']),
  original_prompt_matches=canonical_prompt_checks,complete_rule_text_matches=rule_reuse_count,
  exact_patient_rule_observation_matches=matched_ann_count,new_annotations=False,new_training=False,new_gpu_forwards=0,
  CF_F0_status='Only qualified exact-input existing annotations; no invented values or labels; unavailable categories recorded',
  reused_cache_files=cachefiles,elapsed_seconds=time.time()-started,
  files={name:str(OUT/(name+'.jsonl')) for name in handles},
  scoring='run_primary.py --cases views_cases.jsonl --reuse-root this_directory; postprocess raw numeric scores from nodes/feature using frozen adapters, not dummy binary accuracy')
 dump(OUT/'inventory.json',summary);print(json.dumps(summary,ensure_ascii=True))
if __name__=='__main__':main()

