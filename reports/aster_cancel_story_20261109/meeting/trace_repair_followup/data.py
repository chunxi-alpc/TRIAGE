"""Prepare source-held-out trace repair follow-up data; no model scoring."""
import os,sys,json,csv,copy,re,time,random,gzip,argparse
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
sys.path.insert(0,str(REPO));sys.path.insert(0,str(HERE/'data'))
from experiments.aster_closed_form.data import canonical_guide,prompt,condition,ancestors
from repair_action_guides import action_guides
BIG=Path('/home/gcx/aster_trace_repair_20261110/data')
OUT=HERE/'data'
BASE=Path('/home/gcx/aster_cr_repair_v2_run_20261108')
CATALOG=Path('/home/gcx/aster_next_20261109/inventory/train_val_path_cases.jsonl')
MAP=Path('/home/gcx/aster_manuscript_20261030/clinical01/source_clusters.json')
RAW=Path('/home/gcx/TRIAGE/data/MEGA-CDP/train_val')
SEED=20261110
VARIANTS={'presence':'action_presence','neutral':'action_neutral','swap':'action_swap'}

def rows(p):
 p=Path(p)
 if not p.exists():return
 with p.open() as f:
  for line in f:
   if line.strip():yield json.loads(line)
def dump(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def emit(f,row):f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
def lang(c):
 if c.get('language'):return c['language']
 text=canonical_guide(c['guideline'])['nodes'][0]['condition']
 return 'zh' if re.search(r'[\u4e00-\u9fff]',text) else 'en'
def task_id(c,gid,variant):return c['case_id']+':'+gid+':'+variant

def case_variants(c,split):
 base={k:c[k] for k in ['case_id','source_id','guideline','patient','definitions','gold_path','gold_leaf']}
 for k in ['guideline_id','source_filename']:
  if k in c:base[k]=c[k]
 base.update(split=split,language=lang(c),variant='original',original_dataset='MEGA-CDP generated vignettes')
 yield base,[]
 group=dict(c,language=lang(c))
 for mode,guide,protected in action_guides(group):yield dict(base,guideline=guide,variant=VARIANTS[mode]),protected

def original_task(c,gid,variant,guide,label,protected=()):
 return dict(id=task_id(c,gid,variant),case_id=c['case_id'],source_id=c['source_id'],split=c['split'],variant=variant,gid=gid,label=label,
  kind='original' if variant=='original' else 'A',prompt=prompt(guide,c['definitions'],c['patient'],gid),protected_action_ids=list(protected))

def prepare_split(split,cases):
 BIG.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
 counts=Counter();variants=Counter();unavailable=Counter();source_counts=Counter();pairs_by_kind=Counter();start=time.time()
 with (BIG/f'cases_{split}.jsonl').open('w') as cf,(BIG/f'tasks_{split}.jsonl').open('w') as tf,(BIG/f'pairs_{split}.jsonl').open('w') as pf,(OUT/f'case_index_{split}.jsonl').open('w') as index:
  for c in cases:
   c=dict(c,split=split);gold={g['gid']:g['label'] for g in c['gold_path']};vv=list(case_variants(c,split))
   source_counts[c['source_id']]+=1
   if len(vv)<4:unavailable['fewer_than_three_valid_action_changes']+=1
   nodes=canonical_guide(c['guideline'])['nodes']
   selected=nodes if split!='fit' else [n for n in nodes if n['id'] in gold]
   for v,protected in vv:
    emit(cf,v);variants[v['variant']]+=1
    emit(index,{k:v[k] for k in ['case_id','source_id','split','variant','gold_path','gold_leaf','guideline_id','language']})
    for n in selected:
     gid=n['id'];label=gold.get(gid)
     emit(tf,original_task(c,gid,v['variant'],v['guideline'],label,protected));counts['tasks']+=1;counts['gold_tasks' if label is not None else 'off_gold_tasks']+=1
     if v['variant']!='original':
      emit(pf,dict(pair_id=c['case_id']+':'+gid+':A:'+v['variant'],left=task_id(c,gid,'original'),right=task_id(c,gid,v['variant']),kind='A',source_id=c['source_id'],case_id=c['case_id'],gid=gid,split=split,left_label=label,right_label=label,protected_action_ids=protected));pairs_by_kind['A']+=1
   counts['cases_original']+=1
 dump(OUT/f'{split}_coverage.json',dict(split=split,sources=len(source_counts),source_case_counts=source_counts,counts=counts,case_variants=variants,pairs=pairs_by_kind,unavailable=unavailable,task_scope='gold path nodes only' if split=='fit' else 'all decision nodes per selected case, enabling each method own rollout',labels='released gold-path node labels; off-gold nodes null',seconds=time.time()-start,evidence_type='existing_released_generated_vignettes_and_program_action_interventions',tasks=str(BIG/f'tasks_{split}.jsonl'),cases=str(BIG/f'cases_{split}.jsonl')))
 dump(OUT/f'{split}_ready.json',dict(split=split,ready=True,tasks=str(BIG/f'tasks_{split}.jsonl'),cases=str(BIG/f'cases_{split}.jsonl'),pairs=str(BIG/f'pairs_{split}.jsonl'),final_scoring_allowed=False if split=='final' else True))
 return counts

def fit():
 rr=[x for x in rows(BASE/'data/path_cases.jsonl') if x['split']=='fit']
 return prepare_split('fit',rr)

def match_source(r,records):
 out=set()
 for k in ['source_id','source_key','case_id','original_case_id','node_id','id']:
  value=str(r.get(k,''));m=re.fullmatch(r'(train_val|test):(\d+)',value)
  if m:reference=m.group(1)+':'+m.group(2)
  elif value.startswith('mega:'):
   m=re.search(r'(train_val|test):(\d+)',value);reference=m.group(1)+':'+m.group(2) if m else None
  else:reference=None
  if reference in records:out.add(records[reference]['cluster'])
 return out

def current_history():
 old=json.loads(Path('/home/gcx/aster_cr_20261108/data/history_usage.json').read_text())
 records=json.loads(MAP.read_text())['records'];used=set(old['excluded']);files=[];malformed=0
 # Current original fit/dev/test and repair inputs are all treated as historical use.
 for p in [BASE/'data/sources.jsonl',BASE/'data/views.jsonl',Path('/home/gcx/aster_cr_repair_20261108/data/sources.jsonl')]:
  for r in rows(p):used.update(match_source(r,records))
 roots=[]
 for r in [Path('/home/gcx/TRIAGE-worktrees'),Path('/home/gcx')]:
  for p in r.iterdir():
   if p.is_dir() and (p/'reports').is_dir() and p!=REPO:roots.append(p/'reports')
 roots += [Path('/home/gcx/aster_next_20261109'),Path('/home/gcx/aster_hard_optimization_20261109/fit_augmentation'),Path('/home/gcx/aster_cross_task_transfer_20261109')]
 seen=set()
 for root in roots:
  for folder,dirs,names in os.walk(root):
   dirs[:]=[d for d in dirs if d not in ['.git','features','path_features','__pycache__','inventory','models','checkpoints','trace_repair_followup']]
   for name in names:
    if not name.endswith(('.jsonl','.jsonl.gz','.csv','.csv.gz')) or not any(x in name for x in ['nodes','results','path','score','prediction','metadata','pair','cases']):continue
    p=Path(folder)/name
    if str(p) in seen:continue
    seen.add(str(p));source_set=set();n=0
    opener=gzip.open if name.endswith('.gz') else open
    with opener(p,'rt',encoding='utf8',errors='replace') as f:
     rr=csv.DictReader(f) if '.csv' in name else f
     for r in rr:
      if isinstance(r,str):
       try:r=json.loads(r)
       except json.JSONDecodeError:malformed+=1;continue
      if not isinstance(r,dict):continue
      source_set.update(match_source(r,records));n+=1
    used.update(source_set)
    files.append(dict(path=str(p),rows=n,observed_sources=len(source_set)))
 return records,used,dict(scanned_files=files,malformed_or_partial_lines=malformed,history_source='/home/gcx/aster_cr_20261108/data/history_usage.json',used_source_count=len(used),excluded_source_ids=sorted(used),policy='Count actual experimental inputs/results, not unscored candidate inventory/catalogues; include reports under worktrees and publication checkouts.')

def choose_new():
 if (OUT/'source_selection.json').exists():
  selection=json.loads((OUT/'source_selection.json').read_text());materialize_new(selection);return selection
 records,used,history=current_history();cases_per_gid=Counter(int(x['guideline_id']) for x in rows(RAW/'vignette_cases.jsonl'));path_gids={int(x['guideline_id']) for x in rows(RAW/'case_paths.jsonl')}
 meta=defaultdict(lambda:dict(gids=[],languages=set(),nodes=0,cases=0))
 for t in rows(RAW/'decision_trees.jsonl'):
  gid=int(t['guideline_id']);key='train_val:'+str(gid)
  if gid not in path_gids or not cases_per_gid[gid] or key not in records:continue
  sid=records[key]['cluster']
  if sid in used:continue
  m=meta[sid];m['gids'].append(gid);m['languages'].add(t['language']);m['cases']+=cases_per_gid[gid]
  raw=json.loads(t['text'])['decision_tree']
  def count(n):
   if isinstance(n,list):return sum(count(x) for x in n)
   if isinstance(n,dict):return int('description' in n and not n.get('leaf',False))+sum(count(v) for k,v in n.items() if isinstance(v,(list,dict)))
   return 0
  m['nodes']+=count(raw)
 keys=sorted(meta);sizes=np.array([meta[s]['nodes'] for s in keys]);cuts=np.quantile(sizes,[.25,.5,.75])
 # Rank quartiles remain defined if two node-count cutoffs coincide.
 strata=[','.join(sorted(meta[s]['languages']))+':'+str(int(np.searchsorted(cuts,meta[s]['nodes'],side='right'))) for s in keys]
 from sklearn.model_selection import train_test_split
 selected,_=train_test_split(keys,train_size=80,random_state=SEED,stratify=strata)
 labels={k:v for k,v in zip(keys,strata)}
 dev,final=train_test_split(selected,train_size=20,test_size=60,random_state=SEED+1,stratify=[labels[s] for s in selected])
 splitmap={s:'dev' for s in dev};splitmap.update({s:'final' for s in final})
 selection=dict(seed=SEED,dev_sources=sorted(dev),final_sources=sorted(final),unused_source_candidates=len(keys),candidate_trees=sum(len(meta[s]['gids']) for s in keys),candidate_cases=sum(meta[s]['cases'] for s in keys),source_selection='language x decision-tree-size quartiles, fixed random selection; no model correctness, score, error, or performance filtering',source_metadata={s:dict(guideline_ids=meta[s]['gids'],languages=sorted(meta[s]['languages']),decision_nodes=meta[s]['nodes'],cases=meta[s]['cases'],stratum=labels[s]) for s in sorted(selected)},final_scoring_allowed=False,final_status='manifest_only_until_freeze',terminology='Sources with no recorded project use in scanned histories; cannot establish absence from Qwen pretraining; cases remain existing generated MEGA vignettes.')
 dump(OUT/'history_scan.json',history);dump(OUT/'source_selection.json',selection)
 materialize_new(selection)
 return selection

def materialize_new(selection):
 splitmap={s:'dev' for s in selection['dev_sources']};splitmap.update({s:'final' for s in selection['final_sources']})
 grouped=defaultdict(list)
 for c in rows(CATALOG):
  if c['source_id'] in splitmap:grouped[c['source_id']].append(c)
 rng=random.Random(SEED);by_split=defaultdict(list)
 for sid in sorted(splitmap):
  cc=sorted(grouped[sid],key=lambda x:x['case_id']);rng.shuffle(cc)
  # Sampling cases before all model scoring; never drop a small source.
  by_split[splitmap[sid]].extend(cc[:15])
 for split in ['dev','final']:prepare_split(split,by_split[split])

def combine():
 if not (BIG/'cases.jsonl').exists():
  with (BIG/'cases.jsonl').open('w') as f:
   for split in ['fit','dev','final']:
    for line in (BIG/f'cases_{split}.jsonl').open():f.write(line)
 with (BIG/'pairs.jsonl').open('w') as f:
  for split in ['fit','dev','final']:
   for line in (BIG/f'pairs_{split}.jsonl').open():f.write(line)
   if (BIG/f'pairs_diagnostic_{split}.jsonl').exists():
    for line in (BIG/f'pairs_diagnostic_{split}.jsonl').open():f.write(line)
 dump(OUT/'manifest.json',dict(seed=SEED,data_root=str(BIG),cases=str(BIG/'cases.jsonl'),tasks={s:str(BIG/f'tasks_{s}.jsonl') for s in ['fit','dev','final']},pairs=str(BIG/'pairs.jsonl'),diagnostic_tasks={s:str(BIG/f'tasks_diagnostic_{s}.jsonl') for s in ['dev','final']},diagnostic_pairs={s:str(BIG/f'pairs_diagnostic_{s}.jsonl') for s in ['dev','final']},final_scoring_allowed=False,variants=['original','action_presence','action_neutral','action_swap'],case_id='Original released case ID; variant is separate, task IDs combine case/gid/variant',original_fit_sources='Existing80 source assignment, existing1712 original cases',source_file=str(CATALOG),history=str(OUT/'history_scan.json'),action_code=str(OUT/'repair_action_guides.py'),action_code_origin='/home/gcx/TRIAGE-worktrees/aster-cr-20261108/experiments/aster_cr/repair/build.py:action_guides',diagnostic_references=['reports/aster_cancel_story_20261109/meeting/trace_validation/exp3/README.zh.md','reports/aster_cancel_story_20261109/meeting/trace_validation/exp3/full/source_metrics.csv'],evidence_type='existing_real_model_dataset_inputs_and_new_program_action_variants'))

def main():
 p=argparse.ArgumentParser();p.add_argument('--fit',action='store_true');p.add_argument('--new',action='store_true');p.add_argument('--combine',action='store_true');p.add_argument('--diagnostics',action='store_true');a=p.parse_args()
 if a.fit:fit()
 elif a.new:choose_new()
 elif a.combine:combine()
 elif a.diagnostics:
  from prepare_diagnostics import main as diagnostic_main
  diagnostic_main();combine()
 else:
  fit();choose_new()
  completed=Path('/home/gcx/aster_trace_repair_20261110/annotations/bulk/status_0.json')
  if completed.exists() and json.loads(completed.read_text()).get('stage')=='complete':
   from enhance_diagnostics import export
   export()
  else:
   from prepare_diagnostics import main as diagnostic_main
   diagnostic_main();combine()
if __name__=='__main__':main()
