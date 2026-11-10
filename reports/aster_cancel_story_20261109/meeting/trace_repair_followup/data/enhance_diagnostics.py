"""Enhance fixed diagnostic nodes with original source-grounded repair extraction."""
import os,sys,json,time,random,argparse
from pathlib import Path
from collections import defaultdict,Counter
HERE=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE/'data'))
from data import BIG,OUT,SEED,rows,dump,emit,task_id,original_task,canonical_guide,combine
from repair_reference.core import ROOT

def prepare():
 groups=[];indices=[];rng_seed=SEED+17
 for split in ['dev','final']:
  grouped=defaultdict(list)
  for c in rows(BIG/f'cases_{split}.jsonl'):
   if c['variant']=='original':
    for gold in c['gold_path']:grouped[c['source_id']].append((c,gold))
  rng=random.Random(rng_seed)
  for sid in sorted(grouped):
   cc=grouped[sid];rng.shuffle(cc);seen=set();selected=[]
   for c,gold in cc:
    key=(c['guideline_id'],gold['gid'])
    if key in seen:continue
    seen.add(key);selected.append((c,gold))
    if len(selected)>=10:break
   for c,gold in selected:
    gid=gold['gid'];nodes=canonical_guide(c['guideline'])['nodes'];conds={n['id']:n['condition'] for n in nodes};i=next(j for j,x in enumerate(c['gold_path']) if x['gid']==gid)
    g=dict(c,gid=gid,group_id='followup-'+c['case_id']+':'+gid,original_label=gold['label'],original_selected=True,cohort='new_source_diagnostics',ancestors=[dict(gid=x['gid'],condition=conds[x['gid']],label=x['label']) for x in c['gold_path'][:i]])
    groups.append(g);indices.append({k:g[k] for k in ['group_id','case_id','source_id','split','gid','original_label','guideline_id']})
 (ROOT/'data').mkdir(parents=True,exist_ok=True);(ROOT/'bulk').mkdir(exist_ok=True)
 with (ROOT/'data/nodes.jsonl').open('w') as f:
  for g in groups:emit(f,g)
 with (OUT/'diagnostic_node_index.jsonl').open('w') as f:
  for g in indices:emit(f,g)
 dump(ROOT/'bulk/assignment.json',dict(shards=[[g['group_id'] for g in groups]],seed=SEED,selection='same fixed at-most10 gold nodes/source as program-only draft, no model error filtering'))
 dump(OUT/'diagnostic_extraction_plan.json',dict(nodes=len(groups),split_counts=dict(Counter(g['split'] for g in groups)),sources={s:len({g['source_id'] for g in groups if g['split']==s}) for s in ['dev','final']},annotation_root=str(ROOT),model='/home/gcx/TRIAGE/models/modelscope/Qwen3-8B',rule_scope='Only current source condition and original definitions, without patient or released answer',facts_scope='Only requested variables and numbered original patient sentences, without answer',edit_scope='One observed variable, exact original spans, all repeats and affected ancestors; no invented patient',final_scope='Evaluation-pair construction only; no method scores, final judgments or model selection',source_unchanged=True,main_tasks_unchanged=True,reference='Original repair/core.py, extract.py, program_edits.py, build.py copied under data/repair_reference; ROOT redirected into new annotations directory, one-shard assignment permitted.'))
 return groups

def export():
 from repair_reference.build import build
 from repair_reference.core import evaluate
 built=build('bulk');folder=ROOT/'bulk/built';views={v['view_id']:v for v in rows(folder/'views.jsonl')};groups={g['group_id']:g for g in rows(ROOT/'data/nodes.jsonl')};allpairs=[]
 for record in rows(folder/'pairs.jsonl'):
  group=groups[record['group_id']]
  # The original builder stores group_id, not gid. Resolve identity from that exact group.
  normalized=dict(record,case_id=group['case_id'],gid=group['gid'],source_id=group['source_id'],split=group['split'])
  allpairs.append(normalized)
 dump(OUT/'pair_identity_normalization.json',dict(pairs=len(allpairs),identity_source='Exact group_id join against annotations/data/nodes.jsonl; never parse case IDs to infer identity',fields=['case_id','gid','source_id','split'],label_policy='Original generated left_label/right_label retained unchanged',groups=len(groups)))
 draft={s:json.loads((OUT/f'diagnostic_program_coverage_{s}.json').read_text()) if (OUT/f'diagnostic_program_coverage_{s}.json').exists() else {} for s in ['dev','final']}
 for split in ['dev','final']:
  ps=[p for p in allpairs if p['split']==split and p['kind'] in ['P','C','F']];tasks={};pairs=[];counts=Counter();source_kinds=defaultdict(set)
  for p in ps:
   g=groups[p['group_id']];left=views[p['left']];right=views[p['right']];gid=g['gid'];lid=task_id(g,gid,'original');variant='diagnostic_'+right['variant'];rid=task_id(g,gid,variant)
   tasks[lid]=dict(id=lid,source_id=g['source_id'],case_id=g['case_id'],gid=gid,split=split,variant='original',kind='original',prompt=left['prompt'],label=int(p['left_label']))
   tasks[rid]=dict(id=rid,source_id=g['source_id'],case_id=g['case_id'],gid=gid,split=split,variant=variant,kind=p['kind'],prompt=right['prompt'],label=int(p['right_label']))
   meta={k:v for k,v in p.items() if k not in ['left','right','pair_id','cohort','construction_version']}
   pair=dict(meta,pair_id=g['case_id']+':'+gid+':'+p['kind']+':'+right['variant'],left=lid,right=rid,scope='node_pair_only_no_modified_path_claim',truth_method='original_source_only_rule_compilation_patient_sentence_grounding_and_program_recomputation')
   pairs.append(pair);counts[p['kind']]+=1;source_kinds[p['kind']].add(g['source_id'])
  # Keep valid direct numeric examples if the larger extractor fails on one of those nodes.
  # Their baseline facts and flips were already verified without any model output.
  backup=BIG/f'pairs_diagnostic_program_{split}.jsonl';backtasks=BIG/f'tasks_diagnostic_program_{split}.jsonl'
  if backup.exists() and backtasks.exists():
   oldtasks={t['id']:t for t in rows(backtasks)}
   covered={(p['case_id'],p['gid'],p['kind']) for p in pairs}
   for p in rows(backup):
    if (p['case_id'],p['gid'],p['kind']) in covered:continue
    pairs.append(p);counts[p['kind']]+=1;source_kinds[p['kind']].add(p['source_id'])
    for ident in [p['left'],p['right']]:tasks[ident]=oldtasks[ident]
  with (BIG/f'tasks_diagnostic_{split}.jsonl').open('w') as f:
   for t in tasks.values():emit(f,t)
  with (BIG/f'pairs_diagnostic_{split}.jsonl').open('w') as f:
   for p in pairs:emit(f,p)
  failures=[x for x in rows(folder/'unavailable.jsonl') if x['split']==split]
  attempted=sum(g['split']==split for g in groups.values())
  coverage=dict(split=split,attempted_gold_nodes=attempted,maximum_attempts_per_source=10,pairs=dict(counts),tasks=len(tasks),pair_sources={k:len(v) for k,v in source_kinds.items()},failure_reasons={kind:dict(Counter(x['reason'] for x in failures if x['kind']==kind)) for kind in ['P','C','F','rule']},annotation_root=str(ROOT),evidence_type='new_Qwen_source_fact_edit_extraction_plus_program_verified_interventions',scope='node_pair_diagnostics_only',final_scoring_allowed=False if split=='final' else True,program_only_draft=draft[split])
  dump(OUT/f'diagnostic_coverage_{split}.json',coverage)
  # Compact reproducibility records retained in the repository; full prompts/raw generations stay in the large cache.
  with (OUT/f'diagnostic_pairs_{split}.jsonl').open('w') as f:
   for p in pairs:emit(f,p)
 combine()
 for split in ['dev','final']:
  dump(OUT/f'diagnostic_{split}_ready.json',dict(state='complete',ready=True,split=split,tasks=str(BIG/f'tasks_diagnostic_{split}.jsonl'),pairs=str(BIG/f'pairs_diagnostic_{split}.jsonl'),coverage=str(OUT/f'diagnostic_coverage_{split}.json'),final_scoring_allowed=False if split=='final' else True))
 dump(OUT/'diagnostic_extraction_complete.json',dict(state='complete',time=time.strftime('%Y-%m-%d %H:%M:%S'),annotation_root=str(ROOT)))

def run(batch):
 status=ROOT/'bulk/status_0.json'
 if status.exists() and json.loads(status.read_text()).get('stage')=='complete':
  export();return
 if not (ROOT/'data/nodes.jsonl').exists():prepare()
 for split in ['dev','final']:
  dump(OUT/f'diagnostic_{split}_status.json',dict(state='building',ready=False,split=split,reason='source-grounded Qwen extraction and program verification in progress'))
 from repair_reference.extract import run as extract_run
 extract_run(shard=0,scope='bulk',batch=batch)
 export()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--export',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--batch',type=int,default=6);a=p.parse_args()
 if a.prepare:prepare()
 elif a.export:export()
 elif a.run:run(a.batch)
 else:p.error('choose --prepare, --run or --export')
