
"""One fixed evidence-alignment prompt; no labels or scores enter construction."""
from pathlib import Path
import json,sys,re
from collections import defaultdict,Counter
ROOT=Path('/home/gcx/aster_cr_repair_v2_run_20261108')
OUT=Path('/home/gcx/aster_next_20261109/diagnosis')
sys.path.insert(0,str(OUT))
from evidence_retrieval import retrieve,parse_original_prompt
sys.path.insert(0,'/home/gcx/TRIAGE-worktrees/aster-cr-20261108')
from experiments.aster_closed_form.data import prompt as original_node_prompt
ALIGNMENT_INSTRUCTION='''请在作答前逐项核对当前条件与病例中的对应证据：
1. 对齐变量名称。不要将名称相近但含义不同的指标相互替代。
2. 对齐时间点。区分出生时、既往、当前、初次检查和复查；使用当前条件指定时间点的证据。
3. 对齐单位与比较关系。核对数值、单位、阈值及大于、小于、包含端点等关系。
4. 核对复合条件与否定。分别核对各分句，再按“并且”“或者”“不”“未”等原有关系判断；不要忽略否定的作用范围。
5. 当前判断依据是当前条件、相关定义和病例事实。后续处置或其他节点的结论不能代替这些证据。
保留原任务的答案标签，只输出“是”或“否”。'''
METHOD='evidence_alignment_v1'
def enhance_prompt(original_prompt):
 """Add fixed guidance and one lexical-match verbatim patient sentence."""
 condition,patient=parse_original_prompt(original_prompt)
 selected=retrieve(condition,patient,topk=1) if condition and patient else []
 excerpt=''
 if selected:
  excerpt='按当前条件词汇匹配得到的病例原文候选（仅辅助定位，不代表已确认相关或判断结论）：\n'+selected[0]['quote']+'\n\n'
 cue='\n答案：'
 if original_prompt.endswith(cue):
  return original_prompt[:-len(cue)]+'\n\n'+excerpt+ALIGNMENT_INSTRUCTION+cue
 return original_prompt+'\n\n'+excerpt+ALIGNMENT_INSTRUCTION+cue
def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def writejsonl(path,values):
 with Path(path).open('w',encoding='utf-8') as f:
  for value in values:f.write(json.dumps(value,ensure_ascii=False)+'\n')
def entry(identifier,split,source_id,case_id,gid,label,original_prompt,tags,view_id=None):
 return dict(id=identifier,view_id=view_id or identifier,method=METHOD,split=split,source_id=source_id,
  case_id=case_id,gid=gid,label=int(label),original_prompt=original_prompt,prompt=enhance_prompt(original_prompt),tags=tags)
def dedup(values):
 byprompt={}
 for value in values:
  original=value['original_prompt']
  if original in byprompt:
   old=byprompt[original]
   if old['label']!=value['label']:raise ValueError('same original prompt has conflicting evaluation labels')
   old['tags']=sorted(set(old['tags'])|set(value['tags']))
   old.setdefault('aliases',[]).append(value['id'])
  else:byprompt[original]=dict(value)
 return list(byprompt.values())
def build_manifests():
 views=[v for v in rows(ROOT/'data/views.jsonl') if v['split'] in ('fit','dev') and v['variant']=='full' and v['original']]
 natural={}
 for split in ('fit','dev'):
  vv=[v for v in views if v['split']==split]
  natural[split]=[entry(v['view_id'],split,v['source_id'],v['case_id'],v['group_id'].rsplit('-',1)[-1],v['label'],v['prompt'],
   [split+'_natural'],view_id=v['view_id']) for v in vv]
 examples=json.loads((OUT/'error_examples.json').read_text())
 focused_keys={(e['case_id'],e['gid']) for e in examples if e['first_error_for']['raw'] and e['source_id']=='train_val:2072' and e['gid']=='G2'}
 cases=[c for c in rows(ROOT/'data/path_cases.jsonl') if c['split']=='dev']
 focused=[];full=[]
 for c in cases:
  for gold in c['gold_path']:
   gid=gold['gid'];text=original_node_prompt(c['guideline'],c['definitions'],c['patient'],gid)
   tags=['dev_gold_path_node']
   if (c['case_id'],gid) in focused_keys:tags.append('dev_diagnostic_PMA_family')
   item=entry('path|'+c['case_id']+':'+gid,'dev',c['source_id'],c['case_id'],gid,gold['label'],text,tags)
   full.append(item)
   if (c['case_id'],gid) in focused_keys:focused.append(item)
 short=dedup(natural['dev']+focused)
 full_plus=dedup(natural['dev']+full)
 for name,rr in [('fit_natural',natural['fit']),('dev_natural',natural['dev']),('dev_focus',focused),('dev_short',short),('dev_full_gold',full_plus)]:
  writejsonl(OUT/(METHOD+'.'+name+'.jsonl'),rr)
 # Builder does not accept a label or use a score in enhance_prompt.
 assert enhance_prompt(natural['fit'][0]['original_prompt'])==natural['fit'][0]['prompt']
 metadata=dict(method=METHOD,candidate_count=1,instruction=ALIGNMENT_INSTRUCTION,retrieval=dict(topk=1,method='fixed per-input lexical TF-IDF with English unigrams/bigrams and Chinese character bigrams',quotes='exact patient substrings'),
  construction='Before the answer cue, insert one patient sentence selected by fixed lexical TF-IDF matching to the current condition plus the fixed alignment instruction. The entire original guideline, definitions, patient and condition remain. No gold/facts annotation or model score enters construction',
  selection_scope='fit/dev only; focused dev nodes were identified in the development error diagnosis',
  manifests={name:dict(path=str(OUT/(METHOD+'.'+name+'.jsonl')),rows=len(rr)) for name,rr in [('fit_natural',natural['fit']),('dev_natural',natural['dev']),('dev_focus',focused),('dev_short',short),('dev_full_gold',full_plus)]},
  run_scope='Single final candidate updated before any GPU candidate scoring; neural_search confirmed no candidate run had started. No GPU/model forward executed by this builder.',
  evaluation_note='Report natural and diagnostic-focus results separately; full path accuracy requires scoring all dev gold nodes or all reachable nodes. Short-list accuracy is not path accuracy.')
 (OUT/(METHOD+'.json')).write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(metadata['manifests'],ensure_ascii=True))
if __name__=='__main__':build_manifests()

