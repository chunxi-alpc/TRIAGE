
"""Condition-only input control; no labels, scores or domain-specific answers."""
from pathlib import Path
import json
OUT=Path('/home/gcx/aster_next_20261109/diagnosis')
METHOD='condition_only_v1'
def enhance_prompt(original_prompt):
 """Keep complete definitions, complete patient, and the current condition."""
 definition_header='指南原文定义与说明：'
 patient_header='\n\n病历：\n'
 current_node_header='\n\n当前节点：'
 condition_header='\n当前条件：'
 if definition_header not in original_prompt or patient_header not in original_prompt or condition_header not in original_prompt:
  raise ValueError('Expected canonical guideline/patient/current-condition prompt')
 definitions=original_prompt.split(definition_header,1)[1].split(patient_header,1)[0]
 patient=original_prompt.split(patient_header,1)[1].split(current_node_header,1)[0]
 condition=original_prompt.rsplit(condition_header,1)[1].split('\n请判断',1)[0]
 return definition_header+definitions+patient_header+patient+'\n\n当前条件：'+condition+'\n请根据以上定义、病例和当前条件判断该条件是否成立。只输出“是”或“否”。\n答案：'
def main():
 manifests={}
 for tag in ['fit_natural','dev_natural','dev_focus','dev_short','dev_full_gold']:
  src=OUT/('evidence_alignment_v1.'+tag+'.jsonl');dst=OUT/(METHOD+'.'+tag+'.jsonl')
  rows=[json.loads(s) for s in src.read_text().splitlines() if s.strip()]
  with dst.open('w',encoding='utf-8') as f:
   for row in rows:
    row['method']=METHOD;row['prompt']=enhance_prompt(row['original_prompt'])
    f.write(json.dumps(row,ensure_ascii=False)+'\n')
  manifests[tag]=dict(path=str(dst),rows=len(rows))
 meta=dict(method=METHOD,candidate_index=2,candidate_count_this_family=1,
  construction='Keep the full original definitions section, full original patient and current condition; remove the other tree-node conditions and terminal action text. No lexical excerpt, extra alignment checklist, labels, scores or clinical rules are injected.',
  construction_module=str(OUT/'condition_only.py'),manifests=manifests,selection_scope='fit/dev only',
  warning='This input control removes other conditions/actions; it does not establish a constrained score-editor guarantee. The original full-input methods remain unchanged.')
 (OUT/(METHOD+'.json')).write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(manifests))
if __name__=='__main__':main()

