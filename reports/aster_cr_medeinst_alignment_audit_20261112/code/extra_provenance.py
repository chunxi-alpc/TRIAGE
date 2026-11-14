from pathlib import Path
import json,csv,re,collections,difflib
root=Path('/home/gcx/aster-cr-medeinst-alignment-audit-20261112/reports/aster_cr_medeinst_alignment_audit_20261112')
src=Path('/home/gcx/aster-cr-retrain-paired-v2-20261111/reports/aster_cr_multisource_20261112')
def rows(p):return [json.loads(s) for s in Path(p).open() if s.strip()]
def key(t):
 return '\n'.join(x.strip() for x in t.splitlines() if x.strip() and not re.match(r'^Sex:\s*[MF],\s*Age:',x.strip()) and x.strip() not in ['text','---','```','```text'])
med=rows(src/'balanced_10006/data/medeinst_train_selected.jsonl')
groups=collections.defaultdict(dict)
for r in rows(src/'raw/medeinst_test.jsonl'):groups[r['case_id']][r['case_type']]=r
testkeys=collections.defaultdict(list)
for cid,rs in groups.items():
 if set(rs)=={'control','trap'}:testkeys[(key(rs['control']['narrative']),key(rs['trap']['narrative']))].append(cid)
trainkeys=collections.defaultdict(list)
for r in med:trainkeys[(key(r['original_input']),key(r['edited_input']))].append(r['pair_id'])
matches=[]
for p in med:
 k=(key(p['original_input']),key(p['edited_input']))
 for cid in testkeys.get(k,[]):
  rs=groups[cid]
  matches.append(dict(train_pair_id=p['pair_id'],test_case_id=cid,same_diagnosis_transition=(p['label_before'],p['label_after'])==(rs['control']['ground_truth'],rs['trap']['ground_truth']),
   finding='both clinical narratives identical after removing demographics header and Markdown wrappers; demographic values may differ; not proof of same original DDXPlus patient'))
with (root/'demographic_template_matches.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,list(matches[0]));w.writeheader();w.writerows(matches)
summary=dict(method='remove Sex/Age header and Markdown-only wrappers; preserve all clinical text and remaining numbers; exact equality of both ordered narratives',
 train_pairs_matching_test=len({p['train_pair_id'] for p in matches}),test_cases_matched=len({p['test_case_id'] for p in matches}),pair_matches=len(matches),
 within_train_duplicate_template_groups=sum(len(v)>1 for v in trainkeys.values()),within_train_pairs_in_groups=sum(len(v) for v in trainkeys.values() if len(v)>1),
 identical_transition_count=sum(p['same_diagnosis_transition'] for p in matches),interpretation='confirmed narrative template overlap across released splits; demographics still may be clinically important. Does not establish identical patients or causal performance inflation.')
(root/'template_overlap_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
# Verify original MEGA labels against released case path decisions.
full=rows('/home/gcx/aster_cr_pair_dataset_20261111/full_pool_train.jsonl')
cp={int(r['guideline_id']):r['paths'] for r in rows('/home/gcx/TRIAGE/data/MEGA-CDP/train_val/case_paths.jsonl')}
result=[]
for r in full:
 if r['kind']!='F':continue
 gid=int(r['guideline_id']);pid=int(r['case_id'].split(':')[-1]);path=cp[gid][pid-1]
 matching=[n for n in path if str(n.get('index'))==str(r['raw_node_index'])]
 label=(1 if str(matching[0]['decision_outcome']).lower() in ['yes','是'] else 0) if len(matching)==1 else None
 result.append(dict(pair_id=r['pair_id'],path_node_found=len(matching)==1,released_path_label=label,original_label=r['original']['label'],match=label==r['original']['label']))
(root/'mega_original_label_check.json').write_text(json.dumps(dict(n=len(result),matches=sum(r['match'] for r in result),nodes_found=sum(r['path_node_found'] for r in result),mismatches=[r for r in result if not r['match']]),ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(template_overlap=summary,mega_label_matches=sum(r['match'] for r in result)),ensure_ascii=False))
