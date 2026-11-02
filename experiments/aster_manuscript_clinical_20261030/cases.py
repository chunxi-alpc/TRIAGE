"""Clinical benchmark manifests. Gold references are used only by scoring."""
import argparse,json,re
from pathlib import Path
from collections import Counter
from .data import ROOT,BASE,read,write,jsonl,tree_guideline
QUESTION='请依据完整指南给出患者的临床决策路径及最终处置。最后输出 PATH: 经过的规则编号；FINAL: 终点处置的L编号。'
def build(split,assignments,clusters):
 trees={int(r['guideline_id']):r for r in read(ROOT/split/'decision_trees.jsonl')}
 paths={int(r['guideline_id']):r['paths'] for r in read(ROOT/split/'case_paths.jsonl')}
 rows=[]
 for c in read(ROOT/split/'vignette_cases.jsonl'):
  gid=int(c['guideline_id']);key=split+':'+str(gid)
  if split=='train_val' and gid not in assignments:continue
  record={'case_id':f'mega:{split}:{gid}:{c["path_id"]}','dataset':'MEGA-CDP','source_split':split,'source_id':clusters[key]['cluster'],
   'phase':assignments.get(gid,'official_test') if split=='train_val' else 'official_test','guideline_id':gid,'path_id':int(c['path_id']),
   'filename':c['filename'],'language':c.get('language','en'),'patient':c['text'],'question':QUESTION}
  try:
   tree=json.loads(trees[gid]['text'])['decision_tree'];guide,mapping=tree_guideline(tree)
   goldpath=paths[gid][int(c['path_id'])-1]
   expected=[]
   for n in goldpath[:-1]:
    b=n.get('decision_outcome')
    if b not in ['yes','no']:raise ValueError('source path missing branch outcome')
    expected.append('R'+str(mapping[str(n['index'])]['node'])+('Y' if b=='yes' else 'N'))
   leaf=mapping[str(goldpath[-1]['index'])]
   if not leaf['leaf']:raise ValueError('source terminal is not leaf')
   path=ROOT/split/'guideline_md'/c['filename']/(c['filename']+'.md')
   record.update(guideline=guide,tree_mapping=mapping,source_tree=tree,gold_rules=expected,gold_leaf='L'+str(leaf['node']),gold_conclusion=leaf['description'],original_guideline_path=str(path))
  except Exception as e:record.update(source_failure=type(e).__name__+': '+str(e))
  rows.append(record)
 return rows
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=False)
 candidates=read(BASE/'data02/candidate_pairs.jsonl');assignments={int(r['source_guideline_id']):r['split'] for r in candidates}
 clusters=json.loads((BASE/'source_clusters.json').read_text())
 train=build('train_val',assignments,clusters['records'])
 # Official inputs registered now; no gold references enter fitting or deployment routing.
 test=build('test',{},clusters['records'])
 for phase in ['calibration','development','confirm']:
  jsonl(out/(phase+'.jsonl'),[r for r in train if r['phase']==phase])
 jsonl(out/'official_test.jsonl',test)
 exposure=set(clusters['pr40_clusters'])
 for row in test:row['previous_PR40_exposure']=row['source_id'] in exposure
 # Mark source exposure on the registered test list, never filter by predictions.
 (out/'official_test.jsonl').unlink();jsonl(out/'official_test.jsonl',test)
 import pyarrow.parquet as pq
 med=pq.read_table('/home/gcx/TRIAGE/data/MedGUIDE/data/train-00000-of-00001.parquet',columns=['disease'])
 families=sorted({re.sub(r'_?\d+$','',d) for d in med['disease'].to_pylist()})
 write(out/'manifest_audit.json',{'MEGA_registered':dict(Counter(r['phase'] for r in train+test)),'source_failures':dict(Counter(r['phase'] for r in train+test if 'source_failure' in r)),
  'official_test_cases':len(test),'official_test_sources':len({r['source_id'] for r in test}),'official_test_PR40_exposed_cases':sum(r['previous_PR40_exposure'] for r in test),
  'MedGUIDE_cases':med.num_rows,'MedGUIDE_disease_values':len(set(med['disease'].to_pylist())),'MedGUIDE_source_families':len(families),'MedGUIDE_families':families,
  'MedGUIDE_family_rule':'strip optional underscore AND trailing page numbers; col6 and col_6 join consistently',
  'gold_policy':'official references retained for post-prediction scoring only; no correctness gate in deployment'})
 print(json.dumps({'counts':dict(Counter(r['phase'] for r in train+test)),'source_failures':sum('source_failure' in r for r in train+test),'med_families':len(families)}))
if __name__=='__main__':main()

