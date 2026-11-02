"""Prepare new-source, conditional-node counterfactuals without consulting predictions."""
import json,re
from pathlib import Path
from .data import BASE,ROOT,OUT,read,tree_guideline,write,jsonl
DEST=OUT/'matched_faithfulness'
SPECS=[[3, 18, "Clopidogrel is being considered after a neurointerventional procedure. Documented prior stroke or TIA: {}.", "present", "absent"], [9, 17, "The treating physician measured the closest margin at {} cm.", "0.8", "1.2"], [15, 37, "绝对淋巴细胞计数为 {} ×10^9/L。", "0.6", "0.4"], [22, 8, "Current gestational age: {} weeks.", "26", "30"], [30, 15, "CLDN18.2 immunohistochemistry result: {}.", "available", "unavailable"], [32, 41, "After the heparin lead-in, planned total duration of oral anticoagulation is {} months.", "4", "2"], [33, 27, "The recorded QRISK2 ten-year cardiovascular risk is {}%.", "11", "9"], [41, 14, "The largest intra-abdominal abscess measures {} cm on imaging.", "3.5", "2.5"], [45, 31, "The child's current age is {} months.", "15", "9"], [47, 31, "During therapeutic balloon occlusion of the ICA, mean CBFV fell by {}% relative to the individual baseline.", "55", "45"], [66, 21, "On standard proctological examination, macroscopic anal lesions are {}.", "present", "absent"], [56, 6, "Current HbA1c: {} mmol/mol.", "47", "49"], [65, 13, "Curative surgery occurred 12 months ago. Primary therapy was completed {} months ago.", "5", "7"], [67, 28, "预计术后保留功能性残肝体积为35%，预计肿瘤切除率为{}%。", "75", "65"], [68, 34, "Molecular testing of the leukemic clone: BRAF V600E {}.", "detected", "not detected"], [70, 2, "Current eGFR: {} mL/min/1.73 m².", "32", "28"], [74, 4, "标准血清凝集试验滴度为 1:{}。", "160", "80"], [76, 46, "当前 ECOG PS 评分为 {}。", "2", "3"], [88, 3, "BRCA1/2 检测结论：{}致病突变。", "检出", "未检出"], [89, 42, "The bidirectional signal has a predominant frequency of {} Hz.", "350", "450"]]
def prepare():
 DEST.mkdir(exist_ok=True)
 if (DEST/'pairs.jsonl').exists():return
 manifest=json.loads((OUT/'manifest.json').read_text())
 excluded=set(manifest['excluded_sources'])|{r['source_id'] for r in manifest['sources']}
 clusters=json.loads((BASE/'source_clusters.json').read_text())
 excluded.update(clusters['official_test_clusters'])
 for split in ['development','age_train','age_diagnostic','train','confirmation']:
  excluded.update(c['source_id'] for c in read(OUT/(split+'.jsonl')))
 trees={int(r['guideline_id']):r for r in read(ROOT/'train_val/decision_trees.jsonl')}
 pairs=[];seen=set()
 for gid,node,template,yes,no in SPECS:
  source=clusters['records'][f'train_val:{gid}']['cluster']
  if source in excluded or source in seen:continue
  guideline,_=tree_guideline(json.loads(trees[gid]['text'])['decision_tree'])
  groups={g['group_id']:g for g in guideline['groups']}
  target=groups[f'G{node}'];rs=target['rules']
  assert [r['rule_id'] for r in rs]==[f'R{node}Y',f'R{node}N']
  leaves=[re.fullmatch(r'(L\d+[：:])(.*)',r['action'],re.S) for r in rs]
  if not all(leaves) or leaves[0][2].strip()==leaves[1][2].strip():continue
  hist=[];a=target
  while a.get('parent_rule_id'):
   rid=a['parent_rule_id'];hist.insert(0,rid);a=groups['G'+rid[1:-1]]
  pairs.append(dict(pair_id=f'matched:{gid}:{node}',source_id=source,gid=f'G{node}',history=hist,guideline=guideline,
               fact_template=template,yes=template.format(yes),no=template.format(no),question='Choose the branch at the current guideline decision.'))
  seen.add(source)
 jsonl(DEST/'pairs.jsonl',pairs)
 write(DEST/'design.json',dict(pairs=len(pairs),sources=len(seen),views_per_pair=4,methods=2,decision_forwards=len(pairs)*8,
  scope='Constructed single-fact records at a supplied node and fixed ancestor prefix; node adherence, not full-case routing.',checkpoint='no_js_layer18/epoch3.pt',training='reuse exact 216 records, original training views, order, no-JS loss, 3 epochs; fixed epoch 3 for both arms',selection='Manually specified literal conditions before viewing any predictions; distinct new source clusters; no outcome-based selection.'))
if __name__=='__main__':prepare()
