#!/usr/bin/env python3
"""Apply the disclosed, source-reviewed target-action alias catalogue offline."""
import gzip,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];p=ROOT/'results/acre_pilot/round2';rpath=p/'development_references.json'
if not (p/'development_references.pre_alias_audit.json.gz').exists():
 (p/'development_references.pre_alias_audit.json.gz').write_bytes(gzip.compress(rpath.read_bytes()))
refs=json.loads(rpath.read_text());items=json.loads((p/'development_manifest.json').read_text())['samples'];byid={x['case_id']:x for x in items}
catalog={'v2_628_31':['mega_628_leaf_17','mega_628_leaf_32']}
it=next(i for i in items if i['pair_id']=='v2_1476_5')
catalog['v2_1476_5']=[x['rule_id'] for x in it['guideline']['rules'] if x['action'].startswith('Provide intensive phototherapy as first-line treatment while proceeding with emergent ET;')]
reviews=[]
for cid,r in refs.items():
 aliases=[r['target_action']]
 if r['pair_id'] in catalog:aliases=list(dict.fromkeys(x['action'] for x in byid[cid]['guideline']['rules'] if x['rule_id'] in catalog[r['pair_id']]))
 assert r['target_action'] in aliases
 r['target_action_aliases']=aliases
 if r['target_applicability']=='inapplicable':
  assert r['gold_action'] not in aliases
  reviews.append({'pair_id':r['pair_id'],'target_action':r['target_action'],'misuse_action_aliases':aliases,'reference_action':r['gold_action'],'source_rule_ids':catalog.get(r['pair_id'],[r['target_rule_id']]),'scope':'Exact target action plus explicitly source-reviewed full-action paraphrases; conditional/negated mentions and different procedures excluded.'})
assert len(reviews)==40
rpath.write_text(json.dumps(refs,ensure_ascii=False,indent=2)+'\n')
audit=json.loads((p/'data_audit.json').read_text());audit.setdefault('original_frozen_references_sha256',audit['references_sha256']);audit['references_sha256']=hashlib.sha256(rpath.read_bytes()).hexdigest();audit['posthoc_metric_correction']='Source-action alias review after discovering exact-label OAR undercount; all 40 pairs scanned, two pairs with full-action paraphrases. No patients, gold routes, predictions, oracle masks or beta grid changed. Original refs archived. This is a disclosed evaluation correction, not preregistered.'
(p/'data_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
(p/'action_alias_audit.json').write_text(json.dumps({'scope':'All 40 target action sets reviewed against full original action options','posthoc_evaluation_correction':True,'alias_pairs':catalog,'pairs':reviews},ensure_ascii=False,indent=2)+'\n')
print('40 offline target-action sets audited; two full-action paraphrase groups.')
