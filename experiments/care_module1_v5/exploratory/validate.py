"""Verify the frozen exploratory data, exact source offsets and no input labels."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).resolve().parent

def load(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def sha(raw):return hashlib.sha256(raw).hexdigest()

def main():
 data=BASE/'data'; frozen=json.loads((data/'FREEZE.json').read_text())
 for filename,expected in frozen['files'].items():
  actual=sha((BASE/filename).read_bytes())
  assert actual==expected,(filename,actual,expected)
 current_builder=(BASE/'build.py').read_bytes()
 builder_revision='original frozen builder'
 if frozen['builder_sha256']!=sha(current_builder):
  note=json.loads((BASE/'REPRODUCTION_PATH_FIX.json').read_text())
  assert note['revision_type']=='post_run_reproduction_path_fix_only'
  assert note['original_builder_sha256']==frozen['builder_sha256']
  assert note['original_frozen_builder_sha256']==frozen['builder_sha256']
  assert note['revised_builder_sha256']==sha(current_builder)
  assert note['original_freeze_sha256']==sha((data/'FREEZE.json').read_bytes())
  assert note['original_freeze_unchanged'] is True
  assert note['all_original_data_files_unchanged'] is True
  # Verify this is exactly the disclosed single output-path expression fix,
  # not an arbitrary changed builder accepted solely because a note names it.
  fixed="str(Path('data') / p.relative_to(a.output))"
  original='str(p.relative_to(BASE))'
  current_text=current_builder.decode('utf-8')
  assert current_text.count(fixed)==1 and original not in current_text
  assert sha(current_text.replace(fixed,original).encode('utf-8'))==frozen['builder_sha256']
  comparisons=note['data_file_comparisons']
  assert len(comparisons)==8
  for name,comparison in comparisons.items():
   assert comparison['match'] is True
   assert comparison['frozen_sha256']==comparison['rebuilt_sha256']
   assert comparison['frozen_sha256']==frozen['files']['data/'+name]
   assert comparison['frozen_sha256']==sha((data/name).read_bytes())
  builder_revision='disclosed single expression reproduction fix verified'

 inputs=load(data/'inputs_public.jsonl');labels=load(data/'labels_public.jsonl')
 rules=load(data/'private/rules.jsonl');sources={r['rule_id']:r['text'] for r in load(data/'private/source_trees.jsonl')}
 assert len(inputs)==len(labels)==24 and len(rules)==4
 assert len({r['sample_id'] for r in inputs})==24
 assert {r['sample_id'] for r in inputs}=={r['sample_id'] for r in labels}
 assert all(set(row)=={'sample_id','rule_id','sentences'} for row in inputs)
 for row in inputs:
  label=next(x for x in labels if x['sample_id']==row['sample_id'])
  assert sha(json.dumps(row,sort_keys=True).encode())==label['input_sha256']
  assert label['label'] in {'T','F','U'}
  assert all(1<=i<=len(row['sentences']) for i in label['all_relevant_evidence_ids'])
  assert all(1<=i<=len(row['sentences']) for p in label['acceptable_evidence_proofs'] for i in p)
 for rule in rules:
  source=sources[rule['rule_id']]
  assert sha(source.encode())==rule['provenance']['source_tree_sha256']
  manifest=next(x for x in load(data/'rule_manifest_public.jsonl') if x['rule_id']==rule['rule_id'])
  assert rule['logic']==manifest['logic']
  action=rule['action_span']; assert rule['text'][action['start']:action['end']]==rule['action']
  original=rule['original_action_span'];assert json.loads('"'+source[original['start']:original['end']]+'"')==rule['action']
  for condition in rule['conditions']:
   span=condition['source_span'];assert rule['text'][span['start']:span['end']]==condition['text']
   original=condition['original_source_span']
   original_text=json.loads('"'+source[original['start']:original['end']]+'"')
   def nodes(node):
    yield node
    for branch in ('yes','no'):
     if isinstance(node.get(branch),dict):yield from nodes(node[branch])
   node=next(x for x in nodes(json.loads(source)['decision_tree']) if x['index']==condition['source_node_index'])
   assert original_text in node['description']
   assert sha(original_text.encode())==condition['source_node_fragment_sha256']
 pairs=load(data/'pairs_public.jsonl'); assert len(pairs)==4
 byid={r['sample_id']:r for r in labels}
 for pair in pairs:
  assert byid[pair['original_id']]['label']=='T' and byid[pair['counterfactual_id']]['label']=='F'
 audit=json.loads((data/'history_audit.json').read_text());assert audit['matching_selected_ids']==[]
 print(json.dumps({'status':'passed','samples':24,'rules':4,'counterfactual_pairs':4,'source_rule_action_spans':'exact','frozen_inputs_sha256':'verified','builder_revision':builder_revision,'model_run':False}))

if __name__=='__main__':main()
