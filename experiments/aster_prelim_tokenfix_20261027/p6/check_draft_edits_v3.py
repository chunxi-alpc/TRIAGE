"""Check offset and reference replacement behavior without model inference."""
import tempfile
from pathlib import Path
import draft_action_pairs_v3 as m
def main():
 with tempfile.TemporaryDirectory(prefix='p6-v3-edit-check-') as t:
  root=Path(t);guide='If condition A, do action Alpha. Otherwise, do action Beta.'
  (root/'guide.md').write_text(guide);old=m.C;m.C=root
  c={'pair_id':'synthetic','native_guideline_file':'guide.md','native_reference_path':['Condition A holds; do action Alpha.','Final plan: do action Alpha.']}
  data={'ORIGINAL_SOURCE_EXCERPTS':[{'source_id':'S1','text':guide,'start':0}],'NATIVE_REFERENCE_STEPS':[{'reference_id':'R1','text':c['native_reference_path'][0]},{'reference_id':'R2','text':c['native_reference_path'][1]}]}
  j={'eligible':True,'applicable_source_id':'S1','applicable_action_quote':'do action Alpha','sibling_source_id':'S1','sibling_action_quote':'do action Beta','reference_edits':[{'reference_id':'R1','action_quote':'do action Alpha'},{'reference_id':'R2','action_quote':'do action Alpha'}]}
  patch,status=m.construct(c,data,j)
  assert status=='EXACT_TEXT_DRAFT_PENDING_HUMAN'
  assert patch['decoy_reference_path']==['Condition A holds; do action Beta.','Final plan: do action Beta.']
  edited=guide
  for e in reversed(patch['guide_edits']):edited=edited[:e['start']]+e['after']+edited[e['end']:]
  assert edited=='If condition A, do action Beta. Otherwise, do action Alpha.'
  assert patch['review_status']=='PENDING_HUMAN_REVIEW' and patch['conditions_semantically_unchanged']=='UNVERIFIED'
  bad=dict(j);bad['sibling_source_id']='S999'
  try:m.construct(c,data,bad);raise AssertionError('Invalid source accepted')
  except KeyError:pass
  m.C=old
 print('Reference IDs, repeated action references, exact offsets and invalid source checks passed; synthetic only.')
if __name__=='__main__':main()
