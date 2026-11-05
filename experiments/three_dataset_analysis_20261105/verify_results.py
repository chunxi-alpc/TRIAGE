"""Required checks only: option mapping, pairing, splits, completeness, and fusion counts."""
import json
from collections import Counter
from common import *
def main():
 inp=rows(ROOT/'data/medpair_inputs.jsonl');views=rows(ROOT/'results/medpair_complete_views.jsonl')
 assert len(inp)==538 and len(views)==540
 assert Counter(r['split'] for r in views if r['condition']=='Full')==dict(dev=91,test=89)
 assert Counter(r['split'] for r in views if r['condition']=='C')==dict(dev=31,test=29)
 for r in views:
  assert len(r['labels'])==len(r['raw_logits'])==len(r['log_probabilities'])
  assert r['gold'] in r['labels'] and r['prediction']==r['labels'][int(np.argmax(r['raw_logits']))]
  assert np.allclose(lp(r['raw_logits']),r['log_probabilities'],atol=1e-5,rtol=0)
 v={(r['id'],r['condition']):r for r in views}
 for r in views:
  if r['condition']!='C':continue
  h=v[r['id'],'C+h'];u=v[r['id'],'C+u'];pos=h['insertion_position']
  hs=h['evidence_sids'].copy();us=u['evidence_sids'].copy()
  assert hs.pop(pos)==h['h_sid'] and us.pop(pos)==u['u_sid'] and hs==us==r['evidence_sids']
 guide=rows(ROOT/'data/medguide_scores_and_inputs.jsonl');by={(r['id'],r['variant'],r['view']):r for r in guide}
 assert len(guide)==6562
 split=json.loads((ROOT/'data/medguide_split.json').read_text())
 assert not(set(split['dev_trees'])&set(split['test_trees']))
 assert Counter(split['case_split'].values())==dict(dev=149,test=651)
 for r in guide:
  assert np.allclose(lp([r['raw_logits'][k] for k in r['labels']]),r['log_probabilities'])
  assert split['case_split'][r['id']]==r['split']
  if r['variant'] in ('B','C','D') and r['view']=='hidden':
   a=by[r['id'],'A','hidden'];assert r['prompt']==a['prompt'] and r['prediction']==a['prediction']
 einst=rows(ROOT/'results/medeinst_independent_predictions.jsonl')
 assert len(einst)==6144
 for r in einst:
  assert r['pred_a']==r['labels'][int(np.argmax(r['scores_a']))]
  assert r['pred_b']==r['labels'][int(np.argmax(r['scores_b']))]
  i=r['labels'].index(r['gold_a']);j=r['labels'].index(r['gold_b'])
  assert abs(r['delta']-((r['scores_b'][j]-r['scores_b'][i])-(r['scores_a'][j]-r['scores_a'][i])))<1e-9
 for name in ('medpair_fusion.json','medguide_fusion.json'):
  for r in json.loads((ROOT/'results'/name).read_text()):
   assert r['B']['numerator']==r['same_wrong']['numerator']+r['different_wrong']['numerator']
   assert r['corrected']['numerator']==r['same_wrong_corrected']['numerator']+r['different_wrong_corrected']['numerator']
   assert r['accuracy']['numerator']==r['harm']['denominator']-r['harm']['numerator']+r['corrected']['numerator']
   if r['dataset']=='MedGUIDE' and r['variant']=='CS':assert r['accuracy']['ci95'] is None
 for n in ('REPORT.zh.md','table1.md','table2.md','table3.md','table4.md'):
  s=(ROOT/'results'/n).read_text(encoding='utf-8');assert any('\u4e00'<=c<='\u9fff' for c in s) and '????' not in s
 write_json(ROOT/'results/verification.json',dict(status='passed',unit_tests=4,medpair_unique_inputs=538,medpair_views=540,medguide_branch_rows=6562,medeinst_primary_pairs_per_model=1024,normalization=True,same_question_fusion=True,addback_same_position=True,development_test_separation=True,fusion_count_identities=True,CS_single_tree_interval=None,unicode_reports=True))
if __name__=='__main__':main()
