import json,unittest,copy
from pathlib import Path
import torch
from transformers import AutoTokenizer
from src.acre_pilot.constrained_annotation import JSONTokenGrammar,evidence_catalog,normalize,payload_for
from src.acre_pilot.pair_verification import parse_threshold,convert,interval,infer_boolean,verify_pair
class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  path=json.loads(Path('configs/models.json').read_text())['models']['qwen3_8b']['path'];cls.t=AutoTokenizer.from_pretrained(path,local_files_only=True)
 def test_numeric_words_units_intervals(self):
  self.assertEqual(parse_threshold('Is age at least 65 years?'),{'operator':'>=','threshold':65.})
  self.assertEqual(parse_threshold('血钾低于5.0 mmol/L'),{'operator':'<','threshold':5.})
  self.assertEqual(convert(18,'months','years'),1.5);self.assertEqual(convert(5,'mEq/L','mmol/L',substance='potassium'),5)
  with self.assertRaises(ValueError):convert(5,'mg/dL','mmol/L')
  self.assertTrue(interval(50,50,70));self.assertFalse(interval(50,50,70,False))
 def test_exact_catalog_decimal_repetition(self):
  p='K is 4.3 mmol/L. MRI unavailable. MRI unavailable.';rows=evidence_catalog(p)
  self.assertEqual(rows[0]['text'],'K is 4.3 mmol/L.')
  self.assertEqual(len(rows),3)
  for r in rows:self.assertEqual(p[r['start']:r['end']],r['text'])
  raw=json.dumps({'rules':[{'rule_id':'r','relevant':True,'applicability':'inapplicable','support_ids':[None],'refute_ids':[2]}]})
  out=normalize(raw,[{'rule_id':'r','action':'Scan'}],rows)
  self.assertEqual(out[0]['refuting_evidence'][0]['start'],rows[2]['start'])
 def test_all_grammar_segments_and_eos(self):
  rows=[{'rule_id':'original','action':'Scan'}];catalog=evidence_catalog('MRI unavailable. CT available.')
  for choose in (0,-1):
   g=JSONTokenGrammar(self.t,rows,catalog,2);g.prompt_length=0;g.eos=self.t.eos_token_id;ids=[]
   for segment in g.segments:
    for token in segment[choose]:
     self.assertIn(token,g.allowed(0,torch.tensor(ids,dtype=torch.long)));ids.append(token)
   self.assertEqual(g.allowed(0,torch.tensor(ids)),[g.eos])
   value=json.loads(self.t.decode(ids));self.assertEqual(value['rules'][0]['rule_id'],'original');normalize(self.t.decode(ids),rows,catalog)
 def test_out_of_range_and_unknown_fields_rejected(self):
  r={'rules':[{'rule_id':'r','relevant':True,'applicability':'applicable','support_ids':[999],'refute_ids':[]}]}
  with self.assertRaises(ValueError):normalize(json.dumps(r),[{'rule_id':'r','action':'Scan'}],[])
 def test_boolean_conflict_rejected(self):
  with self.assertRaises(ValueError):infer_boolean('MRI available. MRI unavailable.',r'MRI available',r'MRI unavailable')
 def test_gold_absent_from_payload_and_versioned_cache(self):
  class B:model_path='m'
  item={'guideline':{'rules':[{'rule_id':'r','text':'When needed, scan.','action':'scan.'}]},'patient':'MRI unavailable.','question':'Action?','gold_action':'SECRET','reference_path':'SECRET'}
  p,k=payload_for(item,B(),{});self.assertNotIn('SECRET',json.dumps(p));self.assertIn('source_ids_json_grammar.',p['version'])
 def test_expanded_labels_single_token_and_complete_actions(self):
  from src.acre_pilot.span_mapping import options_for
  item={'decision_protocol':'single_token_label.v2','guideline':{'rules':[{'action':str(i)} for i in range(40)]}}
  options=options_for(item);self.assertEqual(len(options),41)
  self.assertTrue(all(len(self.t.encode(o['label'],add_special_tokens=False))==1 for o in options))
 def test_compact_prompt_annotation_roundtrip(self):
  from src.acre_pilot.constrained_annotation import compact_for_prompt
  catalog=evidence_catalog('MRI unavailable. CT available.')
  raw=json.dumps({'rules':[{'rule_id':'r','relevant':True,'applicability':'inapplicable','support_ids':[1,None],'refute_ids':[0,None]}]})
  rows=normalize(raw,[{'rule_id':'r','action':'MRI.'}],catalog)
  obj=compact_for_prompt({'input':{'catalog':catalog},'rules':rows})
  self.assertEqual(obj['rules'][0]['refuting_evidence'],[0])
  self.assertEqual(obj['patient_evidence_catalog'][0]['text'],'MRI unavailable.')
 def test_applicability_choices_share_exact_quote_context(self):
  g=JSONTokenGrammar(self.t,[{'rule_id':'r'}],[],1)
  states=[self.t.encode(s,add_special_tokens=False) for s in ('applicable','inapplicable','unknown')]
  self.assertIn(states,g.segments)
  segment=g.segments.index(states)
  prefix=[t for alternatives in g.segments[:segment] for t in alternatives[0]]
  self.assertTrue(self.t.decode(prefix).endswith('"applicability":"'))
  g.prompt_length=0;g.eos=self.t.eos_token_id
  allowed=g.allowed(0,torch.tensor(prefix))
  self.assertEqual(set(allowed),{s[0] for s in states})
 def test_frozen_pairs_and_invalid_audits(self):
  reviews=json.loads(Path('results/acre_pilot/round2/pair_reviews.json').read_text());d=Path(json.loads(Path('configs/datasets.json').read_text())['MEGA-CDP']['path'])
  paths={r['guideline_id']:r['paths'] for r in [json.loads(l) for l in (d/'train_val/case_paths.jsonl').read_text().splitlines()]};cases={(r['guideline_id'],r['path_id']):r['text'] for r in [json.loads(l) for l in (d/'train_val/vignette_cases.jsonl').read_text().splitlines()]}
  self.assertEqual(len(reviews),40);self.assertEqual(len({r['pair_id'] for r in reviews}),40)
  for r in reviews:
   gid=r['guideline_id'];pid=r['source_path_id'];source={'patient':cases[(gid,pid)],'path':paths[gid][pid-1]};v=verify_pair(r,source,paths[gid]);self.assertTrue(v['tree_route_verified'])
   bad=copy.deepcopy(r);bad['new']=bad['old']
   with self.assertRaises(ValueError):verify_pair(bad,source,paths[gid])
   bad=copy.deepcopy(r);bad['incompatible_actions']=False
   with self.assertRaises(ValueError):verify_pair(bad,source,paths[gid])
if __name__=='__main__':unittest.main()
