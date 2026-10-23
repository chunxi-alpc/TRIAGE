import collections,json,unittest
from pathlib import Path
from analyze import cell,paired
P=Path('/home/gcx/qwen800_methods_20261023/private')
class ProtocolTest(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.cases=list(map(json.loads,(P/'cases.jsonl').read_text().splitlines()))
 def test_frozen_ids(self):
  source=list(map(json.loads,Path('/home/gcx/medguide_expand_cs_20261023/private/cohort.jsonl').read_text().splitlines()));self.assertEqual({r['id'] for r in source},{r['id'] for r in self.cases});self.assertEqual(len(self.cases),800);self.assertEqual(collections.Counter(r['stratum'] for r in self.cases),{'K':426,'U':374})
 def test_interventions(self):
  for c in self.cases:
   a=c['variant_rules']['A'];b=c['variant_rules']['B'];self.assertEqual([x['condition'] for x in a],[x['condition'] for x in b]);self.assertNotEqual(c['gold_answer']['A'],c['gold_answer']['B'])
   changed=[i for i,(x,y) in enumerate(zip(a,b)) if x!=y];self.assertEqual(changed,[c['app_rule']])
   cc=c['variant_rules']['C'];self.assertEqual([i for i,(x,y) in enumerate(zip(a,cc)) if x!=y],[c['sib_rule']]);self.assertEqual(c['gold_answer']['C'],c['gold_answer']['A']);self.assertNotEqual(c['app_branch'],c['sib_branch'])
   if 'D' in c['variant_rules']:
    d=c['variant_rules']['D'];self.assertEqual(d[c['app_rule']]['action'],a[c['sib_rule']]['action']);self.assertEqual(d[c['sib_rule']]['action'],a[c['app_rule']]['action']);self.assertNotEqual(c['gold_answer']['D'],c['gold_answer']['A'])
 def test_gold_mapping(self):
  for c in self.cases:
   for v,rs in c['variant_rules'].items():self.assertEqual(rs[c['gold_rule'][v]]['action'],c['options'][c['gold_answer'][v]]);self.assertIn(c['gold_rule'][v],c['branches'][c['gold_branch'][v]]['rule_indices'])
 def test_cs_frozen(self):
  source={r['id']:r for r in map(json.loads,Path('experiments/medguide_expand_cs_20261023/results/cs_selected.jsonl').read_text().splitlines())};actual=[c for c in self.cases if 'CS' in c['variant_rules']];self.assertEqual({c['id'] for c in actual},set(source));self.assertEqual(len(actual),84)
  for c in actual:self.assertEqual(c['gold_answer']['CS'],source[c['id']]['new_gold'])
 def test_statistical_denominators(self):
  a=[{'id':'a','disease':'x','correct':True,'prediction':'A'},{'id':'b','disease':'x','correct':False,'prediction':None}];b=[{**a[0],'correct':False},{**a[1],'correct':True,'prediction':'B'}];self.assertEqual(cell(a)['accuracy'],.5);self.assertEqual(cell(a)['parse_failures'],1);self.assertIsNone(cell(a)['disease_ci']);self.assertEqual(paired(a,b)['wins'],1);self.assertEqual(paired(a,b)['losses'],1)
 def test_expected_coverage(self):self.assertEqual(collections.Counter(v for c in self.cases for v in c['variant_rules']),{'A':800,'B':800,'C':800,'D':797,'CS':84})
 def test_original200_reader(self):
  p=Path('experiments/aster_validation_20261022/mechanism_20261023/results/priority200/qwen/full_parser_v2/scores.jsonl');rs=list(map(json.loads,p.read_text().splitlines()));a=[r for r in rs if r['variant']=='A' and r['method']=='existing_baseline'];self.assertEqual(len(a),200);self.assertEqual(cell(a)['accuracy'],.655)
if __name__=='__main__':unittest.main()
