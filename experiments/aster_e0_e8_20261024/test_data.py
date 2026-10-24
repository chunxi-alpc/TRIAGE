import json,unittest
from pathlib import Path
P=Path('/home/gcx/aster_e0_e8_20261024/private')
class DataTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  if not (P/'cases.jsonl').exists():raise unittest.SkipTest('private dataset absent')
  cls.rows=list(map(json.loads,(P/'cases.jsonl').read_text().splitlines()))
 def test_split_and_fixed_options(self):
  dev={r['tree'] for r in self.rows if r['split']=='dev'};test={r['tree'] for r in self.rows if r['split']=='test'};self.assertFalse(dev&test)
  seen={}
  for r in self.rows:
   key=(r['id'],r['seed']);self.assertEqual(seen.setdefault(key,r['options']),r['options']);self.assertEqual(r['options'][r['gold_option']],r['rules'][r['gold_rule']]['action'])
 def test_changes_are_local(self):
  groups={}
  for r in self.rows:groups.setdefault((r['id'],r['seed']),{})[r['variant']]=r
  for g in groups.values():
   base=g['NC'];gr=base['gold_rule'];sr=base['decoy_rule']
   for variant in ['AC','IC','SW']:
    changed={j for j,(a,b) in enumerate(zip(base['rules'],g[variant]['rules'])) if a!=b};self.assertEqual(changed,{gr} if variant=='AC' else {sr} if variant=='IC' else {gr,sr})
   for variant in ['CS','CS-null']:
    if variant in g:
     r=g[variant];self.assertEqual([x['action'] for x in base['rules']],[x['action'] for x in r['rules']]);self.assertEqual(r['gold_rule']!=gr,variant=='CS');self.assertNotEqual(r['threshold_proof']['threshold'],r['threshold_proof']['old_threshold'])
 def test_annotation_unique(self):
  m=json.loads((P/'annotation_manifest.json').read_text());self.assertEqual(len({x.split(':')[0] for x in m['ids']}),150)
if __name__=='__main__':unittest.main()
