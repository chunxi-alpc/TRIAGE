import importlib.util,unittest,tempfile,argparse,json
from pathlib import Path
D=Path(__file__).resolve().parents[1]/'scripts'
def load(name):
 spec=importlib.util.spec_from_file_location(name,D/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
class Regressions(unittest.TestCase):
 def test_order_independent_groups(self):
  m=load('prepare_group_splits');g=['c','a','b','d'];f={'fit':.5,'test':.5}
  self.assertEqual(m.allocate(g,f,7),m.allocate(list(reversed(g)),f,7))
 def test_nonfinite_fraction(self):
  m=load('prepare_group_splits')
  with self.assertRaises(ValueError):m.allocate(['a','b'],{'a':float('nan'),'b':.5},1)
 def test_stats_empty_and_duplicate_fail(self):
  m=load('paired_cluster_stats')
  with self.assertRaises(ValueError):m.summarize([])
  r={'case_id':'a','guideline_id':'g','baseline_correct':False,'method_correct':False}
  with self.assertRaises(ValueError):m.summarize([r,r])
 def test_split_refuses_history_overwrite(self):
  m=load('prepare_group_splits')
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'out').mkdir();(root/'m.json').write_text(json.dumps({'group_to_split':{'g':'fit'}}));(root/'in.jsonl').write_text('{"g":"g"}\n')
   with self.assertRaises(FileExistsError):m.apply(argparse.Namespace(input=root/'in.jsonl',output=root/'out',group_key='g',manifest=root/'m.json'))
