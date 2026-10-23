import copy,unittest
from d_option_order import rotated
from run import branch_prompt,prune_prompt,parse
class RotationTest(unittest.TestCase):
 def test_rotation_changes_only_choices(self):
  c=dict(options={'A':'alpha','B':'beta'},closed_book_prompt='Patient X\nOptions:\nA: alpha\nB: beta\nReturn only one option letter.',patient='Patient X',variant_rules={'D':[{'condition':'x','action':'alpha'},{'condition':'y','action':'beta'}]},branches=[{'rule_indices':[0]},{'rule_indices':[1]}],gold_branch={'D':0})
  before=copy.deepcopy(c);r=rotated(c,{'A':'B','B':'A'})
  self.assertEqual(c,before);self.assertEqual(r['options'],{'A':'beta','B':'alpha'})
  for visibility in ('hidden','visible'):self.assertEqual(branch_prompt(c,'D',visibility),branch_prompt(r,'D',visibility))
  self.assertEqual(prune_prompt(c,'D',bs=[0]).split('Options:')[0],prune_prompt(r,'D',bs=[0]).split('Options:')[0])
 def test_parser_explicit_first_line(self):
  self.assertEqual(parse('**Answer: B**\nExplanation',list('AB')),'B')
  self.assertIsNone(parse('A patient needs treatment',list('AB')))
 def test_reject_missing_option_block(self):
  with self.assertRaises(AssertionError):rotated({'options':{'A':'alpha'},'closed_book_prompt':'invalid'},{'A':'A'})
if __name__=='__main__':unittest.main()
