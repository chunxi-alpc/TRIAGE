import unittest
from src.acre_pilot.evaluator import evaluate_cases
class OARTests(unittest.TestCase):
 def test_source_reviewed_alias_counts_as_same_misuse(self):
  c={'case_id':'x','annotation_errors':[],'guideline_id':1,'options':[{'label':'A','action':'wrong'},{'label':'B','action':'wrong paraphrase'},{'label':'C','action':'right'}],'annotations':[{'rule_id':'r','applicability':'inapplicable','mapped':{}}],'variants':{'B0':{'prediction':'A'},'B3':{'prediction':'B'}}}
  r={'x':{'gold_action':'right','target_rule_id':'r','target_action':'wrong','target_action_aliases':['wrong','wrong paraphrase'],'target_applicability':'inapplicable','incompatible_actions_verified':True}}
  m,_,_=evaluate_cases([c],r);self.assertEqual(m['B0']['oar']['numerator'],1);self.assertEqual(m['B3']['oar']['numerator'],1)
if __name__=='__main__':unittest.main()
