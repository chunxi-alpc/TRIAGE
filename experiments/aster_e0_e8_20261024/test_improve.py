import unittest
from improve_branch import judge_payload,condition_key,select_weights,mix,key
class ImprovementTests(unittest.TestCase):
 def test_judgment_has_no_actions_options_or_gold(self):
  c={'id':'x','patient':'age 20','rules':[{'rid':0,'condition':'adult','action':'hidden'}],'options':['hidden'],'gold_rule':0,'gold_option':0}
  self.assertEqual(judge_payload(c),{'patient':'age 20','conditions':[{'rid':0,'condition':'adult'}]})
  before=condition_key(c);c['gold_rule']=1;c['rules'][0]['action']='other';c['options']=['other'];self.assertEqual(condition_key(c),before)
 def test_heldout_labels_do_not_change_their_weights(self):
  rows=[{'id':str(i),'tree':str(i),'seed':0,'variant':'NC','gold_rule':0} for i in range(3)]
  ps={key(c):[[.8,.2],[.2,.8],[.4,.6]] for c in rows};a=select_weights(rows,ps);rows[0]['gold_rule']=1;b=select_weights(rows,ps)
  self.assertEqual(a['0'],b['0'])
 def test_fusion_probability(self):
  p=mix([[.8,.2],[.2,.8],[.5,.5]],[.5,.25,.25]);self.assertAlmostEqual(sum(p),1.);self.assertAlmostEqual(p[0],.575)
if __name__=='__main__':unittest.main()
