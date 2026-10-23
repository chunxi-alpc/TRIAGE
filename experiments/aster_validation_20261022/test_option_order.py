import unittest
from option_order import rotate
class OrderTest(unittest.TestCase):
 def test_roundtrip_and_gold(self):
  opts={'A':'first','B':'second','C':'third'};rule='rule one';sib='rule two';prompt=rule+'\n'+sib+'\nOptions:\nA: first\nB: second\nC: third'
  row=dict(id='case',options=opts,prompt_text=prompt,closed_book_prompt='Options:\nA: first\nB: second\nC: third',gold_answer='C',original_answer='A',sibling_action_label='C',replacement_action_label='C',original_sibling_action_label='B',applicable_char_span=[0,8],sibling_char_span=[9,17],applicable_rule_text=rule,sibling_rule_text=sib)
  for shift in [1,2]:
   r,back=rotate(row,shift)
   for field in ['gold_answer','original_answer','sibling_action_label','replacement_action_label','original_sibling_action_label']:
    self.assertEqual(back[r[field]],row[field]);self.assertEqual(r['options'][r[field]],opts[row[field]])
   self.assertTrue(all(back[k]!=k for k in opts));self.assertEqual(r['prompt_text'][:17],prompt[:17])
   restored,_=rotate(r,3-shift);self.assertEqual(restored,row)
if __name__=='__main__':unittest.main()
