import unittest
from types import SimpleNamespace
import torch
from .structured import terminal_paths, decode

class Tokenizer:
    def encode(self, text, **kwargs): return list(text.encode('ascii'))
    def decode(self, ids, **kwargs): return bytes(ids).decode('ascii')

class Backend:
    def __init__(self, prefer): self.prefer=prefer; self.calls=[]
    def result(self):
        logits=torch.zeros(1,1,128)
        logits[0,0,ord('Z')]=100.  # An illegal token must never be emitted.
        logits[0,0,ord(self.prefer)]=10.
        return SimpleNamespace(logits=logits,past_key_values=object())
    def _forward(self, ids, **kwargs):
        self.calls.append((ids[0].tolist(),kwargs['start']))
        return self.result()

class StructuredTests(unittest.TestCase):
    def setUp(self):
        self.guide={'groups':[
          {'group_id':'G0','rules':[{'rule_id':'R0Y','action':'继续评估规则组 G1。'},
                                  {'rule_id':'R0N','action':'L9：stop'}]},
          {'group_id':'G1','rules':[{'rule_id':'R1Y','action':'L2：yes'},
                                  {'rule_id':'R1N','action':'L3：no'}]}]}
    def run_decode(self, preference, budget=512):
        back=Backend(preference)
        engine=SimpleNamespace(builder=SimpleNamespace(tokenizer=Tokenizer()),backend=back,
                               freeze={'max_new_tokens':budget})
        return decode(engine,self.guide,back.result(),17),back
    def test_all_full_tree_routes_are_available(self):
        self.assertEqual(terminal_paths(self.guide),[(['R0Y','R1Y'],'L2'),(['R0Y','R1N'],'L3'),(['R0N'],'L9')])
    def test_terminal_stops_and_final_is_looked_up(self):
        r,_=self.run_decode('N')
        self.assertEqual(r['text'],'PATH: R0N\nFINAL: L9')
        self.assertNotIn('R1',r['text'])
    def test_legal_transition_and_cache_positions(self):
        r,back=self.run_decode('Y')
        self.assertEqual(r['text'],'PATH: R0Y, R1Y\nFINAL: L2')
        pos=17
        for tokens,start in back.calls:
            self.assertEqual(start,pos);pos+=len(tokens)
        self.assertNotIn(ord('Z'),r['path_token_ids'])
    def test_limit_is_not_falsely_completed(self):
        r,_=self.run_decode('Y',budget=4)
        self.assertTrue(r['output_limit']);self.assertIsNone(r['selected_leaf'])
        self.assertNotIn('FINAL',r['text'])

if __name__=='__main__':unittest.main()
