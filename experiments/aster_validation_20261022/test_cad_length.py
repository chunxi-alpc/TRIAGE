"""CPU regression: CAD can continue past eight tokens and obey multi-EOS config."""
import ast,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
class CADLength(unittest.TestCase):
 def test_long_output_and_model_eos(self):
  tree=ast.parse(Path(__file__).with_name('run.py').read_text());fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='cad_score')
  code=ast.unparse(fn).replace('range(8)','range(max_new_tokens)').replace('token == tok.eos_token_id','token in eos_ids')
  class FakeModel:
   device='cpu';calls=0
   def __call__(self,**kw):
    step=self.calls//2;self.calls+=1
    logits=torch.zeros(1,1,4);logits[0,0,3 if step==10 else 1]=10
    return SimpleNamespace(logits=logits,past_key_values=None)
  model=FakeModel();tok=SimpleNamespace(eos_token_id=2,decode=lambda ids,**kw:','.join(map(str,ids)))
  ns=dict(torch=torch,model=model,tok=tok,max_new_tokens=128,eos_ids={2,3},encode_prompt=lambda *a:({'input_ids':torch.ones((1,1),dtype=torch.long)},[]),tokenids={'A':[0]},parse_answer=lambda *a:None)
  exec(compile(code,'cad_test','exec'),ns)
  r=ns['cad_score']('context','closed',['A'])
  self.assertEqual(len(r['completion'].split(',')),11)
  self.assertEqual(model.calls,22)
  model.calls=0;ns['max_new_tokens']=5;r=ns['cad_score']('context','closed',['A']);self.assertEqual(len(r['completion'].split(',')),5)
if __name__=='__main__':unittest.main()
