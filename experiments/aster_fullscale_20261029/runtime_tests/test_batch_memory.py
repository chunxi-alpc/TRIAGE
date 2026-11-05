import sys,weakref,unittest,argparse
from pathlib import Path
import torch
from transformers import BatchEncoding
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from batch_engine import BatchEngine
class Tokenizer:
 eos_token_id=0
 def __call__(self,text,**kw):return {'input_ids':[1]*len(text)}
 def pad(self,rows,**kw):
  width=max(map(len,rows['input_ids']));ids=torch.tensor([[0]*(width-len(r))+r for r in rows['input_ids']]);return BatchEncoding({'input_ids':ids,'attention_mask':(ids!=0).long()})
 def decode(self,row,**kw):return 'A'
class Model:
 device=torch.device('cpu');config=argparse.Namespace(max_position_embeddings=40960)
 def __init__(self,force_oom):self.force_oom=force_oom;self.large=None
 def generate(self,input_ids,**kw):
  if self.force_oom and input_ids.shape[0]>2:
   self.large=weakref.ref(input_ids);raise torch.OutOfMemoryError('forced transport OOM')
  if self.large is not None and self.large() is not None:raise AssertionError('failed parent batch still holds its input tensor')
  return torch.cat([input_ids,torch.ones((len(input_ids),1),dtype=torch.long)],1)
def engine(force_oom):
 e=BatchEngine.__new__(BatchEngine);e.tok=Tokenizer();e.model=Model(force_oom);e.prefix_cache={};e.reuse_prefixes=False;e.active=None
 e.batch_stats={'calls':0,'rows':0,'oom_splits':0,'seconds':0.,'prefix_builds':0,'prefix_reuses':0,'prefix_rows':0,'token_budget_splits':0};return e
class MemoryRecovery(unittest.TestCase):
 def test_oom_releases_parent_and_preserves_every_row(self):
  e=engine(True);result=e.generate_many(['x'*10]*4,16,batch_size=4)
  self.assertEqual(result,['A']*4);self.assertEqual(e.batch_stats['oom_splits'],1);self.assertEqual(e.batch_stats['rows'],4)
 def test_long_input_is_split_without_truncation(self):
  e=engine(False);result=e.generate_many(['x'*20000]*4,16,batch_size=4)
  self.assertEqual(result,['A']*4);self.assertEqual(e.batch_stats['token_budget_splits'],1);self.assertEqual(e.batch_stats['rows'],4)
if __name__=='__main__':unittest.main()
