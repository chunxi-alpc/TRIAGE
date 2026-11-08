import numpy as np
import torch
from .features import orth_nuisance
from .readout import optimize,predict
def test_signal_nuisance_orthogonality():
 rng=np.random.default_rng(7);s=np.linalg.qr(rng.normal(size=(12,3)))[0]
 n=np.linalg.qr(rng.normal(size=(12,4)))[0];q=orth_nuisance(s,n)
 assert np.max(np.abs(s.T@q))<1e-5
 assert np.allclose(q.T@q,np.eye(q.shape[1]),atol=1e-5)
def test_calibration_and_parameter_count():
 ell=np.array([-2.,-1.,.2,1.]);y=np.array([0,0,0,1])
 fit=optimize(ell,np.empty((4,0)),y,[],(ell>0)==y,0,{},"bias")
 assert fit["parameters"]==1 and np.isfinite(fit["b"])
class TinyTokenizer:
 def encode(self,text,add_special_tokens=False):return [{"\u662f":1,"\u5426":2,"A":3,"B":4}.get(text,5)]
 def apply_chat_template(self,*args,**kwargs):return "tiny"
 def __call__(self,text,**kwargs):return {"input_ids":torch.tensor([[5,6,7,8]])}
def tiny_engine():
 from transformers import Qwen3Config,Qwen3ForCausalLM
 from .engine import Engine
 config=Qwen3Config(vocab_size=16,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
  num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=128,attention_dropout=0.)
 return Engine("qwen",model=Qwen3ForCausalLM(config),tokenizer=TinyTokenizer(),device="cpu")
def test_adapter_gradients_and_cache_replay(tmp_path):
 from .baselines import Adapter
 for kind in ("loreft","lofit","lora","last_layer"):
  torch.manual_seed(0);eng=tiny_engine();ad=Adapter(eng,kind,17)
  eng.model.train();eng.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False});eng.model.enable_input_require_grads()
  for stage in range(2 if kind=="lofit" else 1):
   params=ad.parameters();opt=torch.optim.SGD(params,lr=.01)
   value=ad.score_train("x");loss=torch.nn.functional.binary_cross_entropy_with_logits(value,torch.tensor(1.))
   loss.backward()
   assert any(p.grad is not None and p.grad.abs().sum()>0 for p in params),kind
   assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params),kind
   opt.step();opt.zero_grad()
   if kind=="lofit" and stage==0:ad.select()
  eng.model.eval();eng.model.gradient_checkpointing_disable();ad.active=True
  dense=float(ad.score_train("x").detach());cached=ad.score("x")
  assert abs(dense-cached)<1e-4,(kind,dense,cached)
  path=tmp_path/(kind+".pt");ad.dump(path)
  assert path.exists()
  ad.close()
  torch.manual_seed(0);other=tiny_engine();restored=Adapter(other,kind,17,load=path)
  assert abs(restored.score("x")-cached)<1e-4,kind
  restored.close()

def test_full_signal_span_has_no_nuisance_space():
 rng=np.random.default_rng(9);s=np.linalg.qr(rng.normal(size=(24,24)))[0].astype(np.float32)
 n=np.linalg.qr(rng.normal(size=(24,8)))[0].astype(np.float32)
 assert orth_nuisance(s,n).shape==(24,0)

def test_h3_historical_errors_keep_separate_accuracy_denominator(tmp_path,monkeypatch):
 from . import mechanisms
 from .common import write,lines,read
 rng=np.random.default_rng(4)
 views=[dict(view_id=str(i),split="fit" if i<20 else "dev",variant="full",label=i%2,source_id=str(i)) for i in range(26)]
 head={v["view_id"]:rng.normal(size=(2,4,8)).astype(np.float32) for v in views}
 residual={v["view_id"]:rng.normal(size=32).astype(np.float32) for v in views}
 class Fake:
  def __init__(self,model):self.views=views
  def ell(self,vid):return -1 if int(vid)%2 else 1
  def matrix(self,vs,heads=None,kind="heads"):
   if kind=="residual":return np.stack([residual[v["view_id"]] for v in vs])
   return np.stack([(head[v["view_id"]].reshape(8,8) if heads is None else head[v["view_id"]].reshape(8,8)[heads]).reshape(-1) for v in vs])
 monkeypatch.setattr(mechanisms,"ROOT",tmp_path);monkeypatch.setattr(mechanisms,"Features",Fake)
 write(tmp_path/"qwen/selection.json",dict(heads=[0,1]))
 feature=tmp_path/"historical.npz";np.savez(feature,heads=rng.normal(size=(2,4,8)),residual=rng.normal(size=32))
 lines(tmp_path/"qwen/historical_h3/scores.jsonl",[dict(status="ok",current_error=True,feature=str(feature),label=1)])
 mechanisms.h3("qwen")
 result=read(tmp_path/"qwen/h3.json")["probes"]["selected_heads"]
 assert result["dev_n"]==6 and result["historical_error_nodes"]==1 and result["error_nodes"]==7
