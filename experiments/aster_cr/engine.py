"""One canonical score protocol for every new input, including interventions."""
import torch,time
from experiments.m1_diagnostics.engine import Engine as Base
SYSTEM="根据给定规则和病例，判断当前条件是否成立。只输出用户指定的答案标签。For English inputs, 是 means Yes and 否 means No."
class Engine(Base):
 def render(self,prompt,task="binary"):
  return self.tokenizer.apply_chat_template([dict(role="system",content=SYSTEM),dict(role="user",content=prompt)],
    tokenize=False,add_generation_prompt=True,enable_thinking=False)
 def score(self,prompt,labels=None,capture=True,replacements=None,grad=False):
  started=time.perf_counter();context=self.prepare(prompt);residual={}
  def hook(module,args,out):
   value=out[0] if isinstance(out,tuple) else out
   residual["value"]=value[0,-1].detach().float().cpu()
  handle=self.layers[-1].register_forward_hook(hook) if capture else None
  try:
   ids=None
   if labels:
    tokens=[self.tokenizer.encode(s,add_special_tokens=False) for s in labels]
    if any(len(t)!=1 for t in tokens):raise ValueError("Labels must be single tokens")
    ids=[t[0] for t in tokens]
   result=self.run(context,capture=capture,replacements=replacements,label_ids=ids,grad=grad)
   if ids is not None:
    result["ell"]=result["label_logits"][0]-result["label_logits"][1]
    result["prediction"]=int(result["ell"]>0)
   if capture:result["residual"]=residual["value"]
   result["inference_seconds"]=time.perf_counter()-started
   return result
  finally:
   if handle:handle.remove()
