"""Small actual-model validation before the long repaired-data run."""
import argparse,time
import numpy as np
from .common import *
def run(model):
 import torch
 from .engine import Engine
 from .external_score import EXTERNAL_SYSTEM,parse_answer
 torch.set_num_threads(4);start=time.time();eng=Engine(model)
 views=[v for v in rows(ROOT/"data/views.jsonl") if v["split"]=="fit"]
 chosen=[next(v for v in views if v["kind"]==kind) for kind in ("original","A","P","C","F","F0")]
 longest=max(views,key=lambda v:len(v["prompt"]))
 if longest not in chosen:chosen.append(longest)
 existing={r["view_id"]:r for p in (ROOT/model/"scores").glob("fit_*.jsonl") for r in rows(p)}
 results=[]
 for v in chosen:
  t=time.time();r=eng.score(v["prompt"],capture=True,grad=True)
  assert all(torch.isfinite(r[k]).all() for k in ("heads","residual","grads"))
  assert np.isfinite(r["ell"])
  delta=abs(r["ell"]-existing[v["view_id"]]["ell"]) if v["view_id"] in existing else None
  if delta is not None:assert delta<1e-5,(v["view_id"],delta)
  results.append(dict(view_id=v["view_id"],kind=v["kind"],tokens=r["tokens"],ell=r["ell"],cache_score_difference=delta,seconds=time.time()-t,heads_shape=list(r["heads"].shape)))
 # A synthetic format example validates native multi-label generation without scoring external test items.
 prompt='Return exactly this JSON array: ["A", "C"]'
 text=eng.tokenizer.apply_chat_template([dict(role="system",content=EXTERNAL_SYSTEM),dict(role="user",content=prompt)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
 inp=eng.tokenizer(text,return_tensors="pt",add_special_tokens=False).to(eng.device)
 with torch.inference_mode():out=eng.model.generate(**inp,max_new_tokens=32,do_sample=False,logits_to_keep=1,pad_token_id=eng.tokenizer.eos_token_id)
 text=eng.tokenizer.decode(out[0,inp["input_ids"].shape[1]:],skip_special_tokens=True)
 parsed=parse_answer("medpic",text);assert parsed==["A","C"],text
 write(ROOT/"verification"/(model+"_forward.json"),dict(passed=True,records=results,external_format_fixture=parsed,peak_gib=torch.cuda.max_memory_allocated()/2**30,seconds=time.time()-start))
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--model",default="qwen");run(p.parse_args().model)
