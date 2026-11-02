"""One native end-to-end and one real two-view training-case timing."""
import json,time,gc
from pathlib import Path
import torch
from .core import rows,nodes,deployment,swap_actions,make_module,objective,write_json
from .engine import Engine

def run(root):
 root=Path(root);engine=Engine.load(root/"model",root)
 cases=rows(root/"data/train.jsonl")
 def size(c):return len(c["patient"])+sum(len(r["condition"])+len(r["action"]) for g in c["guideline"]["groups"] for r in g["rules"])
 case=max(cases,key=size);layer=engine.config.num_hidden_layers//2
 report={"case_selection":"largest training text by character count","characters":size(case)}
 start=time.time()
 with torch.no_grad():result=engine.generate(deployment(case),mode="zero")
 report["zero_generation_seconds"]=time.time()-start;report["path_steps"]=len(result["path"])
 engine.clear_reference_memory();gc.collect();torch.cuda.empty_cache()
 spec=dict(kind="whole",layer=layer);module=make_module(spec,engine.config,engine.layer_device(layer))
 start=time.time();contexts=[engine.context(deployment(x)) for x in (case,swap_actions(case))]
 for node in nodes(case):
  reference=engine.reference(case,node["gid"]);scores=[];base=[];magnitudes=[]
  for context in contexts:
   query=engine.decision_query(context,node["history"],node["gid"])
   with torch.no_grad():clean,_=engine.scores(query)
   changed,diag=engine.scores(query,layer=layer,module=module,reference=reference)
   scores.append(changed);base.append(clean);magnitudes.append(diag["magnitude"].to(changed.device))
  target=torch.tensor(0 if node["target"].endswith("Y") else 1,device=scores[0].device)
  loss,_=objective(scores,base,target,magnitudes,use_js=False);(loss/len(nodes(case))/8).backward()
  del scores,base,magnitudes,loss,changed,diag
 report["training_case_seconds"]=time.time()-start
 report["training_nodes"]=len(nodes(case))
 report["finite_gradients"]=all(p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters())
 report["peak_allocated_gib"]=[torch.cuda.max_memory_allocated(i)/2**30 for i in (0,1)]
 report["peak_reserved_gib"]=[torch.cuda.max_memory_reserved(i)/2**30 for i in (0,1)]
 write_json(root/"runtime_benchmark.json",report)
 print(json.dumps(report),flush=True)

if __name__=="__main__":run("/home/gcx/gemma3_27b_head_pilot_20261103")
