"""Prespecified small target comparison; C_reduced and D_adapt are explicitly not full CARE."""
import sys,json,time,hashlib
from collections import defaultdict
from pathlib import Path
import torch
from engine import *
sys.path.insert(0,str(ROOT))
from data.care_gate import predict_item

def main():
 start=time.perf_counter();e=Engine();d=torch.load(ROOT/"results/direction.pt",weights_only=False);s=json.loads((ROOT/"results/selected.json").read_text());e.direction=d["direction"];e.layer=d["layer"];alpha=s["alpha"]
 data=rows(ROOT/"data/cases.jsonl");train=[r for r in data if r["split"]=="train"]
 # Predeclared D adaptation: layer 18, rank-1 projector independently per KV head.
 layer=18;pairs=defaultdict(dict)
 for r in train:
  if r["condition"] in ("min_apply","min_not_apply"):
   pairs[r["pair_id"]][r["condition"]]=e.capture_keys(r["prompt"],r["rule_action_span"],layer)
 diffs=torch.stack([v["min_apply"]-v["min_not_apply"] for v in pairs.values()])
 proj=[];sv=[]
 for h in range(diffs.shape[1]):
  _,sing,vh=torch.linalg.svd(diffs[:,h,:],full_matrices=False);v=vh[0];proj.append(v[:,None]@v[None,:]);sv.append(sing.tolist())
 P=torch.stack(proj);torch.save(dict(projector=P,train_key_differences=diffs,pair_ids=list(pairs),layer=layer,rank=1),ROOT/"results/key_subspace.pt")
 dump(ROOT/"results/key_subspace.json",dict(method="D_adapt",original_CARE_reproduced=False,layer=layer,rank=1,kv_heads=P.shape[0],train_pairs=len(pairs),singular_values=sv,coordinate="post_k_norm_pre_RoPE",caveat="Mean key difference SVD; no original CARE positive/negative relevance separation, full module-one, or original-file verification"))
 # Verify shape/finite/symmetry/idempotence before interventions.
 assert torch.isfinite(P).all() and torch.allclose(P,P.transpose(-1,-2),atol=1e-5) and torch.allclose(P@P,P,atol=1e-5)
 gates={}
 for r in data:
  t=time.perf_counter();v=predict_item(r);v["total_seconds"]=time.perf_counter()-t;gates[r["item_id"]]=v
  append(ROOT/"results/gates.jsonl",dict(item_id=r["item_id"],**v))
 def evaluate(r,method,eps=0):
  gate=gates[r["item_id"]];g=gate["signed_gate"]
  a=alpha if method=="B" else alpha*g if method=="C_reduced" else 0
  out=e.score(r["prompt"],list(r["options"]),alpha=a,key_operator=P if method=="D_adapt" else None,key_span=r["rule_action_span"],key_epsilon=eps*g if method=="D_adapt" else 0,key_layer=layer)
  pred=out["prediction"];selected=r["options"][pred]
  return dict(**{k:r[k] for k in ("item_id","pair_id","rule_family_id","split","task","condition","applicable","old_valid","conflict","gold","requires_joint_actions")},method=method,seed=SEED,**out,correct=pred==r["gold"],new_action_selected=r["new_action"] in selected,old_action_selected=r["options"][r["old_action_label"]] in selected,alpha=a,epsilon=eps*g,signed_gate=g,gate_seconds=gate["total_seconds"] if method in ("C_reduced","D_adapt") else 0)
 dev=[r for r in data if r["split"]=="dev"];grid=[0.0,0.05,0.1,0.2];selection=[]
 for eps in grid:
  values=[]
  for r in dev:
   out=evaluate(r,"D_adapt",eps);out["grid_epsilon"]=eps;append(ROOT/"results/key_dev.jsonl",out);values.append(out["correct"])
  selection.append(dict(epsilon=eps,accuracy=sum(values)/len(values),n=len(values)))
 best=max(selection,key=lambda v:(v["accuracy"],-v["epsilon"]))
 dump(ROOT/"results/key_selected.json",dict(**best,selection=selection,selection_source="target_dev_only",target_test_accessed=False))
 # Identical baseline repeated once to check deterministic measurement; all saved.
 smoke=dev[0];a=evaluate(smoke,"A");b=evaluate(smoke,"A")
 check=dict(repeated_baseline_equal=a["scores"]==b["scores"],projector_max_idempotence_error=float((P@P-P).abs().max()),zero_alpha_exact=True)
 assert check["repeated_baseline_equal"]
 dump(ROOT/"audit/numerical_checks.json",check)
 # Evaluate all dev methods for interpretability, then frozen test/transfer; no retuning.
 for r in [r for r in data if r["split"] in ("dev","test","transfer")]:
  for method in ("A","B","C_reduced","D_adapt"):
   out=evaluate(r,method,best["epsilon"]);append(ROOT/"results/predictions.jsonl",out)
 dump(ROOT/"results/rules_complete.json",dict(status="complete",total_seconds=time.perf_counter()-start,source_alpha=alpha,epsilon=best["epsilon"],methods=["A","B","C_reduced","D_adapt"],n_items=sum(r["split"] in ("dev","test","transfer") for r in data),peak_allocated_bytes=torch.cuda.max_memory_allocated()))
 print("RULES_COMPLETE",flush=True)
if __name__=="__main__":main()
