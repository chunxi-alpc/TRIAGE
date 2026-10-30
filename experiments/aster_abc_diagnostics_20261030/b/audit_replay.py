"""Reproduce per-query-head sign, bidirectional mass, unknown and rounding audit."""
import argparse,json
from pathlib import Path
import torch

def main():
 p=argparse.ArgumentParser();p.add_argument("--run",required=True);p.add_argument("--output",required=True);a=p.parse_args()
 root=Path(a.run);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 rows=[json.loads(l) for l in (root/"records.jsonl").open()];idx={(r["pair_id"],r["state"],r["method"]):r for r in rows}
 agg={};unknown_equal=[];first_changes=[]
 for r in rows:
  if r["status"]!="complete":continue
  d=torch.load(root/r["detail"],map_location="cpu",weights_only=False)
  base=torch.load(root/idx[r["pair_id"],r["state"],"baseline"]["detail"],map_location="cpu",weights_only=False)
  if r["state"]=="unknown":unknown_equal.append(torch.equal(d["first_logits"],base["first_logits"]))
  if r["method"]!="spectral":continue
  first_changes.append({"pair_id":r["pair_id"],"state":r["state"],"max_abs_first_logit_delta":float((d["first_logits"]-base["first_logits"]).abs().max()),"top1_changed":int(d["first_logits"].argmax())!=int(base["first_logits"].argmax())})
  lookup={(t["layer"],t["kv_head"],t["query_offset"]):t for t in base["traces"]}
  for t in d["traces"]:
   if r["state"]=="unknown":continue
   ref=lookup[t["layer"],t["kv_head"],t["query_offset"]]
   key=r["state"]+":"+str(t["query_offset"]);x=agg.setdefault(key,{k:[] for k in ("eligible_mass_deltas","ineligible_mass_deltas","bidirectional","eligible_actual_score_positive","ineligible_actual_score_negative","theory_actual_sign_flip","theory_abs","residual_abs")})
   yes,no=None,None
   for rid,v in t["rules"].items():
    delta=v["mass"]-ref["rules"][rid]["mass"]
    if rid==r["expected_root"]:
     x["eligible_mass_deltas"].extend(delta.tolist());yes=delta
     x["eligible_actual_score_positive"].extend((v["score_delta_actual"]>0).flatten().tolist())
    else:
     x["ineligible_mass_deltas"].extend(delta.tolist());no=delta
     x["ineligible_actual_score_negative"].extend((v["score_delta_actual"]<0).flatten().tolist())
    x["theory_actual_sign_flip"].extend(((v["score_delta_actual"]*v["score_delta_theory"])<0).flatten().tolist())
    x["theory_abs"].extend(v["score_delta_theory"].abs().flatten().tolist())
    x["residual_abs"].extend((v["score_delta_actual"]-v["score_delta_theory"]).abs().flatten().tolist())
   x["bidirectional"].extend(((yes>0)&(no<0)).tolist())
 for _,x in agg.items():
  for k,v in list(x.items()):x[k]={"mean":sum(v)/len(v),"n_repeated_observations":len(v),"max":max(v)}
 result={"positions":agg,"unknown_first_logits_exactly_equal":all(unknown_equal),"unknown_comparisons":len(unknown_equal),
  "spectral_first_logits":first_changes,"scope":"Head/token counts are repeated observations; source cluster is independent unit",
  "actual_score_definition":"float32 q dot (bf16-edited key minus bf16-original key), excluding separate matmul-kernel rounding"}
 (out/"bidirectional.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
if __name__=="__main__":main()
