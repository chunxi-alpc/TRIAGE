"""Postprocess independent replay with paired source-cluster denominators."""
import argparse,json
from pathlib import Path
import torch
from .numeric import cluster_ci

def main():
 p=argparse.ArgumentParser();p.add_argument("--run",required=True);p.add_argument("--output",required=True);a=p.parse_args()
 run=Path(a.run);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 pre=json.loads((run/"preflight.json").read_text())
 rows=[json.loads(x) for x in (run/"records.jsonl").open() if x.strip()]
 indexed={(r["pair_id"],r["state"],r["method"]):r for r in rows}
 result={"planned_records":pre["expected_records"],"attempted_records":len(rows),
   "unattempted_records":max(0,pre["expected_records"]-len(rows)),"technical_failures":sum(r["status"]!="complete" for r in rows),
   "coverage_by_method":{},"paired_differences":{},"position_effects":{},"random_seeds":5,"minimum_empirical_p":1/6,
   "scope":"Exploratory age-rule root diagnostic; 2 sources do not establish clinical generalization."}
 for method in pre["methods"]:
  rs=[r for r in rows if r["method"]==method]
  result["coverage_by_method"][method]={"planned":pre["expected_records"]//len(pre["methods"]),"complete":sum(r["status"]=="complete" for r in rs),"failed":sum(r["status"]!="complete" for r in rs)}
 for comparator in pre["methods"]:
  if comparator=="spectral":continue
  clusters={};pairs=0
  for r in rows:
   if r["method"]!="spectral" or r["state"]=="unknown" or r["status"]!="complete":continue
   other=indexed.get((r["pair_id"],r["state"],comparator))
   if other is None or other["status"]!="complete":continue
   clusters.setdefault(r["source_id"],[]).append(r["signed_mass_change"]-other["signed_mass_change"]);pairs+=1
  vals=[sum(v)/len(v) for v in clusters.values()]
  result["paired_differences"]["spectral-minus-"+comparator]={**cluster_ci(vals),"paired_case_states":pairs}
 for r in rows:
  if r["status"]!="complete":continue
  d=torch.load(run/r["detail"],map_location="cpu",weights_only=False)
  baseline=indexed.get((r["pair_id"],r["state"],"baseline"))
  if baseline is None or baseline["status"]!="complete":continue
  b=torch.load(run/baseline["detail"],map_location="cpu",weights_only=False)
  lookup={(x["layer"],x["kv_head"],x["query_offset"]):x for x in b["traces"]}
  for t in d["traces"]:
   b=lookup[t["layer"],t["kv_head"],t["query_offset"]]
   key=r["method"]+":"+r["state"]+":"+str(t["query_offset"])
   store=result["position_effects"].setdefault(key,{"by_source":{},"max_formula_quantization_residual":0.})
   vals=[]
   for rid,measure in t["rules"].items():
    sign=1 if rid==r["expected_root"] else -1
    vals.append(sign*float((measure["mass"]-b["rules"][rid]["mass"]).mean()))
    store["max_formula_quantization_residual"]=max(store["max_formula_quantization_residual"],float((measure["score_delta_actual"]-measure["score_delta_theory"]).abs().max()))
   store["by_source"].setdefault(r["source_id"],[]).append(sum(vals)/len(vals))
 for key,s in result["position_effects"].items():
  vals=[sum(v)/len(v) for v in s.pop("by_source").values()]
  s.update(cluster_ci(vals))
 unknown=[r for r in rows if r["state"]=="unknown" and r["status"]=="complete"]
 result["unknown_zero_key_edit"]=all(r["edit"]["changed_key_vectors"]==0 for r in unknown)
 result["unknown_complete_records"]=len(unknown)
 result["max_formula_quantization_residual"]=max((r.get("max_score_formula_quantization_residual",0.) for r in rows),default=0.)
 (out/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
if __name__=="__main__":main()
