"""CPU analysis of saved online queries, not a new online forward."""
import argparse,json
from pathlib import Path
import torch
from src.aster.manuscript_core import load_expert
from experiments.aster_manuscript_clinical_20261030.analyze import load
from .numeric import directions,score_change,SEEDS,cluster_ci

def main():
 p=argparse.ArgumentParser();p.add_argument("--input-run",default="/home/gcx/aster_manuscript_20261030/clinical01");p.add_argument("--output",required=True);a=p.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
 expert=load_expert(Path(a.input_run)/"analysis/expert.json");freeze=json.loads((Path(a.input_run)/"frozen.json").read_text())
 rows=[r for r in load(a.input_run) if r["row"]["split"]=="confirm" and not r["row"].get("control")]
 methods=["spectral","reverse","mean"]+[f"random:{s}" for s in SEEDS]
 records=[];raw={}
 for r in rows:
  row=r["row"]
  for method in methods:
   dirs=directions(expert,method)
   values=[]
   for head,vec in dirs.items():
    for state in ("positive","negative"):
     response=score_change(r["queries"][state][head],vec,r["cos"],r["sin"],freeze["g_plus"])
     raw[(row["pair_id"],method,head,state)]=response
     values.append(float(response.mean()))
     records.append({"pair_id":row["pair_id"],"source_id":row["source_id"],"method":method,"head":list(head),"state":state,
       "mean_unit_score_response":float(response.mean()),"positive_head_token_fraction":float((response>0).double().mean()),
       "query_heads":response.shape[0],"target_tokens":response.shape[1]})
 torch.save(raw,out/"per_head_target_response.pt")
 with (out/"records.jsonl").open("x") as f:
  for r in records:f.write(json.dumps(r,ensure_ascii=False)+"\n")
 summary={"scope":"CPU calculation from saved unedited first online queries, not new GPU measurement","pairs":len(rows),
   "source_clusters":len({r["row"]["source_id"] for r in rows}),"methods":{}}
 for method in methods:
  clusters={}
  for r in records:
   if r["method"]==method:clusters.setdefault(r["source_id"],[]).append(r["mean_unit_score_response"])
  summary["methods"][method]=cluster_ci([sum(v)/len(v) for v in clusters.values()])
 (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
if __name__=="__main__":main()
