"""CPU paired comparison of two completed independent root-replay runs."""
import argparse,json
from pathlib import Path
import torch
from .numeric import cluster_ci

def main():
 p=argparse.ArgumentParser();p.add_argument("--gold-run",required=True);p.add_argument("--predicted-run",required=True);p.add_argument("--output",required=True);a=p.parse_args()
 goldroot,predroot=Path(a.gold_run),Path(a.predicted_run)
 gold=[json.loads(x) for x in (goldroot/"records.jsonl").open()]
 pred=[json.loads(x) for x in (predroot/"records.jsonl").open()]
 idx={(r["pair_id"],r["state"],r["method"]):r for r in gold}
 cases=[];clusters={}
 for r in pred:
  if r["method"]!="spectral" or r["status"]!="complete":continue
  g=idx.get((r["pair_id"],r["state"],r["method"]))
  if g is None or g["status"]!="complete":continue
  detail=torch.load(predroot/r["detail"],map_location="cpu",weights_only=False)
  route=next((rid for rid,gain in detail["gains"].items() if gain>0),"NONE")
  cases.append({"pair_id":r["pair_id"],"source_id":r["source_id"],"state":r["state"],"gold":r["expected_root"],"predicted":route,"routing_correct":route==r["expected_root"],"gains":detail["gains"],"gold_effect":g["signed_mass_change"],"predicted_effect":r["signed_mass_change"]})
  if r["state"]!="unknown":clusters.setdefault(r["source_id"],[]).append(r["signed_mass_change"]-g["signed_mass_change"])
 gmean=json.loads((goldroot/"summary.json").read_text())["methods"]["spectral"]["mean"]
 pmean=json.loads((predroot/"summary.json").read_text())["methods"]["spectral"]["mean"]
 out={"cases":cases,"spectral_predicted_minus_gold":cluster_ci([sum(v)/len(v) for v in clusters.values()]),"spectral_relative_attenuation":(gmean-pmean)/gmean,
   "scope":"Exploratory; unknown effects excluded from primary signed mass comparison"}
 target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True)
 with target.open("x") as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write("\n")
if __name__=="__main__":main()
