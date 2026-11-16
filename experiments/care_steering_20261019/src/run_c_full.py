#!/usr/bin/env python3
"""Outcome-independent 20-case V6 supplement. No modification of primary predictions."""
from __future__ import annotations
import argparse,copy,gc,hashlib,importlib.metadata,json,os,random,sys,time,traceback
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sources.care_v6_frozen import TrialMatchAIAdapter,BioLinkBERTMedNLI,CAREModule1V6,_rule_list
OUT=ROOT/"results"/"c_full"
MODELS={k:str(Path("/home/gcx/models/care_module1_v5")/v) for k,v in [
("base_model","phi-4"),("adapter_model","trialmatchai-phi4-reasoning-lora"),("evidence_model","biolinkbert-mednli")]}
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def fhash(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+".tmp");tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+"\n");tmp.replace(p)
def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def key(patient,condition,kind):
 return digest({"patient":patient,"criterion_text":condition["text"],"criterion_kind":"exclusion" if kind=="exclusion" else "inclusion"})
def selected_rules(row):
 return _rule_list(copy.deepcopy([r for r in row["structured_rules"] if r["role"] in {"new","independent"}]))
def prepare(cases):
 req={}
 for row in cases:
  for rule in selected_rules(row):
   for group,kind in [("conditions","inclusion"),("exceptions","exclusion")]:
    for c in rule[group]:
     kind_actual=c.get("criterion_kind",kind);k=key(row["patient"],c,kind_actual)
     req.setdefault(k,{"request_key":k,"patient":row["patient"],"condition":{"text":c["text"]},"criterion_kind":kind_actual,"uses":[]})
     req[k]["uses"].append({"item_id":row["item_id"],"rule_id":rule["rule_id"],"condition_id":c["condition_id"]})
 return list(req.values())
class Replay:
 def __init__(self,cache):self.cache=cache
 def judge(self,patient,condition,criterion_kind="inclusion"):
  k=key(patient,condition,criterion_kind)
  if k not in self.cache:raise KeyError("Unprepared model request "+k)
  return copy.deepcopy(self.cache[k])
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--prepare-only",action="store_true");args=ap.parse_args()
 cases=read(ROOT/"data/c_full_subset.jsonl");assert len(cases)==20 and len({c["rule_family_id"] for c in cases})==2
 requests=prepare(cases);assert len(requests)==48
 contract={"cases_sha256":fhash(ROOT/"data/c_full_subset.jsonl"),"request_sha256":digest(requests),"model_paths":MODELS,
 "source_sha256":{str(p.relative_to(ROOT)):fhash(p) for p in [Path(__file__),ROOT/"sources/care_v6_frozen.py",ROOT/"sources/trialmatch_parser.py",ROOT/"src/engine.py"]},
 "generation":{"max_new_tokens":2048,"batch_size":8,"do_sample":False},"seed":261019,
 "input_whitelist":["patient","structured_rules","question"],"old_rule_validity":"explicit public structured field",
 "condition_kind":"Unchanged V6 inclusion/exclusion mapping","selection":"Lexicographic first family in test and transfer; all 10 cases; no output-based selection"}
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/"contract.json").exists():
  assert json.loads((OUT/"contract.json").read_text())==contract,"Run contract changed"
 else:save(OUT/"contract.json",contract)
 save(OUT/"requests.json",requests)
 if args.prepare_only:
  save(OUT/"prepared.json",{"cases":20,"unique_requests":48,"cuda_started":False,"contract_sha256":digest(contract)})
  print("Prepared 20 cases and 48 requests; no GPU inference.")
  return
 if (OUT/"complete.json").exists():raise RuntimeError("Completed supplement is immutable")
 import torch,numpy as np
 random.seed(261019);np.random.seed(261019);torch.manual_seed(261019);torch.cuda.manual_seed_all(261019);torch.set_num_threads(4)
 torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
 versions={n:importlib.metadata.version(n) for n in ["torch","transformers","peft","bitsandbytes","accelerate","tokenizers"]}
 started=time.perf_counter()
 status={"started_utc":datetime.now(timezone.utc).isoformat(),"status":"loading_trialmatch","cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
 "versions":versions,"python":sys.executable,"deps_path":str(ROOT/".deps"),"contract_sha256":digest(contract)}
 save(OUT/"status.json",status)
 cache={};batch_files=[]
 for i in range(0,len(requests),8):
  p=OUT/"batches"/f"batch_{i:03d}.json"
  if p.exists():
   obj=json.loads(p.read_text());assert obj["request_keys"]==[r["request_key"] for r in requests[i:i+8]] and obj["contract_sha256"]==digest(contract)
   for k,o in zip(obj["request_keys"],obj["outputs"]):cache[k]=o
  batch_files.append((i,p))
 phi_load=0.0
 if len(cache)<len(requests):
  t=time.perf_counter()
  adapter=TrialMatchAIAdapter.from_pretrained(device="cuda:0",base_model=MODELS["base_model"],adapter_model=MODELS["adapter_model"])
  phi_load=time.perf_counter()-t
  status.update(status="generating_trialmatch",phi_load_seconds=phi_load,active_adapters=list(adapter.model.active_adapters));save(OUT/"status.json",status)
  for i,p in batch_files:
   if p.exists():continue
   batch=requests[i:i+8];outputs=adapter.judge_many(batch,batch_size=8);assert len(outputs)==len(batch)
   save(p,{"request_keys":[r["request_key"] for r in batch],"outputs":outputs,"contract_sha256":digest(contract)})
   for r,o in zip(batch,outputs):cache[r["request_key"]]=o
   status.update(completed_requests=len(cache),elapsed_seconds=time.perf_counter()-started);save(OUT/"status.json",status)
  del adapter;gc.collect();torch.cuda.empty_cache()
 with (OUT/"trialmatch_generations.jsonl").open("w") as f:
  for r in requests:f.write(json.dumps({"request_key":r["request_key"],"output":cache[r["request_key"]]},ensure_ascii=False)+"\n")
 status.update(status="evidence_and_rule_inference",completed_requests=len(cache));save(OUT/"status.json",status)
 t=time.perf_counter()
 mednli=BioLinkBERTMedNLI.from_pretrained(device="cuda:0",model_name=MODELS["evidence_model"])
 evidence_load=time.perf_counter()-t
 model1=CAREModule1V6(Replay(cache),mednli,nli_batch_size=32,candidate_limit=8)
 allocation={r["request_key"]:float(cache[r["request_key"]].get("inference_seconds",0))/len(r["uses"]) for r in requests}
 gates={};structures=[]
 for row in cases:
  rules=selected_rules(row);t=time.perf_counter();output=model1.run(rules,row["patient"],row["question"]);wall=time.perf_counter()-t
  new=next(r for r in output["rules"] if r["rule_id"]=="NEW")
  old=next(r for r in row["structured_rules"] if r["role"]=="old")
  public_new=next(r for r in row["structured_rules"] if r["role"]=="new")
  conflict=old["action"]!=public_new["action"];state=new["applicability"];evidence=new["evidence_status"]
  g=(1.0 if state=="T" else -1.0 if state=="F" and old["valid"] else 0.0)
  if evidence!="supported" or not conflict:g=0.0
  keys=[]
  for r in rules:
   for group,kind in [("conditions","inclusion"),("exceptions","exclusion")]:
    for c in r[group]:keys.append(key(row["patient"],c,c.get("criterion_kind",kind)))
  shared=sum(allocation[k] for k in keys)
  gate={"new_state":state,"new_evidence_status":evidence,"signed_gate":g,"v6_action_gate":output["action_gate"]["NEW"],
        "old_valid":old["valid"],"conflict":conflict,"full_existing_v6_executed":True,"original_CARE_verified":False,
        "rule_inference_wall_seconds":wall,"trialmatch_batch_amortized_seconds":shared,"total_seconds":shared+wall,
        "timing_note":"Batch generation time allocated across requests, then across repeated request uses. Not isolated per-case latency."}
  gates[row["item_id"]]=gate
  structures.append({"item_id":row["item_id"],"output":output,"gate":gate})
 with (OUT/"structures.jsonl").open("w") as f:
  for x in structures:f.write(json.dumps(x,ensure_ascii=False)+"\n")
 save(OUT/"gates.json",gates)
 m1_peak={"allocated_bytes":torch.cuda.max_memory_allocated(),"reserved_bytes":torch.cuda.max_memory_reserved()}
 del model1,mednli;gc.collect();torch.cuda.empty_cache()
 status.update(status="scoring_qwen",module1_peak=m1_peak);save(OUT/"status.json",status)
 import engine
 # Engine uses a shared default manifest path; route only this metadata write to this run.
 original_dump=engine.dump
 def own_dump(path,obj):
  if Path(path)==ROOT/"results/engine.json":save(OUT/"engine.json",obj)
  else:original_dump(path,obj)
 engine.dump=own_dump
 e=engine.Engine()
 direction=torch.load(ROOT/"results/direction.pt",weights_only=False)
 selected=json.loads((ROOT/"results/selected.json").read_text())
 e.direction=direction["direction"];e.layer=direction["layer"];alpha=selected["alpha"]
 predictions=[]
 for r in cases:
  gate=gates[r["item_id"]];a=alpha*gate["signed_gate"]
  out=e.score(r["prompt"],list(r["options"]),alpha=a)
  chosen=r["options"][out["prediction"]]
  predictions.append({**{k:r[k] for k in ("item_id","pair_id","rule_family_id","split","task","condition","applicable","old_valid","conflict","gold","requires_joint_actions")},
   "method":"C_V6","seed":261019,**out,"correct":out["prediction"]==r["gold"],
   "new_action_selected":r["new_action"] in chosen,"old_action_selected":r["options"][r["old_action_label"]] in chosen,
   "alpha":a,"epsilon":0.0,"signed_gate":gate["signed_gate"],"gate_seconds":gate["total_seconds"],
   "module1_rule_inference_wall_seconds":gate["rule_inference_wall_seconds"],
   "module1_trialmatch_batch_amortized_seconds":gate["trialmatch_batch_amortized_seconds"],
   "module1_state":gate["new_state"],"module1_evidence_status":gate["new_evidence_status"],
   "end_to_end_amortized_seconds_excluding_load":out["total_seconds"]+gate["total_seconds"]})
 tmp=ROOT/"results/c_full_predictions.jsonl.tmp"
 tmp.write_text("".join(json.dumps(p,ensure_ascii=False,allow_nan=False)+"\n" for p in predictions))
 tmp.replace(ROOT/"results/c_full_predictions.jsonl")
 summary={"status":"complete","items":len(cases),"unique_requests":len(requests),"logical_atom_calls":sum(len(r["uses"]) for r in requests),
 "actual_elapsed_seconds":time.perf_counter()-started,"phi_load_seconds_this_execution":phi_load,"evidence_load_seconds":evidence_load,
 "qwen_load_seconds":e.load_seconds,"trialmatch_total_batch_wall_seconds":sum(float(x.get("inference_seconds",0)) for x in cache.values()),
 "module1_rule_inference_wall_seconds":sum(g["rule_inference_wall_seconds"] for g in gates.values()),
 "gate_counts":dict(Counter(str(g["signed_gate"]) for g in gates.values())),"state_counts":dict(Counter(g["new_state"] for g in gates.values())),
 "evidence_counts":dict(Counter(g["new_evidence_status"] for g in gates.values())),"module1_peak":m1_peak,
 "overall_peak_allocated_bytes":torch.cuda.max_memory_allocated(),"overall_peak_reserved_bytes":torch.cuda.max_memory_reserved(),
 "alpha":alpha,"layer":e.layer,"correct":sum(p["correct"] for p in predictions),"selection":contract["selection"],
 "direction_sha256":fhash(ROOT/"results/direction.pt"),"selected_sha256":fhash(ROOT/"results/selected.json"),
 "predictions_sha256":fhash(ROOT/"results/c_full_predictions.jsonl"),"primary_predictions_modified":False,
 "gate_policy":"T:+1; F and old_valid:-1; U:0; missing/disputed evidence or no conflict:0. V6 own action_gate separately retained."}
 save(OUT/"complete.json",summary);status.update(summary);save(OUT/"status.json",status)
 print("C_V6_COMPLETE",flush=True)
if __name__=="__main__":
 try:main()
 except Exception as exc:
  OUT.mkdir(parents=True,exist_ok=True)
  save(OUT/"failure.json",{"error_type":type(exc).__name__,"error":str(exc),"traceback":traceback.format_exc()})
  raise
