"""Short gold/predicted-root teacher-forced replay, isolated from official test."""
import argparse, json, time, traceback
from pathlib import Path
import torch
from src.aster.manuscript_core import load_expert, edit_kv_layers, plan_gains
from src.aster.manuscript_runtime import ManuscriptBackbone, PromptBuilder
from experiments.aster_manuscript_clinical_20261030.analyze import load
from experiments.aster_manuscript_clinical_20261030.classify import model_and_classifier
from experiments.aster_manuscript_clinical_20261030.engine import decisions_from
from .numeric import directions, SEEDS, summarize
from .probe import probe

def write(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input-run", default="/home/gcx/aster_manuscript_20261030/clinical01")
    p.add_argument("--output",required=True)
    p.add_argument("--max-sources",type=int,default=8)
    p.add_argument("--steps",type=int,default=17)
    p.add_argument("--route",choices=["gold_root","predicted_root"],default="gold_root")
    p.add_argument("--split",choices=["confirm","development"],default="confirm")
    a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);(out/"detail").mkdir()
    torch.set_num_threads(4);torch.manual_seed(SEEDS[0]);start=time.time()
    src=Path(a.input_run);freeze=json.loads((src/"frozen.json").read_text())
    expert=load_expert(src/"analysis/expert.json")
    chosen={}
    for r in sorted(load(src),key=lambda r:r["row"]["pair_id"]):
        row=r["row"]
        if row["split"]==a.split and not row.get("control"):
            chosen.setdefault(row["source_id"],r)
    rows=list(chosen.values())[:a.max_sources]
    if not rows:raise ValueError("No eligible non-control source clusters")
    methods=["baseline","spectral","reverse","mean","score_bias"]+[f"random:{s}" for s in SEEDS]
    write(out/"preflight.json",{"input_run":str(src),"input_split":a.split,"source_clusters":[r["row"]["source_id"] for r in rows],
        "pairs":[r["row"]["pair_id"] for r in rows],"methods":methods,"steps":a.steps,"route":a.route,
        "gain_plus":freeze["g_plus"],"gain_minus":freeze.get("g_minus",freeze["g_plus"]),
        "scope":"age-condition rule-verified root; physician_adjudicated_pairs=0","official_test_accessed":False,
        "expected_records":len(rows)*3*len(methods)})
    tok,model,_=model_and_classifier()
    backend=ManuscriptBackbone(model,model_id="Qwen/Qwen3-8B",model_revision=freeze["model_revision"])
    builder=PromptBuilder(tok,chat_template_kwargs={"enable_thinking":False})
    heads=list(directions(expert,"spectral"))
    records=[]
    with (out/"records.jsonl").open("x") as handle:
      for r in rows:
       row=r["row"]
       for state,patient in row["states"].items():
        prompt=builder.prepare(row["guideline"],patient,row["question"],online=True)
        prefix=backend.prefill(prompt);base_layers=prefix.cache.to_legacy_cache()
        if a.route=="gold_root":
            expected=row["target_rule_id"] if state=="positive" else row["verification"]["negative_rule_id"] if state=="negative" else "NONE"
            decisions={g["group_id"]:None for g in row["guideline"]["groups"]}
            decisions["G0"]={"prediction":expected,"abstain":False,"diagnostic_only":True}
        else:
            raw={g["group_id"]:None for g in row["guideline"]["groups"]}
            raw["G0"]=r["classification"].get(state) if "logits" in r["classification"].get(state,{}) else None
            decisions=decisions_from(raw,freeze)
            expected=row["target_rule_id"] if state=="positive" else row["verification"]["negative_rule_id"] if state=="negative" else "NONE"
        plan=plan_gains(row["guideline"],decisions,g_plus=freeze["g_plus"],g_minus=freeze.get("g_minus",freeze["g_plus"]))
        positions={rid:prompt.positions[rid] for rid in ("R0Y","R0N")}
        gains={rid:plan["gains"][rid] for rid in positions}
        baseline_tokens=None;baseline_traces=None
        for method in methods:
         record={"pair_id":row["pair_id"],"source_id":row["source_id"],"state":state,"method":method,"route":a.route,"expected_root":expected}
         try:
          vecs={} if method in ("baseline","score_bias") else directions(expert,method)
          bias=freeze["bias_unit_rms"] if method=="score_bias" else {}
          with backend.session():
           edited,audit=edit_kv_layers(base_layers,vecs,positions,{rid:0. if method=="baseline" else g for rid,g in gains.items()},prefix.cos,prefix.sin)
           cache=type(prefix.cache).from_legacy_cache(tuple(edited))
           tokens=[];first_logits=None
           with probe(backend,heads,positions,gains,bias,base_layers,edited,vecs,prefix.cos,prefix.sin,
                      prompt.prefix_length,prompt.input_ids.shape[1]-1) as traces:
            result=backend._forward(prompt.input_ids[:,prompt.prefix_length:],cache=cache,start=prompt.prefix_length,use_cache=True)
            for step in range(a.steps):
             if step==0:first_logits=result.logits[0,-1].float().cpu()
             token=int(result.logits[0,-1].argmax()) if method=="baseline" else baseline_tokens[step]
             tokens.append(token)
             if step+1<a.steps:
              result=backend._forward(torch.tensor([[token]],dtype=torch.long),cache=result.past_key_values,
                                      start=prompt.input_ids.shape[1]+step,use_cache=True)
          if method=="baseline":
           baseline_tokens=tokens;baseline_traces=traces
          reference={(t["layer"],t["kv_head"],t["query_offset"]):t for t in baseline_traces}
          effects=[];local=[];errors=[]
          for t in traces:
           b=reference[(t["layer"],t["kv_head"],t["query_offset"])]
           for rid,d in t["rules"].items():
            signed=1 if rid==expected else -1
            effects.append(signed*float((d["mass"]-b["rules"][rid]["mass"]).mean()))
            local.append(signed*float((d["mass"]-d["mass_same_query_undo"]).mean()))
            errors.append(float((d["score_delta_actual"]-d["score_delta_theory"]).abs().max()))
          detail_name=f"{len(records):04d}.pt"
          torch.save({"traces":traces,"teacher_forced_token_ids":tokens,"first_logits":first_logits,
                      "gains":gains,"edit":audit,"prompt_target_positions":positions},out/"detail"/detail_name)
          record.update(status="complete",signed_mass_change=sum(effects)/len(effects),signed_local_mass_change=sum(local)/len(local),
              max_score_formula_quantization_residual=max(errors),detail="detail/"+detail_name,query_records=len(traces),edit=audit,
              first_token_depends_on_edited_suffix_forward=True)
         except Exception as e:
          record.update(status="failed",error_type=type(e).__name__,error=str(e));traceback.print_exc()
         edited=None;cache=None;result=None
         records.append(record);handle.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+"\n");handle.flush()
         write(out/"status.json",{"status":"running","records":len(records),"technical_failures":sum(x["status"]!="complete" for x in records),"elapsed_seconds":time.time()-start})
         if method=="baseline" and baseline_tokens is None:
          for remaining in methods[1:]:
           failure={**record,"method":remaining,"status":"failed","error_type":"BaselineReplayUnavailable","error":"baseline failed; shared trajectory unavailable"}
           records.append(failure);handle.write(json.dumps(failure,ensure_ascii=False)+"\n")
          handle.flush()
          break
        del prefix,base_layers
        torch.cuda.empty_cache()
        if len(records)>=len(methods) and sum(x["status"]!="complete" for x in records)/len(records)>.2:
         write(out/"summary.json",summarize(records))
         write(out/"status.json",{"status":"stopped","reason":"technical_failure_fraction_above_20_percent","records":len(records)})
         return
    summary=summarize(records);summary.update(elapsed_seconds=time.time()-start,peak_memory_gb=torch.cuda.max_memory_allocated()/1e9,
        source_clusters=len(rows),scope="confirm age root only; shared baseline teacher-forced token trajectories")
    write(out/"summary.json",summary)
    write(out/"status.json",{"status":"complete","records":len(records),"technical_failures":summary["technical_failures"],"elapsed_seconds":time.time()-start,"peak_memory_gb":summary["peak_memory_gb"]})
if __name__=="__main__":main()
