#!/usr/bin/env python3
"""24-case offline diagnostic. Labels isolated from normal inference and gates."""
import argparse, copy, hashlib, json, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from src.module1 import Module1
from src.module1.backend import LocalBackend
from src.pipeline.tokens import build_prompt_and_mapping
from src.module2.localizer import scale_query_heads
from src.module3 import GatedAttentionIntervention, select_intervention_heads

SYSTEM=("Use explicit patient findings to answer the management question. When a guideline is supplied, "
        "follow only rules whose required conditions are satisfied. Never invent evidence. "
        "Select exactly one supplied option. If the evidence is insufficient, select the insufficient-evidence option. "
        "Return only its single capital letter, without explanation or punctuation.")
OUT=ROOT/"results/algorithm_diagnostic"

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):
    temp=path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    temp.replace(path)

def options_for(item):
    rows=[]
    for rule in item["guideline"]["rules"]:
        action=rule["action"]
        row=next((r for r in rows if r["action"]==action),None)
        if row is None:
            rows.append({"label":chr(65+len(rows)),"action":action,"rule_ids":[rule["rule_id"]]})
        else: row["rule_ids"].append(rule["rule_id"])
    rows.append({"label":chr(65+len(rows)),"action":"Insufficient explicit patient evidence to select a guideline recommendation.","rule_ids":[]})
    if len(rows)>26: raise ValueError("Too many decision options")
    return rows

def option_text(options):
    return "\n\nDecision options (all guideline actions, not reference-selected):\n"+"\n".join(r["label"]+". "+r["action"] for r in options)+"\nAnswer with one letter."

class DecisionTokenizer:
    def __init__(self,tokenizer): self.inner=tokenizer
    def __getattr__(self,key): return getattr(self.inner,key)
    def __call__(self,*a,**kw): return self.inner(*a,**kw)
    def apply_chat_template(self,messages,**kwargs):
        messages=copy.deepcopy(messages)
        clinical=messages[0]["content"].replace(
            "Return only a concise JSON object with decision, actions (list), status "
            "(decided or insufficient_evidence), and rationale. ", "")
        messages[0]["content"]=clinical+" "+SYSTEM
        return self.inner.apply_chat_template(messages,**kwargs)

def raw_prompt(backend,item,options,rules=None):
    content="Question:\n"+item["question"]+"\n\nPatient note:\n"+item["patient"]
    if rules is not None: content+="\n\nCurrent guideline rules:\n"+"\n\n".join(r["text"] for r in rules)
    content+=option_text(options)
    text=backend.tokenizer.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":content}],
         tokenize=False,add_generation_prompt=True,enable_thinking=False)
    encoded=backend.tokenizer(text,return_tensors="pt",add_special_tokens=False).to(backend.model.device)
    if encoded["input_ids"].shape[1]>backend.max_input_tokens: raise ValueError("Prompt budget exceeded")
    return text,encoded

def logits(backend,encoded):
    with torch.inference_mode():
        return backend.model(**encoded,use_cache=False,logits_to_keep=1).logits[0,-1].float().cpu()

def summarize_logits(value,options,label_ids,prompt):
    chosen=value[label_ids]; lp=value.log_softmax(-1)[label_ids]; cp=chosen.softmax(0)
    index=int(chosen.argmax())
    return {"decision_index":index,"decision_label":options[index]["label"],"decision_action":options[index]["action"],
            "decision_rule_ids":options[index]["rule_ids"],
            "option_logits":chosen.tolist(),"option_log_probabilities":lp.tolist(),
            "option_normalized_probabilities":cp.tolist(),"prompt_tokens":prompt["input_ids"].shape[1]}

def zero_graph(graph):
    zero=copy.deepcopy(graph)
    zero["action_gate"]={rid:0. for rid in zero["action_gate"]}
    zero["binding_gate"]={rid:{cid:0. for cid in rows} for rid,rows in zero["binding_gate"].items()}
    for row in zero["relation_weights"]:
        for k in ("weight","signed_weight","binding_gate","action_gate","prior_competition_weight"): row[k]=0.
    return zero

def evaluate(result,reference):
    options=result["options"]; gold=next(i for i,r in enumerate(options) if reference["gold_rule_id"] in r["rule_ids"])
    prior=result["variants"]["original"]["decision_index"]
    rawguide=result["variants"]["original_full_guideline"]["decision_index"]
    competitor=max((i for i in range(len(options)) if i!=gold),key=lambda i:result["variants"]["original"]["option_logits"][i])
    base=result["variants"]["original"]
    for scope,row in result["variants"].items():
        row["correct"]=row["decision_index"]==gold
        row["correct_log_probability"]=row["option_log_probabilities"][gold]
        row["correct_probability"]=float(torch.tensor(row["correct_log_probability"],dtype=torch.float64).exp())
        row["correct_option_probability"]=row["option_normalized_probabilities"][gold]
        row["correct_minus_prior_log_probability"]=row["option_logits"][gold]-row["option_logits"][prior]
        row["correct_minus_strongest_incorrect_prior"]=row["option_logits"][gold]-row["option_logits"][competitor]
        row["correct_minus_raw_guideline_decision"]=row["option_logits"][gold]-row["option_logits"][rawguide]
        row["correct_probability_change_vs_original"]=row["correct_probability"]-float(torch.tensor(base["option_log_probabilities"][gold],dtype=torch.float64).exp())
    gold_id=reference["gold_rule_id"]; graph=result["module1"]
    result["evaluation"]={"gold_rule_id":gold_id,"gold_action":reference["gold_action"],"gold_option_index":gold,
       "prior_option_index":prior,"prior_equals_gold":prior==gold,"comparison_incorrect_option_index":competitor,
       "key_rule_recalled_before":gold_id in result["old_top3_rule_ids"],
       "key_rule_recalled_after":gold_id in [r["rule_id"] for r in graph["candidate_rules"]],
       "gold_rule_applicability":next((r["status"] for r in graph["applicability"] if r["rule_id"]==gold_id),"not_retrieved"),
       "gold_action_gate":graph["action_gate"].get(gold_id,0.),
       "applicability_correct":sum((r["status"]=="applicable")==(r["rule_id"]==gold_id) and r["status"]!="unknown" for r in graph["applicability"]),
       "applicability_total":len(graph["applicability"])}
    known={r["index"]:r["decision_outcome"] for r in reference["gold_path"] if "decision_outcome" in r}
    seen=set(); rows=[]
    import re
    for row in graph["condition_evidence_relations"]:
        match=re.search(r"_node_(\d+)_(yes|no)$",row["condition_id"])
        if match is None or int(match[1]) not in known: continue
        key=row["condition_id"]
        if key in seen: continue
        seen.add(key); expected="satisfied" if match[2]==known[int(match[1])] else "not_satisfied"
        rows.append({"condition_id":key,"expected":expected,"predicted":row["status"],"correct":row["status"]==expected,
          "evidence":row.get("evidence",[]),"method":row.get("method"),"reason":row.get("reason")})
    result["evaluation"]["conditions"]=rows
    for head,row in result["heads"].items():
        row["key_intervention_correct_margin_delta"]=row["intervention"]["option_logits"][gold]-row["intervention"]["option_logits"][competitor]-(result["variants"]["triage_prompt_only"]["option_logits"][gold]-result["variants"]["triage_prompt_only"]["option_logits"][competitor])
        row["retained_head_correct_margin_effect"]=(result["variants"]["triage_prompt_only"]["option_logits"][gold]-result["variants"]["triage_prompt_only"]["option_logits"][competitor])-(row["ablated"]["option_logits"][gold]-row["ablated"]["option_logits"][competitor])
    ev=result["evaluation"]; full=result["variants"]["triage_full"]
    reasons=[]
    if not ev["key_rule_recalled_after"]: reasons.append("key_rule_retrieval_miss")
    elif ev["gold_rule_applicability"]!="applicable": reasons.append("key_rule_condition_matching_"+ev["gold_rule_applicability"])
    if not full["correct"]:
        if ev["gold_rule_applicability"]=="applicable": reasons.append("downstream_decision_error_despite_correct_rule")
        if full["decision_index"]==len(options)-1: reasons.append("insufficient_evidence_decision")
        if result["variants"]["oracle_rule"]["correct"]: reasons.append("oracle_guidance_recovers_correct_action")
        else: reasons.append("oracle_guidance_also_fails")
    if ev["gold_action_gate"]==0: reasons.append("correct_rule_has_zero_gate")
    ev["failure_causes"]=reasons

def run_case(backend,module,item,components,projections,cached=None):
    start=time.time(); case_id=item["case_id"]; options=options_for(item)
    label_ids=[backend.tokenizer.encode(r["label"],add_special_tokens=False) for r in options]
    if any(len(x)!=1 for x in label_ids): raise ValueError("Option label is not single token")
    label_ids=[x[0] for x in label_ids]
    rules=item["guideline"]["rules"]
    old_scores=([r["score"] for r in cached["old_question_only_scores"]] if cached else backend.score_pairs(item["question"],[r["text"] for r in rules]))
    old_indices=sorted(range(len(rules)),key=lambda i:(-old_scores[i],i))[:3]
    graph=cached["module1"] if cached else module.run(guideline=item["guideline"],patient=item["patient"],question=item["question"])
    result={"schema_version":"triage_algorithm_case.v1","case_id":case_id,"guideline_id":item["guideline_id"],
      "input_source":item["input_source"],"options":options,"module1":graph,"old_top3_rule_ids":[rules[i]["rule_id"] for i in old_indices],
      "old_question_only_scores":[{"rule_id":r["rule_id"],"score":s} for r,s in zip(rules,old_scores)],
      "variants":{},"heads":{},"normal_inference_reference_answers_used":False,"module1_reused":bool(cached),"pipeline_prompt_constraints_preserved":True,
      "decision_protocol":"Greedy single-token choice among ALL source guideline actions plus insufficient evidence; not free-text generation"}
    for scope,rules_arg in [("original",None),("original_full_guideline",rules)]:
        text,encoded=raw_prompt(backend,item,options,rules_arg)
        result["variants"][scope]=summarize_logits(logits(backend,encoded),options,label_ids,encoded)
    question=item["question"]+option_text(options)
    text,encoded,mapping=build_prompt_and_mapping(DecisionTokenizer(backend.tokenizer),item["patient"],question,graph)
    encoded=encoded.to(backend.model.device)
    if encoded["input_ids"].shape[1]>backend.max_input_tokens: raise ValueError("TRIAGE prompt budget exceeded")
    base=logits(backend,encoded)
    result["variants"]["triage_prompt_only"]=summarize_logits(base,options,label_ids,encoded)
    with GatedAttentionIntervention(backend.model,components,projections,graph,mapping) as hook:
        full=logits(backend,encoded)
    result["variants"]["triage_full"]=summarize_logits(full,options,label_ids,encoded)
    result["intervention_stats"]=hook.stats
    result["max_abs_full_logit_delta"]=float((full-base).abs().max())
    with GatedAttentionIntervention(backend.model,components,projections,zero_graph(graph),mapping) as disabled:
        z=logits(backend,encoded)
    result["zero_gate_control"]={"exact":torch.equal(base,z),"max_abs_logit_delta":float((base-z).abs().max())}
    if not result["zero_gate_control"]["exact"]: raise AssertionError("Zero gate changed original computation")
    with GatedAttentionIntervention(backend.model,components,projections,graph,mapping,context_strength=0.,prior_strength=0.):
        z=logits(backend,encoded)
    result["zero_strength_control"]={"exact":torch.equal(base,z),"max_abs_logit_delta":float((base-z).abs().max())}
    if not result["zero_strength_control"]["exact"]: raise AssertionError("Zero strength changed original computation")
    for h in components["selected_heads"]:
        pair=(h["layer"],h["head"]); one,proj=select_intervention_heads(components,projections,[pair])
        with GatedAttentionIntervention(backend.model,one,proj,graph,mapping) as single:
            value=logits(backend,encoded)
        with scale_query_heads(backend.model,[pair],scale=0.):
            ablated=logits(backend,encoded)
        result["heads"][str(pair[0])+":"+str(pair[1])]={"intervention":summarize_logits(value,options,label_ids,encoded),
            "ablated":summarize_logits(ablated,options,label_ids,encoded),"stats":single.stats}
    # Normal inference is now complete. Only here load offline reference.
    refs=json.loads((OUT/"references.json").read_text())
    reference=next(r for r in refs["samples"] if r["case_id"]==case_id)
    oracle=[r for r in rules if r["rule_id"]==reference["gold_rule_id"]]
    if len(oracle)!=1: raise ValueError("Expected one official oracle fragment")
    text,oracle_encoded=raw_prompt(backend,item,options,oracle)
    result["variants"]["oracle_rule"]=summarize_logits(logits(backend,oracle_encoded),options,label_ids,oracle_encoded)
    result["oracle_scope"]="Correct official rule fragment selected offline; no answer text/path labels; diagnostic only; no hook or gates"
    evaluate(result,reference)
    result["elapsed_seconds"]=round(time.time()-start,3)
    result["gpu_peak_allocated_gib"]=torch.cuda.max_memory_allocated()/1024**3
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--worker",type=int,required=True);parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--module1-cache-dir",type=Path)
    args=parser.parse_args(); OUT.mkdir(exist_ok=True,parents=True)
    torch.manual_seed(20261014); torch.set_num_threads(4)
    inputs=json.loads((OUT/"manifest.json").read_text())["samples"]
    selected=[r for i,r in enumerate(inputs) if i%args.workers==args.worker]
    backend=LocalBackend(); backend._load_qwen(); module=Module1(backend=backend)
    components=json.loads((ROOT/"results/module2/components.json").read_text())
    projections=torch.load(ROOT/"results/module3/projections.pt",map_location="cpu",weights_only=True)
    settings={"schema_version":"triage_algorithm_worker.v1","worker":args.worker,"physical_gpu":os.environ.get("CUDA_VISIBLE_DEVICES"),
      "model":backend.model_path,"seed":20261014,"module1_top_k":module.top_k,"context_strength":.1,"prior_strength":.1,
      "components_sha256":sha(ROOT/"results/module2/components.json"),"projection_sha256":sha(ROOT/"results/module3/projections.pt"),
      "manifest_sha256":sha(OUT/"manifest.json"),"code_sha256":{str(p.relative_to(ROOT)):sha(p) for folder in ("src/module1","src/module3","src/pipeline") for p in (ROOT/folder).glob("*.py")},
      "diagnostic_script_sha256":sha(Path(__file__)),"pipeline_prompt_constraints_preserved":True,
      "module1_cache_dir":str(args.module1_cache_dir) if args.module1_cache_dir else None,
      "torch":torch.__version__,"no_training":True,"offline_models":True,"cases":[]}
    write(OUT/("worker_"+str(args.worker)+".json"),settings)
    for item in selected:
        output=OUT/("case_"+item["case_id"].replace(":","_")+".json")
        if output.exists() and not args.module1_cache_dir:
            existing=json.loads(output.read_text())
            if existing.get("evaluation"): settings["cases"].append(item["case_id"]);continue
        cached=None
        if args.module1_cache_dir:
            cached=json.loads((args.module1_cache_dir/output.name).read_text())
        result=run_case(backend,module,item,components,projections,cached=cached); write(output,result)
        settings["cases"].append(item["case_id"]); write(OUT/("worker_"+str(args.worker)+".json"),settings)
        print("Completed "+item["case_id"]+" in "+str(result["elapsed_seconds"])+"s",flush=True)
    print("Worker complete",flush=True)
if __name__=="__main__": main()

