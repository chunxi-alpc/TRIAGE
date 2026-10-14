#!/usr/bin/env python3
"""Normal inference manifest contains no labels. Offline oracle is a separate pass."""
import argparse,json,os,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.module1.backend import LocalBackend
from src.acre_pilot.annotator import annotate,direct_decision
from src.acre_pilot.span_mapping import build_prompt,map_annotations,select_targets,options_for
from src.acre_pilot.attention_bias import AttentionBias

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n");tmp.replace(path)

def logits(backend,encoded):
    with torch.inference_mode(): return backend.model(**encoded,use_cache=False,logits_to_keep=1).logits[0,-1].float().cpu()

def run_case(backend,item,config,out,oracle=None,betas=None):
    start=time.time();options=options_for(item)
    token_ids=[backend.tokenizer.encode(r["label"],add_special_tokens=False) for r in options]
    if any(len(x)!=1 for x in token_ids): raise ValueError("option_label_not_single_token")
    ids=[x[0] for x in token_ids]
    prompt,encoded,mapping=build_prompt(backend.tokenizer,item);encoded=encoded.to(backend.model.device)
    if mapping["prompt_tokens"]>config["max_input_tokens"]: raise ValueError("decision_input_budget_exceeded")
    ann=annotate(backend,item,config,out/"annotations")
    mapped=map_annotations(item,ann["rules"],mapping)
    result={"case_id":item["case_id"],"guideline_id":item["guideline_id"],"pair_id":item.get("pair_id"),"options":options,"annotation_key":ann["cache_key"],"annotation_errors":ann["errors"],"annotations":mapped,"variants":{},"normal_reference_used":False,"base_prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest(),"prompt_tokens":mapping["prompt_tokens"],"prompt_character_fields":[{"field":list(k),"start":a,"end":b} for k,(a,b) in mapping["fields"].items()]}
    def score(name,inputs,targets=(),beta=0):
        with AttentionBias(backend.model,targets,beta,inputs["input_ids"].shape[1]-1) as bias:
            value=logits(backend,inputs)
        chosen=value[ids];index=int(chosen.argmax());lp=value.log_softmax(-1)[ids]
        result["variants"][name]={"prediction":options[index]["label"],"action":options[index]["action"],"option_logits":chosen.tolist(),"option_log_probabilities":lp.tolist(),"option_normalized_probabilities":chosen.softmax(0).tolist(),"attention":bias.stats,"prompt_tokens":inputs["input_ids"].shape[1],"prompt_sha256":hashlib.sha256((prompt if inputs is encoded else structured_prompt).encode()).hexdigest()}
        return value
    if oracle is None:
        base=score("B0",encoded)
        with AttentionBias(backend.model,[0],0,mapping["prompt_tokens"]-1): z=logits(backend,encoded)
        with AttentionBias(backend.model,[],1,mapping["prompt_tokens"]-1): e=logits(backend,encoded)
        result["identity_controls"]={"beta_zero_exact":torch.equal(base,z),"empty_exact":torch.equal(base,e)}
        if not all(result["identity_controls"].values()): raise AssertionError("identity_control_failed")
        applicable=[r for r in mapped if r["relevant"] and r["applicability"]=="applicable"]
        # Only exact full action matches permit a multi-rule decision interpretation.
        actions={s["text"] for r in applicable for s in r["mapped"]["action_spans"] if any(s["text"]==o["action"] for o in options[:-1])}
        conflict=len(applicable)>1 and (len(actions)!=1 or any(not r["mapped"]["action_spans"] for r in applicable))
        result["out_of_single_rule_scope"]=conflict
        for method in ("B1","B2","B3"):
            targets,errors=select_targets(mapped,mapping,method)
            if conflict: targets=[]
            result[method+"_target_errors"]=errors
            for beta in betas or config["betas"]: score(method+"@"+str(float(beta)),encoded,targets,beta)
        structured_prompt,structured_inputs,_=build_prompt(backend.tokenizer,item,structured=ann["rules"])
        structured_inputs=structured_inputs.to(backend.model.device)
        if structured_inputs["input_ids"].shape[1]>config["max_input_tokens"]: raise ValueError("structured_input_budget_exceeded")
        score("B4",structured_inputs)
        result["variants"]["B5"]={"prediction":direct_decision(mapped,options,item["guideline"]["rules"]),"abstention_allowed":True,"attention":{"enabled":False,"targets":[]}}
    else:
        if oracle.get("verified") is not True: raise ValueError("unverified_oracle_forbidden")
        oracle_mapped=map_annotations(item,oracle["rules"],mapping)
        if any(r["mapping_errors"] for r in oracle_mapped): raise ValueError("invalid_oracle_spans")
        targets,_=select_targets(oracle_mapped,mapping,"B3")
        result["oracle_annotations"]=oracle_mapped
        for beta in betas or config["betas"]: score("B6@"+str(float(beta)),encoded,targets,beta)
        result["oracle_diagnostic_only"]=True
    result["elapsed_seconds"]=time.time()-start;result["gpu_peak_allocated_gib"]=torch.cuda.max_memory_allocated()/1024**3
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument("--manifest",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--worker",type=int,default=0);p.add_argument("--workers",type=int,default=1);p.add_argument("--limit",type=int)
    p.add_argument("--oracle",type=Path);p.add_argument("--betas",type=float,nargs="+")
    a=p.parse_args();config=json.loads((ROOT/"configs/acre_pilot.json").read_text());torch.manual_seed(config["seed"]);torch.set_num_threads(4)
    samples=json.loads(a.manifest.read_text())["samples"]
    if a.limit: samples=samples[:a.limit]
    oracle=json.loads(a.oracle.read_text()) if a.oracle else None
    backend=LocalBackend();backend._load_qwen();a.output.mkdir(parents=True,exist_ok=True)
    config.update({"physical_gpu":os.environ.get("CUDA_VISIBLE_DEVICES"),"model":backend.model_path,"torch":torch.__version__,"manifest_sha256":hashlib.sha256(a.manifest.read_bytes()).hexdigest(),"worker":a.worker,"oracle_only":bool(oracle)})
    write(a.output/("worker_"+str(a.worker)+("_oracle" if oracle else "")+".json"),config)
    for i,item in enumerate(samples):
        if i%a.workers!=a.worker: continue
        path=a.output/("case_"+item["case_id"].replace(":","_")+("_oracle" if oracle else "")+".json")
        if path.exists(): continue
        result=run_case(backend,item,config,a.output,oracle.get(item["case_id"]) if oracle else None,a.betas)
        write(path,result)
    write(a.output/("done_"+str(a.worker)+("_oracle" if oracle else "")+".json"),{"complete":True})
if __name__=="__main__":main()
