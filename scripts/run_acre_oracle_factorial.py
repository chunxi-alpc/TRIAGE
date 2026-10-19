#!/usr/bin/env python3
"""Explicitly non-deployable E1. Never calls or writes the model annotator."""
import argparse, copy, datetime, hashlib, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.module1.backend import LocalBackend
from src.acre_pilot.attention_bias import AttentionBias
from src.acre_pilot.span_mapping import build_prompt,map_annotations,select_targets,options_for
from src.acre_pilot.oracle_diagnostic import *
from scripts.run_acre_pilot import logits

def checks_ready():
    check=read(OUT/"recompute_check.json")
    assert check["E0_passed"] and sha(OUT/"oracle_mask_plan.json")==check["plan_sha256"]
    assert sha(BASE/"development_manifest.json")==check["normal_inputs"]["manifest"]
    assert sha(BASE/"development_oracle.json")==check["oracle_mask_only_inputs"]["oracle"]
    assert sha(BASE/"development_references.json")==check["offline_evaluation_inputs"]["references"]
    for name,value in check["archive_sha256"].items():assert sha(OLD/name)==value

def score(backend,encoded,options,targets,beta,prompt_sha):
    ids=[backend.tokenizer.encode(o["label"],add_special_tokens=False) for o in options]
    assert all(len(x)==1 for x in ids)
    ids=[x[0] for x in ids]
    with AttentionBias(backend.model,targets,beta,encoded["input_ids"].shape[1]-1) as bias:
        value=logits(backend,encoded)
    chosen=value[ids];index=int(chosen.argmax())
    if targets and beta:
        assert len(bias.stats["layers"])==36
        assert all(s["query_heads"]==32 and s["kv_heads"]==8 and s["active_query_positions"]==[encoded["input_ids"].shape[1]-1] for s in bias.stats["layers"].values())
    return {"prediction":options[index]["label"],"action":options[index]["action"],
            "option_logits":chosen.tolist(),"option_log_probabilities":value.log_softmax(-1)[ids].tolist(),
            "option_normalized_probabilities":chosen.softmax(0).tolist(),"attention":bias.stats,
            "prompt_sha256":prompt_sha,"prompt_tokens":encoded["input_ids"].shape[1]},value

def get_case(backend,item,c,oracle,plan):
    prompt,encoded,mapping=build_prompt(backend.tokenizer,item)
    h=hashlib.sha256(prompt.encode()).hexdigest()
    assert h==c["base_prompt_sha256"]==plan["base_prompt_sha256"]
    assert digest(oracle["rules"])==plan["oracle_annotation_sha256"]
    mapped=map_annotations(item,oracle["rules"],mapping)
    assert not any(r["mapping_errors"] for r in mapped)
    # Target ID is only used within this explicit, offline Oracle diagnostic.
    ref=read(BASE/"development_references.json")[item["case_id"]]
    rebuilt={}
    for scope,rows in [("",mapped),("-target",[r for r in mapped if r["rule_id"]==ref["target_rule_id"]])]:
        for method,strategy in [("OF","B2"),("OS","B3")]:
            t,errors=select_targets(rows,mapping,strategy);assert not errors
            rebuilt[method+scope]=t
    assert rebuilt==plan["targets"]
    assert options_for(item)==c["options"]
    return encoded.to(backend.model.device),h

def main():
    p=argparse.ArgumentParser();p.add_argument("--smoke",action="store_true")
    p.add_argument("--worker",type=int,default=0);p.add_argument("--workers",type=int,default=6);a=p.parse_args()
    checks_ready()
    torch.manual_seed(SEED);torch.set_num_threads(4)
    items,cases,_=normal_inputs();oracle,_=oracle_inputs()
    plans={r["case_id"]:r for r in read(OUT/"oracle_mask_plan.json")}
    backend=LocalBackend();start=time.time();backend._load_qwen()
    run=OUT/("smoke" if a.smoke else "workers");run.mkdir(parents=True,exist_ok=True)
    if a.smoke:
        ordered=sorted(items,key=lambda i:(plans[i["case_id"]]["prompt_tokens"],i["case_id"]))
        selected=[ordered[i] for i in [0,1,38,39,78,79]]
        results=[]
        for item in selected:
            cid=item["case_id"];c=cases[cid];plan=plans[cid]
            enc,h=get_case(backend,item,c,oracle[cid],plan)
            v,base=score(backend,enc,c["options"],[],0,h)
            assert v["option_logits"]==c["variants"]["B0"]["option_logits"]
            _,z=score(backend,enc,c["options"],plan["targets"]["OF"],0,h)
            _,e=score(backend,enc,c["options"],[],2,h)
            assert torch.equal(base,z) and torch.equal(base,e)
            for beta in [1.,2.]:
                v,_=score(backend,enc,c["options"],plan["targets"]["OS"],beta,h)
                old=c["variants"]["B6@"+str(beta)]
                assert v["option_logits"]==old["option_logits"],(cid,beta,"B6 logit mismatch")
                assert v["option_log_probabilities"]==old["option_log_probabilities"]
                v["recomputed_for_reuse_validation"]=True
                write(run/(cid+"_OS_"+str(beta)+".json"),v)
            results.append({"case_id":cid,"prompt_tokens":enc["input_ids"].shape[1],
                            "B0_and_B6_logits_exact":True,"zero_and_empty_full_vocabulary_exact":True})
        write(OUT/"smoke_checks.json",{"passed":True,"cases":results,"case_selection":"six fixed prompt-length order statistics: 0,1,38,39,78,79; chosen without inspecting new outcomes",
              "seconds":time.time()-start,"peak_allocated_gib":torch.cuda.max_memory_allocated()/1024**3,
              "peak_reserved_gib":torch.cuda.max_memory_reserved()/1024**3,"physical_gpu":os.environ.get("CUDA_VISIBLE_DEVICES"),
              "operator_checks":"operator_gpu_checks.json","code_fingerprints":code_fingerprints()})
        return
    assert read(OUT/"smoke_checks.json")["passed"]
    assert read(OUT/"operator_gpu_checks.json")["beta_zero_exact"]
    marker=run/("done_"+str(a.worker)+".json")
    if marker.exists():raise ValueError("Completed worker directory is immutable; choose a new run for reruns")
    write(run/("worker_"+str(a.worker)+".json"),{"worker":a.worker,"physical_gpu":os.environ.get("CUDA_VISIBLE_DEVICES"),
        "start_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"config_sha256":sha(ROOT/"configs/acre_oracle_switch.json"),
        "code_fingerprints":code_fingerprints(),"model":backend.model_path,"torch":torch.__version__})
    n=0;forward_calls=0;reused=0
    for index,item in enumerate(items):
        if index%a.workers!=a.worker:continue
        cid=item["case_id"];c=cases[cid];plan=plans[cid]
        path=run/("case_"+cid+".json")
        if path.exists():raise ValueError("Refuse overwrite of an existing case")
        before=time.time();enc,h=get_case(backend,item,c,oracle[cid],plan)
        basev,base=score(backend,enc,c["options"],[],0,h);forward_calls+=1
        assert basev["option_logits"]==c["variants"]["B0"]["option_logits"]
        _,zero=score(backend,enc,c["options"],plan["targets"]["OF"],0,h)
        _,empty=score(backend,enc,c["options"],[],2,h);forward_calls+=2
        assert torch.equal(base,zero) and torch.equal(base,empty)
        variants={"O0":copy.deepcopy(c["variants"]["B0"])};cache={}
        for beta in [2.,1.]:
            old=c["variants"]["B6@"+str(beta)]
            variants["OS@"+str(beta)]=copy.deepcopy(old)
            variants["OS@"+str(beta)]["reused_from"]="revision3 B6; all-case masks replayed; representative GPU logits exact"
            cache[(tuple(plan["targets"]["OS"]),beta)]=variants["OS@"+str(beta)]
            for method in ["OF","OF-target","OS-target"]:
                targets=plan["targets"][method];key=(tuple(targets),beta)
                if key in cache:
                    value=copy.deepcopy(cache[key]);value["same_mask_reuse"]=True;reused+=1
                elif not targets:
                    value=copy.deepcopy(basev);value["same_mask_reuse"]=True;reused+=1
                    value["attention"]={"enabled":False,"targets":[],"beta":beta,"layers":{},"decision_start":enc["input_ids"].shape[1]-1}
                else:
                    value,_=score(backend,enc,c["options"],targets,beta,h);forward_calls+=1;cache[key]=value
                value["oracle_diagnostic_only"]=True
                variants[method+"@"+str(beta)]=value
        result={"case_id":cid,"guideline_id":item["guideline_id"],"pair_id":item["pair_id"],"options":c["options"],
            "variants":variants,"targets":plan["targets"],"base_prompt_sha256":h,
            "oracle_annotation_sha256":plan["oracle_annotation_sha256"],
            "same_oracle_source_for_fixed_and_switch":True,"reference_text_in_prompt":False,
            "reference_usage":"non-deployable masks only; references used here only for prespecified target rule ID",
            "identity_controls":{"beta_zero_full_logits_exact":True,"empty_full_logits_exact":True,"old_B0_options_exact":True},
            "seconds":time.time()-before,"physical_gpu":os.environ.get("CUDA_VISIBLE_DEVICES"),
            "peak_allocated_gib":torch.cuda.max_memory_allocated()/1024**3}
        write(path,result);n+=1
    write(marker,{"complete":True,"worker":a.worker,"cases":n,"seconds_including_model_load":time.time()-start,
          "model_forward_calls":forward_calls,"identical_mask_reuses":reused,
          "peak_allocated_gib":torch.cuda.max_memory_allocated()/1024**3,
          "peak_reserved_gib":torch.cuda.max_memory_reserved()/1024**3})
if __name__=="__main__":main()
