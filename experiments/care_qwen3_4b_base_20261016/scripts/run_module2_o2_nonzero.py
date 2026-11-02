#!/usr/bin/env python3
"""Exploratory O2 sensitivity: select best NONZERO setting using saved dev data.

Designed after observing that the primary dev policy selected epsilon=0.
Preserves that primary result and never uses held-out labels for parameter choice.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from run_module2_followup import prepare_rows, choose, read_jsonl, score, record
from module2_subspaces import (
    attention_bias, bias_support, collect, key_edit, logit_delta, save_json, sha256,
)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root",type=Path,required=True)
    p.add_argument("--artifacts",type=Path,default=Path("/home/gcx/care_20261016/CARE_Codex_Pilot"))
    args=p.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in {"0","1","2","3","6","7"}
    start=time.time()
    source=args.run_root/"o2"
    folder=source/"fold0"
    old=read_jsonl(folder/"predictions.jsonl")
    dev=[r for r in old if r["role"]=="dev" and r["mode"]=="amplify" and r["epsilon"]>0]
    meta=json.loads((folder/"operators.json").read_text())
    strengths=sorted({r["epsilon"] for r in dev})
    selected=choose(dev,meta,strengths)
    assert all(x["epsilon"]>0 for x in selected.values())
    output=args.run_root/"o2_nonzero"
    output.mkdir(exist_ok=True)
    save_json(output/"selection.json",selected)
    operators=torch.load(folder/"operators.pt",weights_only=True)
    model_path=args.artifacts/"models/Qwen3-4B-Base"
    fingerprint=json.loads((source/"run_fingerprint.json").read_text())
    assert {x.name:sha256(x) for x in model_path.glob("*.safetensors")}==fingerprint["model_files"]
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation="sdpa").to("cuda").eval()
    rows=[r for r in prepare_rows(args.artifacts,"o2") if r["split"]=="test" and r["state"] in ("RA","RN")]
    baseline={r["case_id"]:r for r in old if r["role"]=="test" and r["mode"]=="baseline"}
    locations=sorted({tuple(meta[x["config"]]["location"]) for x in selected.values()})
    records=[]
    for row in rows:
        _,full,idx,probs=collect(model,tok,row,locations,capture_q=True)
        assert max(abs(x-y) for x,y in zip(probs,baseline[row["case_id"]]["candidate_probabilities"]))<1e-7
        records.append(record(row,probs,0,"test","baseline",0.,"baseline"))
        for method,setting in selected.items():
            cid,epsilon=setting["config"],setting["epsilon"]
            location=tuple(meta[cid]["location"])
            op=operators[cid]
            with key_edit(model,idx,location,op,epsilon) as change:
                probs=score(model,tok,row)
            records.append(record(row,probs,0,"test",cid,epsilon,"amplify",change))
            delta=logit_delta(model,full[location[0]],idx,location,op,epsilon)
            support=bias_support(delta.shape[-1],idx,location,delta.device)
            attention_bias._support=support
            rms=float(delta[support].square().mean().sqrt())
            probs=score(model,tok,row,attention_bias(model,location,delta,uniform=True))
            records.append(record(row,probs,0,"test",cid,epsilon,"uniform_bias",dict(logit_rms=rms)))
    with (output/"predictions.jsonl").open("w") as stream:
        for row in records:
            stream.write(json.dumps(row)+"\n")
    save_json(output/"complete.json",dict(
        status="completed_real_model_inference",scope="exploratory nonzero O2 sensitivity; primary zero-strength result unchanged",
        selection_uses_test_labels=False,design_after_primary_test_inspection=True,
        records=len(records),test_cases=len(rows),actual_model_forwards=len(rows)*15,
        seconds=time.time()-start,physical_gpu=os.environ["CUDA_VISIBLE_DEVICES"],
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        source_fingerprint=fingerprint,script_sha256=sha256(Path(__file__))))
if __name__=="__main__":
    with torch.inference_mode():
        main()
