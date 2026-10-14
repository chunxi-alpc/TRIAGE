#!/usr/bin/env python3
"""Run two original MEGA-CDP train_val inputs through TRIAGE."""
import argparse
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_module1_smoke import expand_tree, load_inputs as original_smoke_inputs

SAMPLES=[(5,"For this patient, what management should be used for feeding difficulties and nutritional problems in dementia?")]

def load_inputs():
    config=json.loads((ROOT/"configs/datasets.json").read_text())
    folder=Path(config["MEGA-CDP"]["path"])/"train_val"
    wanted={gid for gid,_ in SAMPLES}
    cases,trees={},{}
    for index,line in enumerate((folder/"vignette_cases.jsonl").open(),1):
        r=json.loads(line)
        gid=r["guideline_id"]
        if gid in wanted and gid not in cases: cases[gid]=(r,index)
        if len(cases)==len(wanted): break
    for index,line in enumerate((folder/"decision_trees.jsonl").open(),1):
        r=json.loads(line)
        gid=r["guideline_id"]
        if gid in wanted: trees[gid]=(r,index)
        if len(trees)==len(wanted): break
    result=[]
    for gid,question in SAMPLES:
        case,case_line=cases[gid]
        tree,tree_line=trees[gid]
        rules=expand_tree(json.loads(tree["text"])["decision_tree"],gid)
        result.append({"guideline":{"rules":rules},"patient":case["text"],"question":question,
          "input_source":{"dataset":"MEGA-CDP","split":"train_val","guideline_id":gid,"filename":case["filename"],
            "patient":{"path":str(folder/"vignette_cases.jsonl"),"line":case_line,"field":"text"},
            "guideline":{"path":str(folder/"decision_trees.jsonl"),"line":tree_line,"field":"text.decision_tree",
              "all_tree_paths_used":True,"rule_count":len(rules)},
            "question":{"origin":"generic management question; not a reference answer"}}})
    return result+[original_smoke_inputs()[1]]
from src.pipeline import TriagePipeline

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--components",type=Path,default=ROOT/"results/module2/components.json")
    p.add_argument("--projections",type=Path,default=ROOT/"results/module3/projections.pt")
    p.add_argument("--output",type=Path,default=ROOT/"results/pipeline/smoke.json")
    p.add_argument("--module1-cache",type=Path,help="Explicitly reuse existing module1 smoke outputs")
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--max-new-tokens",type=int,default=384)
    p.add_argument("--limit",type=int,default=2)
    p.add_argument("--prior-control",action="store_true")
    args=p.parse_args()
    if not 1<=args.limit<=2: p.error("Minimal smoke supports 1-2 samples")
    inputs=load_inputs()[:args.limit]
    cached=json.loads(args.module1_cache.read_text())["samples"] if args.module1_cache else []
    pipe=TriagePipeline(args.components,args.projections,args.device,max_new_tokens=args.max_new_tokens)
    started=time.time()
    report={"run_kind":"minimal_three_module_real_data_smoke","dataset":"MEGA-CDP",
            "split":"train_val","single_qwen_model_instance":True,"reference_answers_used":False,
            "components_path":str(args.components),"projection_path":str(args.projections),"samples":[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for i,item in enumerate(inputs):
        graph=None
        if cached:
            matches=[r for r in cached if r.get("patient")==item["patient"] and r.get("question")==item["question"]]
            if len(matches)!=1:
                raise ValueError("Expected exactly one matching module1 cache entry for patient/question")
            graph=matches[0].get("result",matches[0].get("module1"))
            if graph is None: raise ValueError("Matched cache entry lacks module1 output")
        result=pipe.run(item["guideline"],item["patient"],item["question"],module1_output=graph,
                       input_source=item["input_source"],check_prior_control=args.prior_control)
        report["samples"].append(result)
        report["elapsed_seconds"]=round(time.time()-started,3)
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
        if not result["baseline"]["json_valid"] or not result["intervened"]["json_valid"]:
            raise AssertionError("Pipeline decision JSON is incomplete; output text preserved in report")
        print("Completed pipeline sample "+str(i+1),flush=True)
    print("Saved "+str(args.output),flush=True)
if __name__=="__main__": main()
